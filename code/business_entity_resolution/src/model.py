"""
LightGBM binary classifier for entity match scoring.

Trains on (positive, hard-negative) pairs from the training ground truth,
tunes the decision threshold to maximise macro-averaged F_0.5 on a
validation split, and provides prediction methods for test inference.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from .config import CFG
from .evaluate import f_beta_score
from .features import FEATURE_NAMES

logger = logging.getLogger(__name__)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    val_pair_ids: Optional[List[Tuple[str, str]]] = None,
    val_ground_truth: Optional[Dict[str, Set[str]]] = None,
    val_s1_ids: Optional[Set[str]] = None,
    fixed_threshold: Optional[float] = None,
) -> Tuple[lgb.LGBMClassifier, float]:
    """Train a LightGBM classifier and tune the match threshold.

    If X_val, y_val, val_pair_ids, and val_ground_truth are provided,
    they are used for early stopping and leak-free threshold tuning.
    Otherwise, if fixed_threshold is provided, it is returned directly.

    Args:
        X_train: Feature matrix for training pairs.
        y_train: Binary labels for training pairs.
        X_val: Feature matrix for held-out validation pairs.
        y_val: Binary labels for held-out validation pairs.
        val_pair_ids: List of (s1_id, cand_id) for validation pairs.
        val_ground_truth: {s1_id: set(true_match_ids)} for holdout entities.
        val_s1_ids: Complete set of holdout S1 entity IDs.
        fixed_threshold: Pre-determined threshold (if skipping tuning).

    Returns:
        (model, best_threshold)
    """
    logger.info(f"Training data: {X_train.shape[0]:,} pairs, {int(y_train.sum()):,} positives, "
                f"{int((~y_train.astype(bool)).sum()):,} negatives")

    model = lgb.LGBMClassifier(**CFG.lgbm_params)

    if X_val is not None and y_val is not None:
        logger.info(f"Holdout validation split: {len(X_val):,} pairs, {int(y_val.sum()):,} positives")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=True),
                lgb.log_evaluation(period=50),
            ],
        )
    else:
        model.fit(X_train, y_train)

    # ── Feature importance ─────────────────────────────────────
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    logger.info("Top 10 features:")
    for i in sorted_idx[:10]:
        logger.info(f"  {FEATURE_NAMES[i]}: {importances[i]}")

    # ── Threshold tuning on held-out entities ───────────────────
    if X_val is not None and val_pair_ids is not None and val_ground_truth is not None:
        val_probs = model.predict_proba(X_val)[:, 1]
        best_threshold = _tune_threshold(
            val_probs, val_pair_ids, val_ground_truth, all_s1_ids=val_s1_ids
        )
    elif fixed_threshold is not None:
        best_threshold = fixed_threshold
    else:
        best_threshold = 0.35

    logger.info(f"Best threshold: {best_threshold:.4f}")
    return model, best_threshold


def _tune_threshold(
    probs: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: Optional[Set[str]] = None,
    max_rank: int = 8,
    margin: float = 0.15,
) -> float:
    """Find the threshold that maximises macro F_0.5 on held-out entities.

    Scans thresholds from 0.20 to 0.75 in steps of 0.02.
    Evaluates across all entities in all_s1_ids (including singletons and
    entities with 0 candidates).
    """
    from collections import defaultdict

    s1_eval_ids = set(all_s1_ids) if all_s1_ids is not None else {s1_id for s1_id, _ in pair_ids}

    # Group candidate predictions by s1_id for fast evaluation
    entity_pairs = defaultdict(list)
    for (s1_id, cand_id), prob in zip(pair_ids, probs):
        entity_pairs[s1_id].append((cand_id, float(prob)))

    best_f05 = -1.0
    best_t = 0.35

    for t in np.arange(0.20, 0.76, 0.02):
        predictions: Dict[str, Set[str]] = {s1: set() for s1 in s1_eval_ids}
        for s1_id, cand_list in entity_pairs.items():
            if s1_id not in predictions:
                continue
            valid = [(c, p) for c, p in cand_list if p >= t]
            if valid:
                valid.sort(key=lambda x: x[1], reverse=True)
                top_p = valid[0][1]
                predictions[s1_id] = {c for c, p in valid[:max_rank] if p >= top_p - margin}

        scores = [f_beta_score(predictions[s1_id], ground_truth.get(s1_id, set()), beta=0.5) for s1_id in s1_eval_ids]
        f05 = float(np.mean(scores)) if scores else 0.0
        if f05 > best_f05:
            best_f05 = f05
            best_t = t

    logger.info(f"Threshold tuning: best F_0.5 = {best_f05:.4f} at threshold = {best_t:.4f}")
    return float(best_t)


def predict_matches(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    threshold: float,
    all_s1_ids: Set[str],
    max_rank: int = 8,
    margin: float = 0.15,
) -> Dict[str, Set[str]]:
    """Apply the model to candidate pairs and return final matches.

    Args:
        model: Trained LightGBM model.
        X: Feature matrix for candidate pairs.
        pair_ids: (s1_id, cand_id) for each row.
        threshold: Classification threshold.
        all_s1_ids: All S1 entity IDs (to ensure complete output).
        max_rank: Maximum matches allowed per entity (default: 8).
        margin: Maximum drop in probability from top candidate.

    Returns:
        {s1_id: set(matched_ids)}  — empty set for singletons.
    """
    from collections import defaultdict
    predictions: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}

    if len(X) == 0:
        logger.info("No candidate pairs to score — all entities are singletons")
        return predictions

    # Score in batches to manage memory
    batch_size = 500_000
    n_batches = (len(X) + batch_size - 1) // batch_size
    logger.info(f"Scoring {len(X):,} pairs in {n_batches} batches (threshold={threshold:.4f}, max_rank={max_rank})")

    entity_matches = defaultdict(list)
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(X))
        probs = model.predict_proba(X[start:end])[:, 1]

        for j, prob in enumerate(probs):
            if prob >= threshold:
                s1_id, cand_id = pair_ids[start + j]
                entity_matches[s1_id].append((cand_id, float(prob)))

        logger.info(f"  Batch {i + 1}/{n_batches}: scored {end - start:,} pairs")

    # Apply adaptive rank and margin pruning
    for s1_id, cands in entity_matches.items():
        cands.sort(key=lambda x: x[1], reverse=True)
        top_prob = cands[0][1]
        predictions[s1_id] = {c for c, p in cands[:max_rank] if p >= top_prob - margin}

    matched_count = sum(1 for v in predictions.values() if v)
    logger.info(
        f"Prediction complete: {matched_count:,} entities matched, "
        f"{len(predictions) - matched_count:,} singletons"
    )
    return predictions


def save_model(model: lgb.LGBMClassifier, threshold: float, path: str):
    """Save model and threshold together."""
    joblib.dump({"model": model, "threshold": threshold}, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str) -> Tuple[lgb.LGBMClassifier, float]:
    """Load model and threshold."""
    data = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return data["model"], data["threshold"]
