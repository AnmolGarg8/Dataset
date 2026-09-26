"""
End-to-end training pipeline.

Usage (from code/business_entity_resolution/ directory):
    python -m src.train --train-dir ../../dataset/train --out-model output/model.joblib

Steps:
  1. Load training source files + ground truth
  2. Preprocess all records
  3. Split S1 entities into train/val (by entity, not by pair)
  4. Generate blocking candidates for train+val S1 entities
  5. Build positive/negative pair labels from ground truth
  6. Extract pairwise features
  7. Train LightGBM + tune threshold on val split
  8. Evaluate on validation set
  9. Save model
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, Set

import numpy as np
import pandas as pd

# Add parent to path for -m execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.preprocessing import preprocess_dataframe
from src.blocking import generate_candidates
from src.features import extract_features_batch, df_to_record_dict, FEATURE_NAMES
from src.model import train_model, save_model
from src.evaluate import macro_f_beta, detailed_evaluation
from src.utils import (
    setup_logging, read_source_tsv, read_ground_truth,
    parse_id_list, ensure_dir,
)

logger = logging.getLogger(__name__)


def _parse_ground_truth(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Convert ground truth DataFrame to {s1_id: set(matched_ids)}."""
    gt = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        ids = parse_id_list(row.get("matched_entity_ids", ""))
        gt[s1_id] = set(ids)
    return gt


def _build_labels(
    pair_ids: list,
    ground_truth: Dict[str, Set[str]],
) -> np.ndarray:
    """Create binary labels: 1 if (s1, cand) is a true match, 0 otherwise."""
    labels = np.zeros(len(pair_ids), dtype=np.float32)
    for i, (s1_id, cand_id) in enumerate(pair_ids):
        if cand_id in ground_truth.get(s1_id, set()):
            labels[i] = 1.0
    return labels


def main():
    parser = argparse.ArgumentParser(description="Train entity resolution model")
    parser.add_argument("--train-dir", required=True, help="Path to training data directory")
    parser.add_argument("--out-model", default="output/model.joblib", help="Output model path")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Optional: fraction of S1 entities to use (for quick testing)")
    args = parser.parse_args()

    setup_logging()
    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────────
    logger.info("Loading training data...")
    s1_df = read_source_tsv(os.path.join(args.train_dir, "train_source1.tsv"))
    s2_df = read_source_tsv(os.path.join(args.train_dir, "train_source2.tsv"))
    s3_df = read_source_tsv(os.path.join(args.train_dir, "train_source3.tsv"))
    gt_df = read_ground_truth(os.path.join(args.train_dir, "train_ground_truth.tsv"))

    logger.info(f"S1: {len(s1_df):,}, S2: {len(s2_df):,}, S3: {len(s3_df):,}, GT: {len(gt_df):,}")

    # Optional sampling for quick iteration
    if args.sample_frac:
        n_sample = max(int(len(s1_df) * args.sample_frac), 100)
        s1_sample_ids = set(s1_df["entity_id"].sample(n=n_sample, random_state=CFG.seed))
        s1_df = s1_df[s1_df["entity_id"].isin(s1_sample_ids)].copy()
        gt_df = gt_df[gt_df["source1_entity_id"].isin(s1_sample_ids)].copy()

        # Collect all true-match S2/S3 IDs so we keep them
        true_match_ids = set()
        for _, row in gt_df.iterrows():
            ids = parse_id_list(row.get("matched_entity_ids", ""))
            true_match_ids.update(ids)

        # Filter S2/S3: keep all true matches + a random sample of others
        countries = set(s1_df["country"].unique())
        s2_df = s2_df[s2_df["country"].isin(countries)].copy()
        s3_df = s3_df[s3_df["country"].isin(countries)].copy()

        s2_match = s2_df[s2_df["entity_id"].isin(true_match_ids)]
        s3_match = s3_df[s3_df["entity_id"].isin(true_match_ids)]
        n_neg = min(len(true_match_ids) * 10, 100_000)
        s2_neg = s2_df[~s2_df["entity_id"].isin(true_match_ids)].sample(
            n=min(n_neg, len(s2_df) - len(s2_match)), random_state=CFG.seed
        )
        s3_neg = s3_df[~s3_df["entity_id"].isin(true_match_ids)].sample(
            n=min(n_neg, len(s3_df) - len(s3_match)), random_state=CFG.seed
        )
        s2_df = pd.concat([s2_match, s2_neg], ignore_index=True)
        s3_df = pd.concat([s3_match, s3_neg], ignore_index=True)

        logger.info(
            f"Sampled to {len(s1_df)} S1, {len(s2_df)} S2, {len(s3_df)} S3 "
            f"({len(true_match_ids)} true matches kept)"
        )

    # ── 2. Parse ground truth ─────────────────────────────────────
    logger.info("Parsing ground truth...")
    ground_truth = _parse_ground_truth(gt_df)

    # ── 3. Leak-Free Entity-Level Split (85% Train, 15% Holdout) ─
    from sklearn.model_selection import train_test_split
    logger.info("Splitting Source-1 entities into 85% Train and 15% Holdout (stratified by country)...")
    train_s1_ids, holdout_s1_ids = train_test_split(
        s1_df["entity_id"].values,
        test_size=0.15,
        random_state=CFG.seed,
        stratify=s1_df["country"].values,
    )
    train_s1_set = set(train_s1_ids)
    holdout_s1_set = set(holdout_s1_ids)

    train_s1_df = s1_df[s1_df["entity_id"].isin(train_s1_set)].copy()
    holdout_s1_df = s1_df[s1_df["entity_id"].isin(holdout_s1_set)].copy()

    train_gt = {s1: ground_truth[s1] for s1 in train_s1_set if s1 in ground_truth}
    holdout_gt = {s1: ground_truth[s1] for s1 in holdout_s1_set if s1 in ground_truth}

    logger.info(
        f"Train entities: {len(train_s1_df):,} S1 ({sum(len(v) for v in train_gt.values()):,} true links), "
        f"Holdout entities: {len(holdout_s1_df):,} S1 ({sum(len(v) for v in holdout_gt.values()):,} true links)"
    )

    # ── 4. Preprocess ─────────────────────────────────────────────
    logger.info("Preprocessing records...")
    train_s1_df = preprocess_dataframe(train_s1_df, CFG.legal_suffixes, CFG.address_abbrevs)
    holdout_s1_df = preprocess_dataframe(holdout_s1_df, CFG.legal_suffixes, CFG.address_abbrevs)
    s2_df = preprocess_dataframe(s2_df, CFG.legal_suffixes, CFG.address_abbrevs)
    s3_df = preprocess_dataframe(s3_df, CFG.legal_suffixes, CFG.address_abbrevs)
    logger.info(f"Preprocessing complete: {time.time() - t0:.1f}s")

    # ── 5. Blocking (Separate for Train and Holdout) ───────────────
    logger.info("Generating blocking candidates for TRAIN entities...")
    train_candidates = generate_candidates(train_s1_df, s2_df, s3_df)

    # Train blocking recall
    tr_in_cand = sum(len(train_gt[s1] & train_candidates.get(s1, set())) for s1 in train_gt)
    tr_total = sum(len(v) for v in train_gt.values())
    tr_recall = tr_in_cand / tr_total if tr_total > 0 else 0.0
    logger.info(f"Train blocking recall: {tr_recall:.4f} ({tr_in_cand:,}/{tr_total:,})")

    logger.info("Generating blocking candidates for HOLDOUT entities...")
    holdout_candidates = generate_candidates(holdout_s1_df, s2_df, s3_df)

    # Holdout blocking recall
    ho_in_cand = sum(len(holdout_gt[s1] & holdout_candidates.get(s1, set())) for s1 in holdout_gt)
    ho_total = sum(len(v) for v in holdout_gt.values())
    ho_recall = ho_in_cand / ho_total if ho_total > 0 else 0.0
    logger.info(f"Holdout blocking recall: {ho_recall:.4f} ({ho_in_cand:,}/{ho_total:,})")

    # ── 6. Build record dicts for fast lookup ─────────────────────
    logger.info("Building record lookup dicts...")
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    pool_records = df_to_record_dict(pool_df)
    del pool_df, s2_df, s3_df  # free memory

    train_s1_records = df_to_record_dict(train_s1_df)
    holdout_s1_records = df_to_record_dict(holdout_s1_df)
    holdout_countries = {row["entity_id"]: row.get("country", "Unknown") for _, row in holdout_s1_df.iterrows()}

    # ── 7. Extract features SEPARATELY ────────────────────────────
    logger.info("Extracting features for TRAIN pairs...")
    train_pair_ids, X_train = extract_features_batch(train_s1_records, pool_records, train_candidates)
    y_train = _build_labels(train_pair_ids, train_gt)
    logger.info(f"Train data: {X_train.shape[0]:,} pairs ({int(y_train.sum()):,} pos, {int((1-y_train).sum()):,} neg)")

    logger.info("Extracting features for HOLDOUT pairs...")
    holdout_pair_ids, X_holdout = extract_features_batch(holdout_s1_records, pool_records, holdout_candidates)
    y_holdout = _build_labels(holdout_pair_ids, holdout_gt)
    logger.info(f"Holdout data: {X_holdout.shape[0]:,} pairs ({int(y_holdout.sum()):,} pos, {int((1-y_holdout).sum()):,} neg)")

    # ── 8. Train model with Leak-Free Holdout Evaluation ──────────
    logger.info("Training LightGBM on TRAIN entities and tuning threshold on HOLDOUT entities...")
    model, threshold = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_holdout,
        y_val=y_holdout,
        val_pair_ids=holdout_pair_ids,
        val_ground_truth=holdout_gt,
        val_s1_ids=holdout_s1_set,
    )

    # ── 9. Save model IMMEDIATELY ─────────────────────────────────
    ensure_dir(os.path.dirname(args.out_model))
    save_model(model, threshold, args.out_model)
    logger.info(f"Model successfully saved to {args.out_model}")

    # ── 10. Honest, Leak-Free Holdout Evaluation ──────────────────
    logger.info("Evaluating model on GENUINE HOLDOUT entities (never seen in training)...")
    from src.model import predict_matches
    from src.evaluate import country_breakdown_evaluation

    holdout_predictions = predict_matches(
        model=model,
        X=X_holdout,
        pair_ids=holdout_pair_ids,
        threshold=threshold,
        all_s1_ids=holdout_s1_set,
    )

    eval_results = detailed_evaluation(holdout_predictions, holdout_gt, beta=0.5)
    logger.info("=" * 60)
    logger.info(f"HONEST HOLDOUT MACRO F_0.5: {eval_results['macro_f_beta']:.4f}")
    logger.info("=" * 60)
    for k, v in eval_results.items():
        logger.info(f"  {k}: {v}")

    # Per-country breakdown on genuine holdout entities
    country_results = country_breakdown_evaluation(holdout_predictions, holdout_gt, holdout_countries, beta=0.5)
    logger.info("-" * 60)
    logger.info("Holdout F_0.5 by Country:")
    for country, c_res in country_results.items():
        logger.info(
            f"  [{country}] Macro F_0.5: {c_res['macro_f_beta']:.4f}, "
            f"Precision: {c_res['precision_avg']:.4f}, Recall: {c_res['recall_avg']:.4f} "
            f"(N={c_res['total_entities']:,})"
        )
    # ── 11. Final Model Retraining on ALL Labeled Data ────────────
    logger.info("=" * 60)
    logger.info("Retraining FINAL production model on Train + Holdout combined (100% of labeled data)...")
    logger.info(f"Locking optimal holdout decision threshold: {threshold:.4f}")
    X_full = np.vstack([X_train, X_holdout])
    y_full = np.concatenate([y_train, y_holdout])
    final_model, _ = train_model(
        X_train=X_full,
        y_train=y_full,
        fixed_threshold=threshold,
    )
    save_model(final_model, threshold, args.out_model)
    logger.info(f"Final production model successfully saved to {args.out_model} (threshold={threshold:.4f})")
    logger.info("=" * 60)

    logger.info(f"Pipeline complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
