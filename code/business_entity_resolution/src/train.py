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

    # ── 3. Preprocess ─────────────────────────────────────────────
    logger.info("Preprocessing records...")
    s1_df = preprocess_dataframe(s1_df, CFG.legal_suffixes, CFG.address_abbrevs)
    s2_df = preprocess_dataframe(s2_df, CFG.legal_suffixes, CFG.address_abbrevs)
    s3_df = preprocess_dataframe(s3_df, CFG.legal_suffixes, CFG.address_abbrevs)
    logger.info(f"Preprocessing complete: {time.time() - t0:.1f}s")

    # ── 4. Blocking ───────────────────────────────────────────────
    logger.info("Generating blocking candidates...")
    candidates = generate_candidates(s1_df, s2_df, s3_df)

    # ── 5. Evaluate blocking recall ───────────────────────────────
    # Check what fraction of true matches appear in blocking candidates
    true_in_cand = 0
    true_total = 0
    for s1_id, true_ids in ground_truth.items():
        cand_ids = candidates.get(s1_id, set())
        true_total += len(true_ids)
        true_in_cand += len(true_ids & cand_ids)

    blocking_recall = true_in_cand / true_total if true_total > 0 else 0
    logger.info(f"Blocking recall: {blocking_recall:.4f} ({true_in_cand:,}/{true_total:,})")

    # ── 6. Build record dicts for fast lookup ─────────────────────
    logger.info("Building record lookup dicts...")
    s1_records = df_to_record_dict(s1_df)
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    pool_records = df_to_record_dict(pool_df)
    del pool_df  # free memory

    # ── 7. Extract features ───────────────────────────────────────
    logger.info("Extracting pairwise features...")
    pair_ids, X = extract_features_batch(s1_records, pool_records, candidates)

    # ── 8. Build labels ───────────────────────────────────────────
    y = _build_labels(pair_ids, ground_truth)
    logger.info(f"Labels: {int(y.sum()):,} positives, {int((1 - y).sum()):,} negatives")

    # ── 9. Train model ────────────────────────────────────────────
    model, threshold = train_model(X, y, pair_ids, ground_truth)

    # ── 10. Save model IMMEDIATELY ─────────────────────────────────
    ensure_dir(os.path.dirname(args.out_model))
    save_model(model, threshold, args.out_model)
    logger.info(f"Model successfully saved to {args.out_model}")

    # ── 11. Quick validation evaluation ───────────────────────────
    logger.info("Evaluating on validation sample...")
    from src.model import predict_matches
    from src.evaluate import country_breakdown_evaluation
    # Evaluate on a 6000 S1 entity subset to keep it fast
    val_s1_sample = list(s1_records.keys())[:min(6000, len(s1_records))]
    val_s1_set = set(val_s1_sample)
    val_pairs = [(s1, c) for s1, c in pair_ids if s1 in val_s1_set]
    val_gt = {s1: ground_truth.get(s1, set()) for s1 in val_s1_set}
    val_countries = {s1: s1_records[s1].get("country", "Unknown") for s1 in val_s1_set}

    if val_pairs:
        val_indices = [i for i, (s1, _) in enumerate(pair_ids) if s1 in val_s1_set]
        val_X = X[val_indices]
        predictions = predict_matches(model, val_X, val_pairs, threshold, val_s1_set)
        eval_results = detailed_evaluation(predictions, val_gt, beta=0.5)
        logger.info(f"Validation F_0.5 Overall: {eval_results['macro_f_beta']:.4f}")
        for k, v in eval_results.items():
            logger.info(f"  {k}: {v}")

        # Per-country breakdown
        country_results = country_breakdown_evaluation(predictions, val_gt, val_countries, beta=0.5)
        logger.info("Validation F_0.5 by Country:")
        for country, c_res in country_results.items():
            logger.info(f"  [{country}] Macro F_0.5: {c_res['macro_f_beta']:.4f}, Precision: {c_res['precision_avg']:.4f}, Recall: {c_res['recall_avg']:.4f} (N={c_res['total_entities']:,})")

    logger.info(f"Training complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
