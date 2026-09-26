"""
End-to-end test inference pipeline.
Memory-safe, high-performance country-partitioned streaming.

Usage (from code/business_entity_resolution/ directory):
    python -m src.predict --test-dir ../../dataset/test \
                          --model output/model.joblib \
                          --out-dir ../../output \
                          --max-candidates 30

Outputs:
  output/matching_results.tsv   (final matches scored on leaderboard)
  output/candidate_pairs.tsv    (blocking candidates for verification)
"""

import argparse
import gc
import logging
import os
import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.preprocessing import preprocess_dataframe
from src.blocking import generate_candidates
from src.features import extract_features_batch, df_to_record_dict
from src.model import load_model
from src.utils import (
    setup_logging, ensure_dir,
    write_matching_results, write_candidate_pairs,
)

logger = logging.getLogger(__name__)


def load_partition(file_path: str, country: str) -> pd.DataFrame:
    """Read only rows corresponding to `country` in chunks to keep memory minimal."""
    chunks = []
    for chunk in pd.read_csv(file_path, sep="\t", chunksize=250_000, dtype=str):
        matched = chunk[chunk["country"] == country]
        if len(matched) > 0:
            chunks.append(matched)
    if chunks:
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    return df


def score_country_candidates(
    model,
    threshold: float,
    s1_records: Dict[str, dict],
    pool_records: Dict[str, dict],
    candidates: Dict[str, Set[str]],
    batch_size_entities: int = 25_000,
) -> Dict[str, Set[str]]:
    """Score candidate pairs in entity-batches to strictly bound peak RAM."""
    predictions: Dict[str, Set[str]] = {s1: set() for s1 in s1_records}
    s1_keys = list(candidates.keys())
    total_entities = len(s1_keys)

    logger.info(f"Scoring candidates across {total_entities:,} entities in batches of {batch_size_entities:,}...")

    for start_idx in range(0, total_entities, batch_size_entities):
        end_idx = min(start_idx + batch_size_entities, total_entities)
        batch_keys = s1_keys[start_idx:end_idx]
        batch_cands = {k: candidates[k] for k in batch_keys if candidates[k]}

        if not batch_cands:
            continue

        pair_ids, X = extract_features_batch(s1_records, pool_records, batch_cands)
        if len(X) > 0:
            probs = model.predict_proba(X)[:, 1]
            from collections import defaultdict
            batch_entity_scores = defaultdict(list)
            for (s1_id, cand_id), prob in zip(pair_ids, probs):
                if prob >= threshold:
                    batch_entity_scores[s1_id].append((cand_id, float(prob)))

            for s1_id, scored_cands in batch_entity_scores.items():
                scored_cands.sort(key=lambda x: x[1], reverse=True)
                top_prob = scored_cands[0][1]
                predictions[s1_id] = {c for c, p in scored_cands[:8] if p >= top_prob - 0.15}

        del pair_ids, X
        gc.collect()

        logger.info(f"  Processed {end_idx:,}/{total_entities:,} entities")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate test predictions with country-partitioned streaming")
    parser.add_argument("--test-dir", required=True, help="Path to test data directory")
    parser.add_argument("--model", required=True, help="Path to trained model.joblib")
    parser.add_argument("--out-dir", required=True, help="Output directory for TSV files")
    parser.add_argument("--max-candidates", type=int, default=30, help="Candidate cap per entity (default: 30)")
    args = parser.parse_args()

    setup_logging()
    t0 = time.time()

    # Update candidate cap in CFG
    CFG.max_candidates_per_entity = args.max_candidates

    # ── 1. Load model ─────────────────────────────────────────────
    logger.info("Loading model...")
    model, threshold = load_model(args.model)
    logger.info(f"Model loaded. Decision threshold: {threshold:.4f}")

    # ── 2. Read full test S1 ──────────────────────────────────────
    s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    logger.info(f"Reading reference entities from {s1_path}...")
    full_s1_df = pd.read_csv(s1_path, sep="\t", dtype=str)
    all_test_s1_ids = list(full_s1_df["entity_id"])
    logger.info(f"Total reference entities in test_source1: {len(all_test_s1_ids):,}")

    # Process countries in ascending order of size (France -> US -> India)
    countries = sorted(list(full_s1_df["country"].unique()), key=lambda c: (full_s1_df["country"] == c).sum())
    logger.info(f"Test countries ordered by size: {countries}")

    all_predictions: Dict[str, Set[str]] = {eid: set() for eid in all_test_s1_ids}
    all_candidates: Dict[str, Set[str]] = {eid: set() for eid in all_test_s1_ids}

    s2_path = os.path.join(args.test_dir, "test_source2.tsv")
    s3_path = os.path.join(args.test_dir, "test_source3.tsv")

    # ── 3. Process country by country ─────────────────────────────
    for country in countries:
        c_start = time.time()
        logger.info(f"\n{'='*30} Processing Country: {country} {'='*30}")

        s1_c = full_s1_df[full_s1_df["country"] == country].copy()
        logger.info(f"Loading S2 and S3 partitions for {country}...")
        s2_c = load_partition(s2_path, country)
        s3_c = load_partition(s3_path, country)
        logger.info(f"  Records loaded: S1={len(s1_c):,}, S2={len(s2_c):,}, S3={len(s3_c):,}")

        logger.info(f"Preprocessing {country} records...")
        s1_c = preprocess_dataframe(s1_c, CFG.legal_suffixes, CFG.address_abbrevs)
        s2_c = preprocess_dataframe(s2_c, CFG.legal_suffixes, CFG.address_abbrevs)
        s3_c = preprocess_dataframe(s3_c, CFG.legal_suffixes, CFG.address_abbrevs)

        logger.info(f"Generating blocking candidates for {country} (cap={CFG.max_candidates_per_entity})...")
        cands_c = generate_candidates(s1_c, s2_c, s3_c)

        logger.info(f"Building lookup records for {country}...")
        s1_recs = df_to_record_dict(s1_c)
        pool_df = pd.concat([s2_c, s3_c], ignore_index=True)
        pool_recs = df_to_record_dict(pool_df)

        del s1_c, s2_c, s3_c, pool_df
        gc.collect()

        logger.info(f"Scoring {country} candidate pairs...")
        preds_c = score_country_candidates(
            model=model,
            threshold=threshold,
            s1_records=s1_recs,
            pool_records=pool_recs,
            candidates=cands_c,
            batch_size_entities=30_000,
        )

        # Merge into global outputs
        for eid, matches in preds_c.items():
            all_predictions[eid] = matches
        for eid, c_set in cands_c.items():
            all_candidates[eid] = c_set

        del s1_recs, pool_recs, cands_c, preds_c
        gc.collect()
        logger.info(f"Completed {country} in {time.time() - c_start:.1f}s")

    # ── 4. Write final output files ───────────────────────────────
    ensure_dir(args.out_dir)
    matching_path = os.path.join(args.out_dir, "matching_results.tsv")
    candidate_path = os.path.join(args.out_dir, "candidate_pairs.tsv")

    logger.info(f"\nWriting output files to {args.out_dir}...")
    write_matching_results(all_predictions, matching_path)
    write_candidate_pairs(all_candidates, candidate_path)

    # Summary statistics
    total_s1 = len(all_predictions)
    total_matched_entities = sum(1 for v in all_predictions.values() if v)
    total_singletons = total_s1 - total_matched_entities
    total_matches = sum(len(v) for v in all_predictions.values())
    total_candidates = sum(len(v) for v in all_candidates.values())

    logger.info(f"\n{'='*30} Inference Summary {'='*30}")
    logger.info(f"Total S1 entities processed: {total_s1:,}")
    logger.info(f"Entities with predicted matches: {total_matched_entities:,} ({total_matched_entities/total_s1*100:.2f}%)")
    logger.info(f"Entities predicted as singletons: {total_singletons:,} ({total_singletons/total_s1*100:.2f}%)")
    logger.info(f"Total candidate pairs: {total_candidates:,} (avg {total_candidates/total_s1:.1f}/entity)")
    logger.info(f"Total match links: {total_matches:,} (avg {total_matches/max(1, total_matched_entities):.2f}/matched entity)")
    logger.info(f"Total time elapsed: {time.time() - t0:.1f}s")
    logger.info(f"Saved: {matching_path}")
    logger.info(f"Saved: {candidate_path}")


if __name__ == "__main__":
    main()
