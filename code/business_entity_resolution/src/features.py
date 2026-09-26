"""
Pairwise similarity feature extraction.

Given an S1 record and a candidate S2/S3 record, compute a feature
vector that captures name similarity, address similarity, and
numeric overlap.  All features are lightweight scalar computations
using RapidFuzz (C-accelerated string metrics).
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────

def _jaccard_tokens(a: list, b: list) -> float:
    """Token-level Jaccard similarity."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _jaccard_ngrams(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa = {a[i : i + n] for i in range(max(len(a) - n + 1, 0))}
    sb = {b[i : i + n] for i in range(max(len(b) - n + 1, 0))}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _number_overlap(nums_a: set, nums_b: set) -> float:
    """Fraction of numbers in common (symmetric)."""
    if not nums_a and not nums_b:
        return 1.0
    if not nums_a or not nums_b:
        return 0.0
    return len(nums_a & nums_b) / len(nums_a | nums_b)


def _containment(a: set, b: set) -> float:
    """What fraction of a is contained in b."""
    if not a:
        return 1.0
    if not b:
        return 0.0
    return len(a & b) / len(a)


# ── Feature names (fixed order) ────────────────────────────────────

FEATURE_NAMES = [
    # Name features
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaccard_tokens",
    "name_jaccard_3gram",
    "name_levenshtein_norm",
    "name_len_diff",
    "name_first_token_match",
    "name_exact_match",
    # Address features
    "addr_jaro_winkler",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaccard_tokens",
    "addr_jaccard_3gram",
    "addr_levenshtein_norm",
    "addr_number_overlap",
    "addr_number_containment_s1",
    "addr_len_diff",
    "addr_exact_match",
    # Postal & Door Semantic Conflict Resolution
    "postal_code_status",
    "door_number_status",
    # Cross & Meta features
    "name_in_addr",
    "country_match",
]

NUM_FEATURES = len(FEATURE_NAMES)


# ── Main feature extraction ────────────────────────────────────────

def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    s1_name_tokens: list,
    s1_addr_tokens: list,
    s1_addr_numbers: set,
    s1_country: str,
    cand_name: str,
    cand_addr: str,
    cand_name_tokens: list,
    cand_addr_tokens: list,
    cand_addr_numbers: set,
    cand_country: str,
) -> np.ndarray:
    """Compute feature vector for a single (S1, candidate) pair.

    All inputs should be preprocessed (normalized, cleaned).
    Returns a numpy array of shape (NUM_FEATURES,).
    """
    feats = np.zeros(NUM_FEATURES, dtype=np.float32)

    # Defaults for empty
    s1_name = s1_name or ""
    s1_addr = s1_addr or ""
    cand_name = cand_name or ""
    cand_addr = cand_addr or ""
    s1_name_tokens = s1_name_tokens or []
    s1_addr_tokens = s1_addr_tokens or []
    cand_name_tokens = cand_name_tokens or []
    cand_addr_tokens = cand_addr_tokens or []
    s1_addr_numbers = s1_addr_numbers or set()
    cand_addr_numbers = cand_addr_numbers or set()

    # ── Name features ──────────────────────────────────────────
    if s1_name and cand_name:
        feats[0] = JaroWinkler.normalized_similarity(s1_name, cand_name)
        feats[1] = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
        feats[2] = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
        feats[3] = fuzz.partial_ratio(s1_name, cand_name) / 100.0
        feats[4] = _jaccard_tokens(s1_name_tokens, cand_name_tokens)
        feats[5] = _jaccard_ngrams(s1_name, cand_name, 3)
        max_len = max(len(s1_name), len(cand_name))
        feats[6] = 1.0 - (Levenshtein.distance(s1_name, cand_name) / max_len) if max_len > 0 else 1.0
        feats[7] = abs(len(s1_name) - len(cand_name)) / max(max_len, 1)
        # First-token match (company core word)
        if s1_name_tokens and cand_name_tokens:
            feats[8] = 1.0 if s1_name_tokens[0] == cand_name_tokens[0] else 0.0
        feats[9] = 1.0 if s1_name == cand_name else 0.0
    else:
        feats[0:10] = 0.0

    # ── Address features ───────────────────────────────────────
    if s1_addr and cand_addr:
        feats[10] = JaroWinkler.normalized_similarity(s1_addr, cand_addr)
        feats[11] = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        feats[12] = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
        feats[13] = _jaccard_tokens(s1_addr_tokens, cand_addr_tokens)
        feats[14] = _jaccard_ngrams(s1_addr, cand_addr, 3)
        max_len = max(len(s1_addr), len(cand_addr))
        feats[15] = 1.0 - (Levenshtein.distance(s1_addr, cand_addr) / max_len) if max_len > 0 else 1.0
        feats[16] = _number_overlap(s1_addr_numbers, cand_addr_numbers)
        feats[17] = _containment(s1_addr_numbers, cand_addr_numbers)
        feats[18] = abs(len(s1_addr) - len(cand_addr)) / max(max_len, 1)
        feats[19] = 1.0 if s1_addr == cand_addr else 0.0
    elif not s1_addr and not cand_addr:
        # Both missing — neutral
        feats[10:20] = 0.5
    else:
        feats[10:20] = 0.0

    # ── Postal Code & Door Number Semantic Conflict Resolution ─
    # Postal code check (5 or 6 digit codes)
    p1 = {n for n in s1_addr_numbers if len(n) in (5, 6)}
    p2 = {n for n in cand_addr_numbers if len(n) in (5, 6)}
    if p1 and p2:
        feats[20] = 1.0 if (p1 & p2) else -1.0  # Conflict if different postal codes!
    else:
        feats[20] = 0.0

    # Door / plot / building number check (1 to 4 digit numbers)
    d1 = {n for n in s1_addr_numbers if len(n) < 5}
    d2 = {n for n in cand_addr_numbers if len(n) < 5}
    if d1 and d2:
        feats[21] = 1.0 if (d1 & d2) else -1.0  # Conflict if different door numbers!
    else:
        feats[21] = 0.0

    # ── Cross features ─────────────────────────────────────────
    # Does the core name appear in the other's address?
    if s1_name_tokens and cand_addr_tokens:
        feats[22] = _containment(set(s1_name_tokens[:2]), set(cand_addr_tokens))
    else:
        feats[22] = 0.0

    feats[23] = 1.0 if s1_country == cand_country else 0.0

    return feats


def extract_features_batch(
    s1_records: Dict[str, dict],
    pool_records: Dict[str, dict],
    candidate_pairs: Dict[str, Set[str]],
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """Extract features for all candidate pairs.

    Args:
        s1_records: {entity_id: {name_clean, addr_clean, name_tokens,
                     addr_tokens, addr_numbers, country}}
        pool_records: same structure for S2+S3 records.
        candidate_pairs: {s1_id: set(candidate_ids)}

    Returns:
        (pair_ids, feature_matrix):
          pair_ids is a list of (s1_id, cand_id) tuples,
          feature_matrix is np.ndarray of shape (n_pairs, NUM_FEATURES).
    """
    pair_ids: List[Tuple[str, str]] = []
    features_list: List[np.ndarray] = []

    total_pairs = sum(len(v) for v in candidate_pairs.values())
    logger.info(f"Extracting features for {total_pairs:,} candidate pairs...")

    processed = 0
    for s1_id, cand_ids in candidate_pairs.items():
        s1 = s1_records.get(s1_id)
        if s1 is None:
            continue

        for cand_id in cand_ids:
            cand = pool_records.get(cand_id)
            if cand is None:
                continue

            feats = compute_pair_features(
                s1_name=s1.get("name_clean", ""),
                s1_addr=s1.get("addr_clean", ""),
                s1_name_tokens=s1.get("name_tokens", []),
                s1_addr_tokens=s1.get("addr_tokens", []),
                s1_addr_numbers=s1.get("addr_numbers", set()),
                s1_country=s1.get("country", ""),
                cand_name=cand.get("name_clean", ""),
                cand_addr=cand.get("addr_clean", ""),
                cand_name_tokens=cand.get("name_tokens", []),
                cand_addr_tokens=cand.get("addr_tokens", []),
                cand_addr_numbers=cand.get("addr_numbers", set()),
                cand_country=cand.get("country", ""),
            )
            pair_ids.append((s1_id, cand_id))
            features_list.append(feats)

            processed += 1
            if processed % 500_000 == 0:
                logger.info(f"  {processed:,}/{total_pairs:,} pairs processed")

    if features_list:
        feature_matrix = np.stack(features_list)
    else:
        feature_matrix = np.empty((0, NUM_FEATURES), dtype=np.float32)

    logger.info(f"Feature extraction complete: {feature_matrix.shape}")
    return pair_ids, feature_matrix


def df_to_record_dict(df: pd.DataFrame) -> Dict[str, dict]:
    """Convert preprocessed DataFrame to a dict-of-dicts for fast lookup.

    Expected columns: entity_id, name_clean, addr_clean, name_tokens,
    addr_tokens, addr_numbers, country.
    """
    records = {}
    for _, row in df.iterrows():
        records[row["entity_id"]] = {
            "name_clean": row.get("name_clean", ""),
            "addr_clean": row.get("addr_clean", ""),
            "name_tokens": row.get("name_tokens", []),
            "addr_tokens": row.get("addr_tokens", []),
            "addr_numbers": row.get("addr_numbers", set()),
            "country": row.get("country", ""),
        }
    return records
