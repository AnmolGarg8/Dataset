"""
Central configuration and hyperparameters.

All tuneable constants live here so the entire pipeline can be
adjusted from one place without touching business logic.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Pipeline-wide settings."""

    # ── Random seed ─────────────────────────────────────────────
    seed: int = 42

    # ── I/O ─────────────────────────────────────────────────────
    sep: str = "\t"
    encoding: str = "utf-8"

    # ── Preprocessing ───────────────────────────────────────────
    # Legal suffixes stripped from business names (lowercased)
    legal_suffixes: List[str] = field(default_factory=lambda: [
        "inc", "incorporated", "corp", "corporation", "co", "company",
        "llc", "llp", "lp", "ltd", "limited", "pvt", "private",
        "plc", "gmbh", "ag", "sa", "sas", "sarl", "sasu", "eurl",
        "srl", "sl", "bv", "nv", "pty", "pte",
        "dba", "trading", "enterprises", "enterprise", "group",
        "holdings", "holding", "intl", "international",
    ])

    # Address abbreviation expansions
    address_abbrevs: dict = field(default_factory=lambda: {
        "st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard",
        "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
        "cir": "circle", "hwy": "highway", "pkwy": "parkway",
        "apt": "apartment", "ste": "suite", "fl": "floor",
        "dept": "department", "bldg": "building",
        "nr": "near", "opp": "opposite",
        "dist": "district", "nagar": "nagar", "marg": "road",
        "rte": "route", "ter": "terrace", "trl": "trail",
    })

    # ── Blocking ────────────────────────────────────────────────
    # Maximum document frequency ratio for a token to be used as
    # a blocking key. Tokens appearing in > this fraction of
    # records within a country partition are suppressed.
    max_token_df_ratio: float = 0.005

    # Minimum token length for name blocking keys
    min_token_len: int = 3

    # Maximum candidates per S1 entity from blocking
    max_candidates_per_entity: int = 100

    # Number of character n-gram shingles for fallback blocking
    shingle_size: int = 4

    # ── Features ────────────────────────────────────────────────
    # (Nothing to configure externally; feature set is fixed)

    # ── Model ───────────────────────────────────────────────────
    # LightGBM parameters
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })

    # Classification threshold — tuned on validation set to maximise F_0.5
    match_threshold: float = 0.5  # will be overwritten by tuning

    # ── Validation ──────────────────────────────────────────────
    val_fraction: float = 0.15  # fraction of S1 entities held out

    # ── Chunk sizes for streaming large files ───────────────────
    chunk_size: int = 200_000  # rows per chunk when reading TSVs


# Singleton instance
CFG = Config()
