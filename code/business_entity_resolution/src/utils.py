"""
I/O utilities module for Business Entity Resolution pipeline.

Amazon ML Challenge 2026.
Provides robust, high-performance I/O helpers for reading and writing
source datasets, candidate pairs, matching results, and ground truth files.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Set, Union

import pandas as pd


__all__ = [
    "setup_logging",
    "read_source_tsv",
    "read_ground_truth",
    "parse_id_list",
    "write_matching_results",
    "write_candidate_pairs",
    "ensure_dir",
]


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with standard format and return logger instance.

    Format: '%(asctime)s [%(levelname)s] %(message)s'

    Args:
        level: Logging level (default: logging.INFO).

    Returns:
        logging.Logger: The configured root logger.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(level)
    return logger


def read_source_tsv(
    path: str,
    usecols: Optional[Union[List[str], List[int], Callable]] = None,
    chunksize: Optional[int] = None,
) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
    """Read a tab-separated source file with UTF-8 encoding.

    Supports reading the complete table into a DataFrame or streaming
    chunks via TextFileReader when chunksize is specified.

    Args:
        path: Path to the TSV file.
        usecols: Optional subset of columns to read.
        chunksize: Optional number of rows per chunk for streaming.

    Returns:
        pd.DataFrame or TextFileReader iterator (when chunksize is set).
    """
    return pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )


def read_ground_truth(path: str) -> pd.DataFrame:
    """Read ground truth TSV with columns: source1_entity_id, matched_entity_ids.

    Args:
        path: Path to the ground truth TSV file.

    Returns:
        pd.DataFrame: DataFrame containing ground truth entity mappings.
    """
    return pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        dtype=str,
    )


def parse_id_list(id_string: Any) -> list:
    """Parse a comma-separated string of entity IDs into a list.

    Handles NaN, None, and empty strings by returning an empty list.
    Strips whitespace from each ID and filters out empty tokens.

    Args:
        id_string: Comma-separated string of entity IDs, NaN, or None.

    Returns:
        List of cleaned entity ID strings.
    """
    if id_string is None or pd.isna(id_string):
        return []
    if isinstance(id_string, (list, tuple, set)):
        return [str(item).strip() for item in id_string if str(item).strip()]
    id_str = str(id_string).strip()
    if not id_str or id_str.lower() in ("nan", "none", "<na>"):
        return []
    return [item.strip() for item in id_str.split(",") if item.strip()]


def _format_id_collection(ids: Any) -> str:
    """Format an iterable or string of IDs into a comma-separated string without spaces."""
    if ids is None or (isinstance(ids, float) and pd.isna(ids)):
        return ""
    if isinstance(ids, str):
        ids_str = ids.strip()
        if not ids_str or ids_str.lower() in ("nan", "none", "<na>"):
            return ""
        tokens = [token.strip() for token in ids_str.split(",") if token.strip()]
        return ",".join(sorted(tokens))
    if hasattr(ids, "__iter__"):
        clean_ids = [str(item).strip() for item in ids if str(item).strip()]
        return ",".join(sorted(clean_ids))
    val_str = str(ids).strip()
    return "" if val_str.lower() in ("nan", "none", "<na>") else val_str


def _write_id_mapping(
    mapping: dict,
    path: str,
    id_col: str,
    val_col: str,
) -> None:
    """Write an ID mapping dictionary to a tab-separated file deterministically.

    Args:
        mapping: Dictionary mapping entity IDs to matched or candidate IDs.
        path: Path where TSV file will be written.
        id_col: Column header name for source entity IDs.
        val_col: Column header name for target entity IDs.
    """
    parent_dir = os.path.dirname(path)
    if parent_dir:
        ensure_dir(parent_dir)

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{id_col}\t{val_col}\n")
        for s1_id in sorted(mapping.keys(), key=str):
            matched_str = _format_id_collection(mapping[s1_id])
            f.write(f"{s1_id}\t{matched_str}\n")


def write_matching_results(results: dict, path: str) -> None:
    """Write final matching results to a tab-separated file.

    Takes a dict mapping {s1_entity_id: set_of_matched_ids} and writes
    a TSV file with header 'source1_entity_id\\tmatched_entity_ids'.
    Matched IDs are comma-separated without spaces. Singletons are
    represented as empty strings. Results are sorted by source1_entity_id
    for determinism.

    Args:
        results: Dict mapping source1_entity_id to a set or collection of matched IDs.
        path: Output TSV file path.
    """
    _write_id_mapping(
        mapping=results,
        path=path,
        id_col="source1_entity_id",
        val_col="matched_entity_ids",
    )


def write_candidate_pairs(candidates: dict, path: str) -> None:
    """Write candidate pairs to a tab-separated file.

    Takes a dict mapping {s1_entity_id: candidate_ids} and writes
    a TSV file with header 'source1_entity_id\\tcandidate_entity_ids'.
    Candidate IDs are comma-separated without spaces. Singletons are
    represented as empty strings. Results are sorted by source1_entity_id
    for determinism.

    Args:
        candidates: Dict mapping source1_entity_id to a set or collection of candidate IDs.
        path: Output TSV file path.
    """
    _write_id_mapping(
        mapping=candidates,
        path=path,
        id_col="source1_entity_id",
        val_col="candidate_entity_ids",
    )


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist.

    Args:
        path: Directory path to create.
    """
    if path:
        os.makedirs(path, exist_ok=True)
