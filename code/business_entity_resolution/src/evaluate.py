"""
F_beta evaluation module for Business Entity Resolution pipeline.

Amazon ML Challenge 2026.
Computes macro-averaged F_0.5 (and arbitrary F_beta) per Source 1 entity,
accounting for singleton entities, empty prediction sets, and precision/recall
trade-offs according to competition rules.
"""

from typing import Any, Dict, List, Optional, Set
import numpy as np


__all__ = [
    "f_beta_score",
    "macro_f_beta",
    "detailed_evaluation",
    "print_detailed_report",
]


def f_beta_score(predicted: Set, true: Set, beta: float = 0.5) -> float:
    """Compute F_beta for a single entity's predicted vs. true match sets.

    Formula:
        F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)

    For beta = 0.5:
        F_0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)

    Special cases:
        - If both predicted and true sets are empty: F_beta = 1.0 (correct singleton)
        - If predicted is empty but true is non-empty: F_beta = 0.0
        - If predicted is non-empty but true is empty: F_beta = 0.0
        - If intersection is empty: F_beta = 0.0
        - precision = |predicted ∩ true| / |predicted|
        - recall = |predicted ∩ true| / |true|

    Args:
        predicted: Set of predicted matched entity IDs.
        true: Set of true matched entity IDs.
        beta: Beta parameter weighting precision vs. recall (default: 0.5).
            beta < 1 gives more weight to precision, beta > 1 gives more weight to recall.

    Returns:
        float: The entity's F_beta score in [0.0, 1.0].
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    # Robust set conversion handling None, lists, tuples, or sets
    pred_set: Set = (
        predicted
        if isinstance(predicted, set)
        else set(predicted)
        if predicted is not None
        else set()
    )
    true_set: Set = (
        true
        if isinstance(true, set)
        else set(true)
        if true is not None
        else set()
    )

    len_pred = len(pred_set)
    len_true = len(true_set)

    # Edge cases
    if len_pred == 0 and len_true == 0:
        return 1.0  # Correct singleton
    if len_pred == 0 or len_true == 0:
        return 0.0  # One empty, other non-empty

    intersection_len = len(pred_set & true_set)
    if intersection_len == 0:
        return 0.0

    precision = intersection_len / len_pred
    recall = intersection_len / len_true

    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return 0.0

    return ((1.0 + beta_sq) * precision * recall) / denom


def macro_f_beta(
    predictions: Dict,
    ground_truth: Dict,
    beta: float = 0.5,
    verbose: bool = True,
) -> float:
    """Compute macro-averaged F_beta across ALL S1 entities in ground_truth.

    For each S1 entity in ground_truth, looks up predicted IDs (defaulting
    to an empty set if absent), computes the per-entity F_beta score, and
    averages across all S1 entities in ground_truth. Also prints summary stats:
    total entities, avg score, num perfect, num zero.

    Args:
        predictions: Dict mapping {s1_id: set_of_predicted_ids}.
        ground_truth: Dict mapping {s1_id: set_of_true_ids}.
        beta: Beta parameter (default: 0.5 for F_0.5).
        verbose: If True, print summary statistics to stdout (default: True).

    Returns:
        float: Macro-averaged F_beta score across all ground_truth entities.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        if verbose:
            print(f"Evaluation Summary (F_{beta:g}):")
            print("  Total entities: 0")
            print("  Avg score:      0.0000")
            print("  Num perfect:    0")
            print("  Num zero:       0")
        return 0.0

    scores: List[float] = []
    num_perfect = 0
    num_zero = 0

    for s1_id, true_val in ground_truth.items():
        pred_val = predictions.get(s1_id, set())
        score = f_beta_score(pred_val, true_val, beta=beta)
        scores.append(score)

        if np.isclose(score, 1.0):
            num_perfect += 1
        if np.isclose(score, 0.0):
            num_zero += 1

    macro_avg = float(np.mean(scores))

    if verbose:
        pct_perfect = (num_perfect / total_entities) * 100.0
        pct_zero = (num_zero / total_entities) * 100.0
        print(f"Evaluation Summary (F_{beta:g}):")
        print(f"  Total entities: {total_entities:,}")
        print(f"  Avg score:      {macro_avg:.4f}")
        print(f"  Num perfect:    {num_perfect:,} ({pct_perfect:.2f}%)")
        print(f"  Num zero:       {num_zero:,} ({pct_zero:.2f}%)")

    return macro_avg


def detailed_evaluation(
    predictions: Dict,
    ground_truth: Dict,
    beta: float = 0.5,
) -> Dict[str, Any]:
    """Compute detailed evaluation breakdown for Entity Resolution.

    Computes macro F_beta score, entity counts, singleton accuracy breakdown,
    and average precision/recall across all non-singleton entities.

    Args:
        predictions: Dict mapping {s1_id: set_of_predicted_ids}.
        ground_truth: Dict mapping {s1_id: set_of_true_ids}.
        beta: Beta parameter (default: 0.5 for F_0.5).

    Returns:
        dict containing:
            - 'macro_f_beta': The macro-averaged F_beta score across all S1 entities.
            - 'total_entities': Total number of S1 entities in ground truth.
            - 'singletons_correct': Count of entities where both predicted and true are empty.
            - 'singletons_wrong': Count where true is empty but predicted is non-empty.
            - 'missed_singletons': Count where true is non-empty but predicted is empty.
            - 'precision_avg': Average precision across non-singleton entities.
            - 'recall_avg': Average recall across non-singleton entities.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {
            "macro_f_beta": 0.0,
            "total_entities": 0,
            "singletons_correct": 0,
            "singletons_wrong": 0,
            "missed_singletons": 0,
            "precision_avg": 0.0,
            "recall_avg": 0.0,
        }

    scores: List[float] = []
    singletons_correct = 0
    singletons_wrong = 0
    missed_singletons = 0
    precisions: List[float] = []
    recalls: List[float] = []

    for s1_id, true_val in ground_truth.items():
        pred_val = predictions.get(s1_id, set())

        # Normalize to set
        pred_set: Set = (
            pred_val
            if isinstance(pred_val, set)
            else set(pred_val)
            if pred_val is not None
            else set()
        )
        true_set: Set = (
            true_val
            if isinstance(true_val, set)
            else set(true_val)
            if true_val is not None
            else set()
        )

        score = f_beta_score(pred_set, true_set, beta=beta)
        scores.append(score)

        len_pred = len(pred_set)
        len_true = len(true_set)

        if len_true == 0:
            if len_pred == 0:
                singletons_correct += 1
            else:
                singletons_wrong += 1
        else:
            # Non-singleton entity in ground truth
            if len_pred == 0:
                missed_singletons += 1
                precisions.append(0.0)
                recalls.append(0.0)
            else:
                intersection_len = len(pred_set & true_set)
                p = intersection_len / len_pred
                r = intersection_len / len_true
                precisions.append(p)
                recalls.append(r)

    macro_avg = float(np.mean(scores))
    precision_avg = float(np.mean(precisions)) if precisions else 0.0
    recall_avg = float(np.mean(recalls)) if recalls else 0.0

    return {
        "macro_f_beta": macro_avg,
        "total_entities": total_entities,
        "singletons_correct": singletons_correct,
        "singletons_wrong": singletons_wrong,
        "missed_singletons": missed_singletons,
        "precision_avg": precision_avg,
        "recall_avg": recall_avg,
    }


def print_detailed_report(report: Dict[str, Any]) -> None:
    """Print a clean, human-readable summary of detailed evaluation metrics.

    Args:
        report: Dict returned by detailed_evaluation().
    """
    total = report.get("total_entities", 0)
    macro_score = report.get("macro_f_beta", 0.0)
    s_corr = report.get("singletons_correct", 0)
    s_wrong = report.get("singletons_wrong", 0)
    m_single = report.get("missed_singletons", 0)
    p_avg = report.get("precision_avg", 0.0)
    r_avg = report.get("recall_avg", 0.0)

    print("=" * 60)
    print("DETAILED EVALUATION REPORT")
    print("=" * 60)
    print(f"Total entities evaluated:   {total:,}")
    print(f"Macro F_beta score:         {macro_score:.4f}")
    print("-" * 60)
    print("Singleton Analysis:")
    pct_corr = (s_corr / total * 100.0) if total > 0 else 0.0
    pct_wrong = (s_wrong / total * 100.0) if total > 0 else 0.0
    pct_miss = (m_single / total * 100.0) if total > 0 else 0.0
    print(f"  Correct singletons:       {s_corr:,} ({pct_corr:.2f}%)")
    print(f"  Wrong singletons:         {s_wrong:,} ({pct_wrong:.2f}%)")
    print(f"  Missed singletons:        {m_single:,} ({pct_miss:.2f}%)")
    print("-" * 60)
    print("Non-Singleton Match Metrics:")
    print(f"  Average precision:        {p_avg:.4f}")
    print(f"  Average recall:           {r_avg:.4f}")
    print("=" * 60)
