"""Statistical tests for comparing AL methods."""

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def wilcoxon_test(
    scores_a: np.ndarray, scores_b: np.ndarray
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired comparisons across folds or slides.
    Returns (statistic, p_value).

    scipy raises when every paired difference is zero (e.g. a method compared
    against itself, or two methods that share the initial-round scores). Treat
    that degenerate case as "no difference" (p=1.0) instead of crashing the
    whole benchmark.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    if not np.any(diff):
        return 0.0, 1.0
    stat, p_value = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
    return float(stat), float(p_value)


def compare_methods(
    results: Dict[str, List[float]],
    baseline: str = "random",
    metric: str = "pcc_per_gene",
) -> Dict[str, Dict[str, float]]:
    """
    Compare all methods against a baseline using Wilcoxon signed-rank test.

    results: {method_name: [per_fold_scores]}
    Returns: {method_name: {'mean': ..., 'std': ..., 'p_value': ..., 'significant': ...}}
    """
    if baseline not in results:
        raise ValueError(f"Baseline '{baseline}' not found in results.")

    baseline_scores = np.array(results[baseline])
    comparisons = {}

    for method, scores in results.items():
        scores_arr = np.array(scores)
        comparison = {
            "mean": float(scores_arr.mean()),
            "std": float(scores_arr.std()),
            "improvement": float(scores_arr.mean() - baseline_scores.mean()),
        }

        if method != baseline and len(scores_arr) >= 5:
            _, p_value = wilcoxon_test(scores_arr, baseline_scores)
            comparison["p_value"] = p_value
            comparison["significant"] = p_value < 0.05
        else:
            comparison["p_value"] = None
            comparison["significant"] = None

        comparisons[method] = comparison

    return comparisons


def cross_validation_summary(
    fold_results: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Summarize metrics across k folds."""
    if not fold_results:
        return {}

    metrics = fold_results[0].keys()
    summary = {}

    for metric in metrics:
        values = [r[metric] for r in fold_results if metric in r]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return summary
