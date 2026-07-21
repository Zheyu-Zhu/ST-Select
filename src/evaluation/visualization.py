"""Visualization utilities for AL experiment results."""

from typing import Dict, List, Optional

import numpy as np


def plot_al_curves(
    results: Dict[str, Dict[str, List]],
    metric: str = "pcc_per_gene",
    title: str = "Active Learning Curves",
    save_path: Optional[str] = None,
):
    """
    Plot AL acquisition curves: metric vs. number of labeled samples.

    results: {method_name: {'n_labeled': [...], metric: [...]}}
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for method_name, method_results in results.items():
        x = method_results["n_labeled"]
        y = method_results[metric]
        ax.plot(x, y, marker="o", label=method_name, linewidth=2, markersize=4)

    ax.set_xlabel("Number of Labeled Samples", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_gene_pcc_heatmap(
    gene_pccs: Dict[str, np.ndarray],
    gene_names: List[str],
    top_n: int = 50,
    save_path: Optional[str] = None,
):
    """
    Heatmap of per-gene PCC across methods.

    gene_pccs: {method_name: np.ndarray of shape (n_genes,)}
    """
    import matplotlib.pyplot as plt

    methods = list(gene_pccs.keys())
    # Sort genes by average PCC across methods
    avg_pcc = np.mean([gene_pccs[m] for m in methods], axis=0)
    top_gene_idx = np.argsort(avg_pcc)[-top_n:][::-1]

    data = np.array([gene_pccs[m][top_gene_idx] for m in methods])

    fig, ax = plt.subplots(1, 1, figsize=(max(12, top_n * 0.3), len(methods) * 0.5 + 2))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=0.8)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([gene_names[i] for i in top_gene_idx], rotation=90, fontsize=7)

    plt.colorbar(im, ax=ax, label="PCC")
    ax.set_title(f"Per-Gene PCC (Top {top_n} Genes)", fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_budget_comparison(
    results: Dict[str, Dict[float, float]],
    metric: str = "pcc_per_gene",
    save_path: Optional[str] = None,
):
    """
    Bar chart comparing methods at different budget ratios.

    results: {method_name: {mask_ratio: metric_value}}
    """
    import matplotlib.pyplot as plt

    methods = list(results.keys())
    ratios = sorted(list(results[methods[0]].keys()))
    n_methods = len(methods)
    n_ratios = len(ratios)

    fig, ax = plt.subplots(1, 1, figsize=(n_ratios * 2, 6))

    width = 0.8 / n_methods
    x = np.arange(n_ratios)

    for i, method in enumerate(methods):
        values = [results[method].get(r, 0) for r in ratios]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=method)

    ax.set_xlabel("Budget Ratio", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.0%}" for r in ratios])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


# Colorblind-safe categorical palette (Okabe–Ito), keyed by reporting family.
_FAMILY_COLORS = {
    "Random": "#999999",
    "Uncertainty": "#E69F00",
    "Coverage": "#56B4E9",
    "Hybrid": "#009E73",
    "Spatial": "#CC79A7",
    "RL": "#D55E00",
    "FullSupervision": "#000000",
}


def plot_family_budget_curves(
    by_ratio_family: Dict[str, Dict[str, float]],
    title: str = "AL family vs. budget",
    save_path: Optional[str] = None,
):
    """Budget/AL curve, one line per reporting family (mean over its methods).

    by_ratio_family: {family: {ratio_str: mean_metric}}
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for family, per_ratio in by_ratio_family.items():
        ratios = sorted(float(r) for r in per_ratio)
        y = [per_ratio[f"{r:.2f}"] for r in ratios]
        ax.plot(
            [r * 100 for r in ratios], y, marker="o", linewidth=2, markersize=5,
            label=family, color=_FAMILY_COLORS.get(family),
            linestyle="--" if family == "FullSupervision" else "-",
        )
    ax.set_xlabel("Labeled budget (% of training pool)", fontsize=12)
    ax.set_ylabel("Per-gene PCC", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_top_n_pcc(
    pcc_topn: Dict[str, Dict[str, float]],
    title: str = "Per-gene PCC vs. gene-set size (top-N HVGs)",
    save_path: Optional[str] = None,
):
    """Trend of per-gene PCC as the reported gene set grows (top-10 -> top-300),
    one line per method. pcc_topn: {method: {N_str: pcc}}."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for method, per_n in pcc_topn.items():
        ns = sorted(int(n) for n in per_n if per_n[n] is not None)
        y = [per_n[str(n)] if str(n) in per_n else per_n[n] for n in ns]
        ax.plot(ns, y, marker="o", linewidth=2, markersize=4, label=method)
    ax.set_xlabel("Top-N dispersion-ranked genes", fontsize=12)
    ax.set_ylabel("Mean per-gene PCC", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
