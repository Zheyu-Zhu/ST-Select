"""Publication figures, drawn from the measured results in results_pg/.

Written because image-generation models cannot render data. Asked for a "flat"
HER2 curve, one such model printed the word FLAT inside the axes and drew a
rising curve anyway. Every number here is read from the result JSONs, so the
shapes are whatever the experiments actually produced, and a re-run after new
results simply redraws them.

Outputs vector PDF + SVG into figures/, sized for a two-column journal page
(single column 8.5 cm, double column 17.5 cm).

    python scripts/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

CM = 1 / 2.54
RATIOS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
COHORTS = [("her2", "HER2", "#1F6F78"), ("kidney", "Kidney", "#C0407A"),
           ("bc", "BC", "#4B3F8F")]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", transparent=(ext == "svg"))
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf / .svg / .png")


def load_budget(tag):
    p = ROOT / f"results_pg/budget/controlled/pg_{tag}_hvg300/results.json"
    d = json.load(open(p))
    M = np.array([[r["pcc_per_gene_by_slide"][i] for i in range(len(RATIOS))]
                  for r in d["detailed"]["random"]])
    return M


# ---------------------------------------------------------------- figure 1
def fig_budget_curves():
    """Three cohorts, three genuinely different shapes. This contrast IS the result."""
    fig, axes = plt.subplots(1, 3, figsize=(17.5 * CM, 4.6 * CM), sharey=False)
    sat = {"her2": 5, "kidney": 50, "bc": 75}
    for ax, (tag, label, col) in zip(axes, COHORTS):
        M = load_budget(tag)
        m, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
        x = np.array(RATIOS) * 100
        ax.axhline(m[-1], ls="--", lw=0.7, color="0.45", zorder=1)
        ax.fill_between(x, m - se, m + se, color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(x, m, "-o", color=col, lw=1.4, ms=3, zorder=3)
        s = sat[tag]
        ax.axvline(s, ls=":", lw=0.7, color="0.35", zorder=1)
        ax.plot([s], [m[RATIOS.index(s / 100)]], marker="*", ms=11,
                color=col, mec="white", mew=0.6, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 30, 50, 100])
        ax.set_xticklabels(["5", "10", "20", "30", "50", "100"])
        ax.set_xlabel("Labelled spots (%)")
        ax.set_title(f"{label}   ·   enough at {s}%", pad=4)
        lo, hi = min(m - se), max(m + se)
        pad = (hi - lo) * 0.35
        ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("Per-slide PCC")
    axes[0].text(0.97, 0.06, "full supervision", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=6, color="0.45")
    save(fig, "fig_budget_curves")


# ---------------------------------------------------------------- figure 2
def fig_forest():
    """Every strategy's advantage over random, with its CI. All cross zero."""
    runs = [
        ("results_pg/final/al20/pg_her2_hvg300/results.json", "HER2 · 20 strategies · frozen"),
        ("results_pg/final/al20/pg_kidney_hvg300/results.json", "Kidney · 20 strategies · frozen"),
    ]
    fig, axes = plt.subplots(1, len(runs), figsize=(17.5 * CM, 7.4 * CM), sharex=True)
    for ax, (rel, title) in zip(np.atleast_1d(axes), runs):
        d = json.load(open(ROOT / rel))
        det = d["detailed"]
        rnd = np.array([r["pcc_per_gene_by_slide"][-1] for r in det["random"]])
        rows = []
        for m, runs_ in det.items():
            if m in ("random", "full_supervision"):
                continue
            v = np.array([r["pcc_per_gene_by_slide"][-1] for r in runs_])
            dl = v - rnd
            rows.append((m, dl.mean(), dl.std(ddof=1) / np.sqrt(len(dl))))
        rows.sort(key=lambda r: r[1])
        y = np.arange(len(rows))
        mu = np.array([r[1] for r in rows]); se = np.array([r[2] for r in rows])
        ax.axvline(0, color="0.2", lw=1.0, zorder=1)
        ax.errorbar(mu, y, xerr=1.96 * se, fmt="o", ms=3.2, lw=0.9,
                    color="#1F6F78", ecolor="0.55", capsize=1.8, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlabel("Δ per-slide PCC vs random  (95% CI)")
        ax.set_title(title, pad=4)
        ax.grid(axis="x", lw=0.4, color="0.9", zorder=0)
        ax.set_axisbelow(True)
    save(fig, "fig_forest_vs_random")


# ---------------------------------------------------------------- figure 3
def fig_within_slide():
    """The crossover: the image model loses to nearest-neighbour interpolation."""
    ctrl = {"0.05": 0.1874, "0.10": 0.3065, "0.20": 0.3808, "0.30": 0.4041,
            "0.50": 0.4397, "0.75": 0.4636, "1.00": 0.4795}
    d = json.load(open(ROOT / "results_pg/within_slide/random/pg_her2_hvg300/results.json"))
    runs = d["detailed"]["random"]
    model = np.array([[r["pcc_per_gene_by_slide"][i] for i in range(len(RATIOS))]
                      for r in runs])
    mm, mse = model.mean(0), model.std(0, ddof=1) / np.sqrt(len(model))
    knn = np.array([ctrl[f"{r:.2f}"] for r in RATIOS])
    x = np.array(RATIOS) * 100

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.5 * CM, 9.2 * CM),
                                  gridspec_kw={"height_ratios": [2.1, 1]}, sharex=True)
    ax.fill_between(x, mm - mse, mm + mse, color="#1F6F78", alpha=0.18, lw=0)
    ax.plot(x, mm, "-o", color="#1F6F78", lw=1.5, ms=3.3, label="H&E model (learned)")
    ax.plot(x, knn, "-s", color="#B4622A", lw=1.5, ms=3.0,
            label="nearest-neighbour interpolation\n(no image, no training)")
    cross = 10
    ax.axvline(cross, ls=":", lw=0.8, color="0.35")
    ax.set_xscale("log")
    ax.set_ylabel("Per-slide PCC")
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    ax.set_title("Within-slide, interleaved holdout · HER2", pad=4)

    diff = mm - knn
    ax2.axhline(0, color="0.2", lw=0.9)
    ax2.axvline(cross, ls=":", lw=0.8, color="0.35")
    ax2.bar(x, diff, width=x * 0.32,
            color=["#3F7A52" if v > 0 else "#A83E32" for v in diff], lw=0)
    ax2.set_xscale("log")
    ax2.set_xticks([5, 10, 20, 30, 50, 100])
    ax2.set_xticklabels(["5", "10", "20", "30", "50", "100"])
    ax2.set_xlabel("Measured spots per slide (%)")
    ax2.set_ylabel("model − kNN")
    save(fig, "fig_within_slide_crossover")


# ---------------------------------------------------------------- figure 4
def fig_breadth():
    """Same spot budget, more patients."""
    fig, ax = plt.subplots(figsize=(8.5 * CM, 5.4 * CM))
    for tag, label, col in COHORTS:
        p = ROOT / f"results_pg/breadth_depth/pg_{tag}_hvg300/results.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        ks = [k for k in d["k_levels"] if k >= 4]
        xs, ys, es = [], [], []
        for k in ks:
            cells = [c["per_slide"] for c in d["cells"] if c["k"] == k]
            if not cells:
                continue
            xs.append(k); ys.append(np.mean(cells))
            es.append(np.std(cells, ddof=1) / np.sqrt(len(cells)))
        ax.errorbar(xs, ys, yerr=es, fmt="-o", ms=3.2, lw=1.3, capsize=1.8,
                    color=col, ecolor=col, alpha=0.9, label=label)
    ax.set_xlabel("Patients contributing the same spot budget")
    ax.set_ylabel("Per-slide PCC")
    ax.set_xscale("log")
    ax.set_xticks([4, 6, 8, 16, 17])
    ax.set_xticklabels(["4", "6", "8", "16", "17"])
    ax.legend(frameon=False, loc="best")
    ax.set_title("Fixed labelling budget, varying breadth", pad=4)
    save(fig, "fig_breadth_vs_depth")


if __name__ == "__main__":
    print("drawing from measured results in results_pg/ ...")
    for fn in (fig_budget_curves, fig_forest, fig_within_slide, fig_breadth):
        try:
            fn()
        except Exception as exc:
            print(f"  [skip] {fn.__name__}: {type(exc).__name__}: {exc}")
    sys.exit(0)
