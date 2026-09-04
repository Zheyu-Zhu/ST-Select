"""One figure per model configuration, drawn from results_pg/.

Four model configurations exist in this project and they are easy to confuse,
because the model is not named in the result path - it is implied by two things:

    model_name        image_mode   dataset suffix
    feature_predictor False        (none)      -> DenseNet-121 frozen features
    feature_predictor False        _uni        -> UNI ViT-L frozen features
    st_net            True         (none)      -> DenseNet-121 end-to-end
    uni_regressor     True         (none)      -> UNI ViT-L end-to-end

The dataset suffix rule is set in scripts/run_tonight_pg.py: the cache is named
pg_<cohort>_<target> for densenet121 and pg_<cohort>_<target>_uni otherwise.

The four figures are deliberately NOT the same chart four times. Only two of the
configurations were ever put through an acquisition sweep; the other two exist
only at full supervision. Drawing an AL curve for those would mean inventing
data, so their panels show what was actually measured and say plainly what was
not run.

    python scripts/make_model_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "models"
OUT.mkdir(parents=True, exist_ok=True)
CM = 1 / 2.54

COHORTS = [("her2", "HER2", "#1F6F78"), ("kidney", "Kidney", "#C0407A"),
           ("bc", "BC", "#4B3F8F")]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8.2,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", transparent=(ext == "svg"))
    plt.close(fig)
    print(f"  wrote figures/models/{name}.pdf / .svg / .png")


def curves(path, n_budgets=None):
    """(strategy -> mean-over-folds vector) for one sweep."""
    d = json.load(open(path))
    det = d["detailed"]
    L = n_budgets or len(det["random"][0]["pcc_per_gene_by_slide"])
    out = {}
    for k, runs in det.items():
        if k == "full_supervision":
            continue
        M = np.array([[r["pcc_per_gene_by_slide"][i] for i in range(L)] for r in runs])
        out[k] = M
    # budget_ratios sits at the top level in some runs and inside config in
    # others, depending on which script wrote the file.
    br = d.get("budget_ratios") or (d.get("config") or {}).get("budget_ratios") or []
    return out, list(br)[:L]


def point(path, strat=None):
    d = json.load(open(path))
    det = d["detailed"]
    k = strat or ("full_supervision" if "full_supervision" in det else list(det)[0])
    v = np.array([r["pcc_per_gene_by_slide"][-1] for r in det[k]])
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def sweep_panel(ax, path, title, col, n_budgets=None):
    """Every strategy as a thin line, random as the heavy black one."""
    try:
        C, ratios = curves(path, n_budgets)
    except Exception as exc:
        ax.text(.5, .5, f"not run\n({type(exc).__name__})", transform=ax.transAxes,
                ha="center", va="center", color="0.5", fontsize=7)
        ax.set_title(title, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        return None
    x = np.array(ratios) * 100
    rnd = C["random"].mean(0)
    for k, M in C.items():
        if k == "random":
            continue
        ax.plot(x, M.mean(0), lw=0.75, color=col, alpha=0.45, zorder=2)
    ax.plot(x, rnd, lw=2.1, color="black", zorder=4, solid_capstyle="round")
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.set_xticklabels(["5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_title(title, pad=11)
    n = len(C) - 1
    win = sum(1 for k, M in C.items() if k != "random" and (M.mean(0) > rnd).all())
    # Above the axes, not inside: the curves fall on HER2/BC and rise on Kidney,
    # so no interior corner is free in all three panels at once.
    ax.text(.5, 1.015, f"{n} strategies  ·  {win} beat random at every budget",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6,
            color="0.4")
    return n


# ------------------------------------------------------------------ figure 1
def fig_densenet_frozen():
    """The only configuration with a full acquisition sweep on all three cohorts."""
    fig, axes = plt.subplots(1, 3, figsize=(17.5 * CM, 5.2 * CM))
    src = {"her2": "results_pg/final/al20/pg_her2_hvg300/results.json",
           "kidney": "results_pg/final/al20/pg_kidney_hvg300/results.json",
           "bc": "results_pg/al_frozen/pg_bc_hvg300/results.json"}
    for ax, (tag, label, col) in zip(axes, COHORTS):
        sweep_panel(ax, ROOT / src[tag], label, col)
        ax.set_xlabel("Labelled spots (%)")
    axes[0].set_ylabel("Per-slide PCC")
    # HER2 and BC fall with more data. That is the fixed-epoch confound from
    # E18, not a property of the task: this legacy sweep trained every budget
    # for the same number of epochs, which overtrains the large budgets harder.
    # The controlled re-run (results_pg/budget/controlled) removes it. The
    # strategy comparison within each budget is unaffected - every arm at a
    # given budget got the same epoch count.
    fig.text(.01, -.13, "Fixed epochs at every budget (legacy sweep) — the downward "
             "drift on HER2 and BC is differential overtraining, not task difficulty. "
             "Comparisons between strategies at the same budget are unaffected.",
             fontsize=5.9, color="0.45", ha="left")
    axes[0].plot([], [], color="black", lw=2.1, label="random")
    axes[0].plot([], [], color=COHORTS[0][2], lw=0.75, alpha=.6, label="each strategy")
    axes[0].legend(frameon=False, loc="lower right", handlelength=1.5)
    fig.suptitle("DenseNet-121 frozen features + MLP head  ·  acquisition sweep",
                 fontsize=9, fontweight="bold", y=1.04, x=.01, ha="left")
    save(fig, "model1_densenet_frozen")


# ------------------------------------------------------------------ figure 2
def fig_uni_frozen():
    """Full supervision only - no sweep was ever run on this configuration."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17.5 * CM, 5.2 * CM),
                                  gridspec_kw={"width_ratios": [1, 1.25]})
    xs, ys, es, cs = [], [], [], []
    for i, (tag, label, col) in enumerate(COHORTS):
        m, se = point(ROOT / f"results_pg/frozen_uni/pg_{tag}_hvg300_uni/results.json")
        xs.append(i); ys.append(m); es.append(se); cs.append(col)
    ax.bar(xs, ys, yerr=es, color=cs, width=.62, capsize=3, lw=0,
           error_kw={"lw": .9, "ecolor": "0.35"})
    ax.set_xticks(xs); ax.set_xticklabels([c[1] for c in COHORTS])
    ax.set_ylabel("Per-slide PCC")
    ax.set_title("Full supervision (100% of spots)", pad=4)

    # the other two tasks this configuration was measured on
    rows = []
    for tag, label, col in COHORTS:
        for task, sub in (("cell-type", "celltype"), ("gene+spatial", "gene_spatial")):
            p = ROOT / f"results_pg/tasks/{sub}/pg_{tag}_{'celltype' if sub=='celltype' else 'hvg300'}_uni/results.json"
            if p.exists():
                m, se = point(p)
                rows.append((f"{label}\n{task}", m, se, col))
    if rows:
        ax2.bar(range(len(rows)), [r[1] for r in rows], yerr=[r[2] for r in rows],
                color=[r[3] for r in rows], width=.62, capsize=2.4, lw=0,
                error_kw={"lw": .9, "ecolor": "0.35"})
        ax2.set_xticks(range(len(rows)))
        ax2.set_xticklabels([r[0] for r in rows], fontsize=5.8)
    ax2.set_title("Other evaluation targets, same configuration", pad=4)
    ax2.text(.5, .5, "no acquisition sweep was run\non this configuration",
             transform=ax2.transAxes, ha="center", va="center", fontsize=7,
             color="0.35", bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                     ec="0.8", lw=.7))
    fig.suptitle("UNI ViT-L frozen features + MLP head  ·  full supervision only",
                 fontsize=9, fontweight="bold", y=1.04, x=.01, ha="left")
    save(fig, "model2_uni_frozen")


# ------------------------------------------------------------------ figure 3
def fig_densenet_e2e():
    """Also full supervision only."""
    fig, ax = plt.subplots(figsize=(8.6 * CM, 5.2 * CM))
    w = .38
    for i, (tag, label, col) in enumerate(COHORTS):
        a, ae = point(ROOT / f"results_pg/frozen_dn/pg_{tag}_hvg300/results.json")
        b, be = point(ROOT / f"results_pg/e2e_dn/pg_{tag}_hvg300/results.json")
        ax.bar(i - w / 2, a, yerr=ae, width=w, color=col, alpha=.38, lw=0,
               capsize=2.4, error_kw={"lw": .9, "ecolor": "0.45"},
               label="frozen features" if i == 0 else None)
        ax.bar(i + w / 2, b, yerr=be, width=w, color=col, lw=0,
               capsize=2.4, error_kw={"lw": .9, "ecolor": "0.35"},
               label="end-to-end" if i == 0 else None)
        ax.text(i, max(a, b) + .022, f"+{(b - a) / a * 100:.0f}%", ha="center",
                fontsize=6.2, color="0.3", fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels([c[1] for c in COHORTS])
    ax.set_ylabel("Per-slide PCC")
    ax.legend(frameon=False, loc="upper right", handlelength=1.3)
    ax.text(.5, .06, "no acquisition sweep was run on this configuration",
            transform=ax.transAxes, ha="center", fontsize=6.2, color="0.45")
    fig.suptitle("DenseNet-121 end-to-end  ·  full supervision only",
                 fontsize=9, fontweight="bold", y=1.04, x=.01, ha="left")
    save(fig, "model3_densenet_e2e")


# ------------------------------------------------------------------ figure 4
def fig_uni_e2e():
    """Budget curve plus the acquisition sweep that is still running."""
    fig, axes = plt.subplots(1, 3, figsize=(17.5 * CM, 5.2 * CM),
                             gridspec_kw={"width_ratios": [1, 1.15, 1]})

    # (1) budget curve, HER2
    d = json.load(open(ROOT / "results_pg/uni_budget/pg_her2_hvg300/results.json"))
    R = d.get("budget_ratios", [.05, .1, .2, .3, .5, .75, 1.])
    M = np.array([[r["pcc_per_gene_by_slide"][i] for i in range(len(R))]
                  for r in d["detailed"]["random"]])
    m, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
    x = np.array(R) * 100
    ax = axes[0]
    ax.axhline(m[-1], ls="--", lw=.8, color="0.45")
    ax.fill_between(x, m - se, m + se, color="#1F6F78", alpha=.18, lw=0)
    ax.plot(x, m, "-o", color="#1F6F78", lw=1.5, ms=3.2)
    ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 50, 100])
    ax.set_xticklabels(["5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("Labelled spots (%)"); ax.set_ylabel("Per-slide PCC")
    ax.set_title("HER2 budget curve (random)", pad=4)
    ax.text(.97, .06, "full supervision", transform=ax.transAxes, ha="right",
            fontsize=5.8, color="0.45")

    # (2) the running sweep
    det = {}
    for p in ("results_pg/al20_e2e_part1/results_partial.json",
              "results_pg/al20_e2e/pg_her2_hvg300/results_partial.json"):
        f = ROOT / p
        if f.exists():
            det.update(json.load(open(f))["detailed"])
    ax = axes[1]
    if det:
        L = 5
        xb = np.array([5, 10, 20, 30, 50])
        C = {k: np.array([[r["pcc_per_gene_by_slide"][i] for i in range(L)] for r in v])
             for k, v in det.items()}
        rnd = C["random"].mean(0)
        for k, Mk in C.items():
            if k == "random":
                continue
            ax.plot(xb, Mk.mean(0), lw=.95, alpha=.75, zorder=2, label=k)
        ax.plot(xb, rnd, lw=2.2, color="black", zorder=4, label="random")
        ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 30, 50])
        ax.set_xticklabels(["5", "10", "20", "30", "50"])
        ax.minorticks_off()
        ax.set_xlabel("Labelled spots (%)")
        ax.set_title(f"HER2 acquisition sweep  ·  {len(C)} of 20 done", pad=4)
        ax.legend(frameon=False, fontsize=5.4, loc="lower left", ncol=2,
                  handlelength=1.1, columnspacing=.8, labelspacing=.25)

    # (3) full supervision across cohorts
    ax = axes[2]
    for i, (tag, label, col) in enumerate(COHORTS):
        m2, se2 = point(ROOT / f"results_pg/final/uni_e2e/pg_{tag}_hvg300/results.json")
        ax.bar(i, m2, yerr=se2, color=col, width=.62, capsize=3, lw=0,
               error_kw={"lw": .9, "ecolor": "0.35"})
    ax.set_xticks(range(3)); ax.set_xticklabels([c[1] for c in COHORTS])
    ax.set_title("Full supervision, all cohorts", pad=4)
    fig.suptitle("UNI ViT-L end-to-end, 304 M params  ·  sweep in progress",
                 fontsize=9, fontweight="bold", y=1.04, x=.01, ha="left")
    save(fig, "model4_uni_e2e")


if __name__ == "__main__":
    print("drawing one figure per model configuration ...")
    for fn in (fig_densenet_frozen, fig_uni_frozen, fig_densenet_e2e, fig_uni_e2e):
        try:
            fn()
        except Exception as exc:
            print(f"  [skip] {fn.__name__}: {type(exc).__name__}: {exc}")
    sys.exit(0)
