"""Panel (d): two evaluation targets across the cohorts that have data.

A 2x2 grid - gene-expression accuracy and spatial fidelity, on HER2 and cSCC -
with one line per acquisition strategy and random drawn heavy and black. The
same shape should be visible in every cell if the finding holds: random on top
at the small budgets, overtaken somewhere near 30%.

Two decisions worth stating.

Rows do not share a y-axis across cohorts. HER2 tops out near 0.24 and cSCC near
0.11, so a common scale would flatten cSCC into a line and hide the very
crossover the panel exists to show. Each cell is scaled to its own data and
labelled, which is the honest way to compare shapes rather than levels.

A budget is plotted only where the strategy has all four folds. cSCC is still
running, so its non-random arms stop early rather than continuing on three folds
or fewer - a three-fold mean is not comparable with a four-fold one when the
folds differ severalfold, and drawing it would invent a crossover.

    python scripts/make_panel_d.py
    python scripts/make_panel_d.py --split           # separate cells, with tick numbers
    python scripts/make_panel_d.py --split --bare    # no numbers either
"""

import glob
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent.parent
ST = Path(r"C:\Users\zheyu\workspace\stnet_ref\ST-Net")
OUT = ROOT / "figures" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

CM = 1 / 2.54
# Log spacing for the budgets. Spacing them by value instead puts 5 and 10 on
# top of each other - the two budgets the whole result rests on - and gives half
# the width to the empty 50-100 stretch. Pass --linearx to see that.
LOGX = True
FINAL = {0.05: 110, 0.1: 80, 0.3: 70, 0.5: 60}
ORDER = ["random", "coreset", "mc_dropout", "tod", "typiclust"]
LABEL = {"random": "Random", "coreset": "CoreSet", "mc_dropout": "MC-dropout",
         "tod": "TOD", "typiclust": "TypiClust"}
COL = {"random": "#1C1B18", "coreset": "#D2601A", "mc_dropout": "#4B3F8F",
       "tod": "#3F7A52", "typiclust": "#C0407A"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0.8, "savefig.dpi": 400, "savefig.transparent": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def pcc_table(ds):
    rows = []
    for c in glob.glob(str(ST / "weights_st" / ds / "*" / "fold*" /
                           "slidewise_metrics" / "epoch_*.csv")):
        p = c.replace("\\", "/").split("/")
        s, r = p[-4].rsplit("_", 1)
        f = int(re.search(r"fold(\d+)", p[-3]).group(1))
        e = int(re.search(r"epoch_(\d+)", p[-1]).group(1))
        # the 100% runs live alongside these but are a different experiment with
        # their own epoch count; ceiling() reads them, this does not
        if float(r) not in FINAL or e != FINAL[float(r)]:
            continue
        rows.append(dict(sampler=s, budget=float(r), fold=f,
                         value=pd.read_csv(c)["pcc"].mean()))
    return pd.DataFrame(rows)


CEIL_EPOCH = 50


def ceiling(ds, key="pcc"):
    """The 100% value for one metric, or None until all four folds are scored.

    At beta=1 there is no selection left to make, so this belongs to no strategy
    and every curve is carried to the same point. PCC comes from the run's own
    slide-wise output; ARI and Moran are read from the spatial file that
    includes the 1.0 budget, which is kept separate from the four-budget one so
    that the scripts reading that file are not handed a fifth budget.
    """
    if key == "pcc":
        vals = []
        for f in range(4):
            c = (ST / "weights_st" / ds / "random_1.0" / f"fold{f}" /
                 "slidewise_metrics" / f"epoch_{CEIL_EPOCH}.csv")
            if not c.exists():
                return None
            vals.append(pd.read_csv(c)["pcc"].mean())
        return float(np.mean(vals))

    f = ST / f"{ds}_spatial_metrics_full.csv"
    if not f.exists():
        return None
    col = {"ari": "ari", "moran": "moran_fidelity"}[key]
    g = pd.read_csv(f).query("sampler == 'random' and budget == 1.0")[col]
    return float(g.mean()) if len(g) == 4 else None


def spatial_table(ds, column):
    f = ST / f"{ds}_spatial_metrics.csv"
    if not f.exists():
        return pd.DataFrame(columns=["sampler", "budget", "fold", "value"])
    d = pd.read_csv(f).rename(columns={column: "value"})
    return d[["sampler", "budget", "fold", "value"]]


def moran_table(ds):
    return spatial_table(ds, "moran_fidelity")


def ari_table(ds):
    return spatial_table(ds, "ari")


def draw(ax, d, title, ylab, show_ylab, ceil=None):
    any_line = False
    for s in ORDER:
        xs, ys = [], []
        for b in sorted(FINAL):
            g = d[(d.sampler == s) & (d.budget == b)].value
            if len(g) < 4:                      # four folds or it is not plotted
                continue
            xs.append(b * 100)
            ys.append(g.mean())
        if len(xs) < 2:
            continue
        any_line = True
        x, y = np.array(xs), np.array(ys)
        lw = 2.4 if s == "random" else 1.5
        z = 5 if s == "random" else 3
        if len(x) >= 3:
            g_ = (np.geomspace(x.min(), x.max(), 300) if LOGX
                  else np.linspace(x.min(), x.max(), 300))
            ax.plot(g_, PchipInterpolator(x, y)(g_), color=COL[s], lw=lw, zorder=z)
        else:
            ax.plot(x, y, color=COL[s], lw=lw, zorder=z)
        ax.plot(x, y, "o", color=COL[s], ms=3.6, zorder=z + 1,
                mec="white", mew=0.6)
        if ceil is not None and len(x) and x[-1] == 50:
            # Every strategy is carried to the shared 100% value. At beta=1 there
            # is no selection left to make - all five arms would label the whole
            # pool - so the five dashed segments converge on one point, and that
            # convergence is the thing worth showing. Dashed because that run
            # also used a different epoch count (50 rather than 60), so the
            # segment spans a gap in the schedule as well as in the budget.
            ax.plot([50, 100], [y[-1], ceil], color=COL[s], lw=1.0,
                    ls=(0, (3, 2)), zorder=z, alpha=0.75)
    if ceil is not None:
        ax.plot([100], [ceil], marker="D", ms=5.2, mfc="white",
                mec=COL["random"], mew=1.5, zorder=7)
    if LOGX:
        ax.set_xscale("log")
    ticks = [5, 10, 30, 50] + ([100] if ceil is not None else [])
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
    ax.minorticks_off()
    ax.tick_params(labelsize=7, width=0.8, length=2.6, pad=1.5)
    if title:
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)
    if show_ylab:
        ax.set_ylabel(ylab, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if not any_line:
        ax.text(0.5, 0.5, "not yet run", transform=ax.transAxes, ha="center",
                va="center", fontsize=8, color="0.55")
        ax.set_xticks([])
        ax.set_yticks([])


def split(ticks=True, only=None) -> int:
    """The same four cells as separate transparent PNGs.

    Titles, axis labels and the legend are dropped so they can be typeset in a
    layout tool. Tick numbers are kept by default: they are data, not captions,
    and a reader cannot judge whether a gap of one line-width is 0.002 or 0.02
    without them. Pass ticks=False for bare panels, in which case the ranges are
    printed below so the scale can be lettered by hand.
    """
    cohorts = [("HER2", "HER2"), ("cscc", "cSCC"), ("Kidney", "Kidney")]
    # fixed numbers: the first six are already placed in the layout and
    # must keep their filenames; Kidney is appended, never inserted
    NUM = {("pcc", "HER2"): 1, ("pcc", "cSCC"): 2,
           ("moran", "HER2"): 3, ("moran", "cSCC"): 4,
           ("ari", "HER2"): 5, ("ari", "cSCC"): 6,
           ("pcc", "Kidney"): 7, ("moran", "Kidney"): 8,
           ("ari", "Kidney"): 9}
    # ARI is appended rather than inserted after PCC so that the numbers already
    # placed in the layout keep their filenames: 1-2 PCC, 3-4 Moran, 5-6 ARI.
    rows = [("pcc", pcc_table, "Per-slide PCC"),
            ("moran", moran_table, "Moran's I fidelity"),
            ("ari", ari_table, "Expression-cluster agreement (ARI)")]
    n = 0
    for ri, (key, table, ylab) in enumerate(rows):
        for ci, (ck, cname) in enumerate(cohorts):
            n += 1
            if only and only not in (key, f"{key}_{cname}", str(n)):
                continue
            d = table(ck)
            fig, ax = plt.subplots(figsize=(5.6 * CM, 4.4 * CM))
            # No cohort title inside the panel - it is typeset in the layout
            # alongside the row and column labels. The filename still carries
            # the identity, and the ranges printed below distinguish the two
            # cohorts if a panel is ever separated from its name.
            draw(ax, d, "", "", False, ceil=ceiling(ck, key))
            if not ticks:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            f = OUT / f"panel_d_{NUM[(key, cname)]}_{key}_{cname}.png"
            for ext in ("png", "svg", "pdf"):
                fig.savefig(f.with_suffix("." + ext))
            lo, hi = ax.get_ylim()
            plt.close(fig)
            print(f"  {f.name:28s} {ylab:20s} {cname:5s} "
                  f"y {lo:.4f}..{hi:.4f}   x 5,10,30,50%")
            for s in ORDER:
                pts = [(int(b * 100), d[(d.sampler == s) & (d.budget == b)].value.mean())
                       for b in sorted(FINAL)
                       if len(d[(d.sampler == s) & (d.budget == b)]) == 4]
                if len(pts) >= 2:
                    print(f"      {LABEL[s]:11s} " +
                          "  ".join(f"{b}%={v:.4f}" for b, v in pts))

    # legend on its own, since the panels no longer carry one
    handles = [plt.Line2D([], [], color=COL[s], lw=2.4 if s == "random" else 1.5,
                          marker="o", ms=3.6, mec="white", mew=0.6, label=LABEL[s])
               for s in ORDER]
    fig = plt.figure(figsize=(11 * CM, 0.9 * CM))
    fig.legend(handles=handles, frameon=False, fontsize=7.5, ncol=5,
               loc="center", handlelength=1.6, columnspacing=1.4)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"panel_d_legend.{ext}")
    plt.close(fig)
    print("  panel_d_legend.png  (the only piece with words on it)")
    return 0


def main() -> int:
    global LOGX
    if "--linearx" in sys.argv:
        LOGX = False
    if "--only" in sys.argv:
        return split(ticks=True, only=sys.argv[sys.argv.index("--only") + 1])
    if "--split" in sys.argv:
        return split(ticks="--bare" not in sys.argv)
    cohorts = [("HER2", "HER2"), ("cscc", "cSCC")]
    fig, axes = plt.subplots(2, len(cohorts),
                             figsize=(len(cohorts) * 5.6 * CM, 9.4 * CM))
    for j, (key, name) in enumerate(cohorts):
        draw(axes[0, j], pcc_table(key), name, "Per-slide PCC", j == 0,
             ceil=ceiling(key, "pcc"))
        draw(axes[1, j], moran_table(key), name, "Moran's $I$ fidelity", j == 0,
             ceil=ceiling(key, "moran"))
    for j in range(len(cohorts)):
        axes[1, j].set_xlabel("Labelled spots (%)", fontsize=8)

    # Row captions go in figure coordinates, clear of the axes. At a fixed axes
    # offset they landed on top of the y-axis labels, whose width depends on how
    # many digits the tick values happen to need.
    fig.canvas.draw()
    left = min(ax.get_tightbbox().x0 for ax in axes.ravel()) / fig.bbox.width
    for row, cap in ((0, "Gene-expression\nprediction"),
                     (1, "Spatial-expression\narchitecture")):
        bb = axes[row, 0].get_position()
        fig.text(left - 0.05, bb.y0 + bb.height / 2, cap, rotation=90,
                 ha="center", va="center", fontsize=8.5, fontweight="bold")

    handles = [plt.Line2D([], [], color=COL[s], lw=2.4 if s == "random" else 1.5,
                          marker="o", ms=3.6, mec="white", mew=0.6, label=LABEL[s])
               for s in ORDER]
    fig.legend(handles=handles, frameon=False, fontsize=7.5, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.055),
               handlelength=1.6, columnspacing=1.4)
    fig.subplots_adjust(hspace=0.42, wspace=0.34, left=0.16)

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"panel_d_targets.{ext}")
    plt.close(fig)
    print("wrote figures/assets/panel_d_targets.png / .svg / .pdf")

    for key, name in cohorts:
        for tab, lab in ((pcc_table(key), "PCC"), (moran_table(key), "Moran")):
            done = [int(b * 100) for s in ORDER for b in sorted(FINAL)
                    if len(tab[(tab.sampler == s) & (tab.budget == b)]) == 4 and s != "random"]
            nrand = sum(1 for b in sorted(FINAL)
                        if len(tab[(tab.sampler == "random") & (tab.budget == b)]) == 4)
            print(f"  {name:5s} {lab:6s} random {nrand}/4 budgets, "
                  f"{len(done)} complete non-random cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
