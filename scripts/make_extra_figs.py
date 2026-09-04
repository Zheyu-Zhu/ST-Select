"""The results that exist on disk but are not in the paper, drawn as candidates.

Six figures, each answering a different question, so they can be compared and
chosen between rather than described:

  cost          what each strategy charges per selection round, and how that
                scales with the budget. TypiClust is the whole story here and a
                linear axis hides it, so the axis is log.
  oversmooth    Moran's I of the predicted maps against the measured ones. Every
                arm predicts maps that are MORE spatially clustered than the
                truth, which the fidelity number alone does not show.
  converge      per-slide PCC against epoch, per budget. The low-budget arms are
                still climbing when training stops, which is the study's most
                attackable claim and deserves a picture.
  ari_nmi       the two clustering scores against each other. The paper asserts
                they track; this is the evidence for that sentence.
  patients      per-patient PCC. Between-patient spread dwarfs every strategy
                effect, which is currently one sentence in the limitations.
  depth         PCC before and after projecting out per-spot library size. About
                a fifth survives.

Colour is assigned by strategy identity and never by rank, so a figure that
drops a series does not repaint the others. Two palettes are available: the one
used elsewhere in these figures, and a colour-vision-safe replacement. The
original fails three of ten pairs under dichromat simulation (orange against
green at 5.5 in protanopia, where 8 is the target), so every line also carries a
marker shape and the legend is always present -- identity is never colour alone.

    python scripts/make_extra_figs.py
    python scripts/make_extra_figs.py --palette safe
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ST = Path(r"C:\Users\zheyu\workspace\stnet_ref\ST-Net")
DI = Path(r"C:\Users\zheyu\workspace\stnet_ref\data_infor")
OUT = ROOT / "figures" / "assets" / "extra"
OUT.mkdir(parents=True, exist_ok=True)
CM = 1 / 2.54

EPOCH = {0.05: 110, 0.1: 80, 0.3: 70, 0.5: 60}
ARMS = ["random", "coreset", "mc_dropout", "tod", "typiclust"]
LABEL = {"random": "Random", "coreset": "CoreSet", "mc_dropout": "MC-dropout",
         "tod": "TOD", "typiclust": "TypiClust"}
# second encoding channel, so identity never rests on colour alone
MARK = {"random": "o", "coreset": "s", "mc_dropout": "^", "tod": "D",
        "typiclust": "v"}
PAL = {
    "house": {"random": "#1C1B18", "coreset": "#D2601A", "mc_dropout": "#4B3F8F",
              "tod": "#3F7A52", "typiclust": "#C0407A"},
    "safe": {"random": "#1F1E1B", "coreset": "#9C0451", "mc_dropout": "#744F00",
             "tod": "#006C82", "typiclust": "#5F35AD"},
}
INK, MUTED, GRID = "#1C1B18", "#5A5A5A", "#DCDCDC"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0.7, "savefig.dpi": 400, "savefig.transparent": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(labelsize=7, width=0.7, length=2.6, pad=2, colors=MUTED)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(INK)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.png")


# ------------------------------------------------------------------ loaders
def slidewise(ds):
    rows = []
    for f in glob.glob(str(ST / "weights_st" / ds / "*" / "fold*" /
                           "slidewise_metrics" / "epoch_*.csv")):
        p = f.replace("\\", "/").split("/")
        s, r = p[-4].rsplit("_", 1)
        try:
            r = float(r)
        except ValueError:
            continue
        e = int(re.search(r"epoch_(\d+)", p[-1]).group(1))
        fold = int(re.search(r"fold(\d+)", p[-3]).group(1))
        d = pd.read_csv(f)
        d["sampler"], d["budget"], d["fold"], d["ep"] = s, r, fold, e
        rows.append(d)
    return pd.concat(rows) if rows else pd.DataFrame()


def cost_table():
    rows = []
    for f in glob.glob(str(ST / "logs" / "*_fold*.log")):
        m = re.match(r"(\w+?)_(random|coreset|mc_dropout|tod|typiclust)"
                     r"_([0-9.]+)_fold(\d)", os.path.basename(f))
        if not m:
            continue
        txt = open(f, encoding="utf-8", errors="replace").read()
        secs = [float(x) for x in re.findall(r"pool=\d+, ([0-9.]+)s", txt)]
        if secs:
            rows.append(dict(dataset=m.group(1), sampler=m.group(2),
                             budget=float(m.group(3)),
                             s=float(np.median(secs))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def fig_cost(C):
    d = cost_table()
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.6 * CM, 5.8 * CM), sharey=True)
    for ax, ds, name in zip(axes, ("HER2", "cscc"), ("HER2", "cSCC")):
        for a in ARMS:
            g = d[(d.sampler == a) & (d.dataset == ds)].groupby("budget").s.median()
            if g.empty:
                continue
            y = np.maximum(g.values, 0.3)      # Random is sub-second
            ax.plot(g.index * 100, y, color=C[a], marker=MARK[a], ms=4.5, lw=1.6,
                    mec="white", mew=0.7, label=LABEL[a], zorder=3)
        _cost_axis(ax, name)
    axes[0].set_ylabel("Median seconds per selection round", fontsize=8, color=INK)
    axes[0].legend(frameon=False, fontsize=6.6, loc="upper left",
                   handlelength=1.5, labelcolor=INK)
    fig.subplots_adjust(wspace=0.10)
    save(fig, "cost_per_round")


def _cost_axis(ax, name):
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 30, 50])
    ax.set_xticklabels(["5", "10", "30", "50"])
    ax.minorticks_off()
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1 s", "10 s", "100 s", "1000 s"])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel("Labelled spots (%)", fontsize=8, color=INK)
    ax.set_title(name, fontsize=9, fontweight="bold", color=INK, pad=4)
    tidy(ax)


def fig_oversmooth(C):
    fig, axes = plt.subplots(1, 2, figsize=(13.0 * CM, 5.6 * CM))
    for ax, ds, name in zip(axes, ("HER2", "cscc"), ("HER2", "cSCC")):
        p = ST / f"{ds}_spatial_metrics.csv"
        if not p.exists():
            continue
        m = pd.read_csv(p)
        t = m.groupby("budget")[["moran_true", "moran_pred"]].mean()
        ax.plot(t.index * 100, t.moran_true, color=MUTED, marker="o", ms=4.5,
                lw=1.6, mec="white", mew=0.7, label="Measured", zorder=3)
        ax.plot(t.index * 100, t.moran_pred, color=C["coreset"], marker="s",
                ms=4.5, lw=1.6, mec="white", mew=0.7, label="Predicted", zorder=3)
        ax.fill_between(t.index * 100, t.moran_true, t.moran_pred,
                        color=C["coreset"], alpha=0.12, lw=0, zorder=1)
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 30, 50])
        ax.set_xticklabels(["5", "10", "30", "50"])
        ax.minorticks_off()
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_title(name, fontsize=9, fontweight="bold", color=INK, pad=4)
        ax.set_xlabel("Labelled spots (%)", fontsize=8, color=INK)
        tidy(ax)
    axes[0].set_ylabel("Moran's $I$, averaged over genes", fontsize=8, color=INK)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right", labelcolor=INK)
    axes[0].text(0.03, 0.94, "predictions are more spatially\nclustered than the truth",
                 transform=axes[0].transAxes, fontsize=6.6, color=MUTED,
                 va="top", linespacing=1.4)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "oversmoothing")


def fig_converge(C, ds="HER2", name="HER2"):
    d = slidewise(ds)
    if d.empty:
        return
    fig, axes = plt.subplots(1, 4, figsize=(17.0 * CM, 5.0 * CM), sharey=True)
    for ax, b in zip(axes, EPOCH):
        for a in ARMS:
            sub_ = d[(d.sampler == a) & (d.budget == b)]
            # four folds or the arm is not drawn: averaging over whichever
            # folds happen to exist put a single-fold line on this panel and
            # made TypiClust look like it collapsed at 50%
            if sub_.fold.nunique() < 4:
                continue
            g = sub_.groupby("ep").pcc.mean()
            if len(g) < 2:
                continue
            ax.plot(g.index, g.values, color=C[a], marker=MARK[a], ms=3.2,
                    lw=1.4, mec="white", mew=0.5, label=LABEL[a], zorder=3)
        ax.set_title(f"{int(b*100)}% budget", fontsize=8.5, fontweight="bold",
                     color=INK, pad=4)
        ax.set_xlabel("Epoch", fontsize=8, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        tidy(ax)
    axes[0].set_ylabel("Per-slide PCC", fontsize=8, color=INK)
    axes[0].legend(frameon=False, fontsize=6.2, loc="upper left",
                   handlelength=1.3, labelcolor=INK)
    axes[-1].text(0.96, 0.06, "flat well before the end",
                  transform=axes[-1].transAxes, ha="right", fontsize=6.2,
                  color=MUTED)
    axes[0].text(0.96, 0.06, "still rising at the last epoch",
                 transform=axes[0].transAxes, ha="right", fontsize=6.2,
                 color=MUTED)
    fig.suptitle(name, fontsize=9, fontweight="bold", color=INK, y=1.02)
    fig.subplots_adjust(wspace=0.12)
    save(fig, f"convergence_{name}")


def fig_ari_nmi(C):
    fig, ax = plt.subplots(figsize=(7.4 * CM, 6.6 * CM))
    for ds, mk in (("HER2", "o"), ("cscc", "s")):
        p = ST / f"{ds}_spatial_metrics.csv"
        if not p.exists():
            continue
        m = pd.read_csv(p)
        for a in ARMS:
            g = m[m.sampler == a]
            ax.scatter(g.ari, g.nmi, c=C[a], marker=mk, s=18, linewidths=0.5,
                       edgecolors="white", zorder=3,
                       label=LABEL[a] if ds == "HER2" else None)
    both = pd.concat([pd.read_csv(ST / f"{d}_spatial_metrics.csv")
                      for d in ("HER2", "cscc")
                      if (ST / f"{d}_spatial_metrics.csv").exists()])
    r = np.corrcoef(both.ari, both.nmi)[0, 1]
    lo = min(both.ari.min(), both.nmi.min())
    hi = max(both.ari.max(), both.nmi.max())
    ax.plot([lo, hi], [lo, hi], color=GRID, lw=0.8, zorder=1)
    ax.set_xlabel("Adjusted Rand index", fontsize=8, color=INK)
    ax.set_ylabel("Normalised mutual information", fontsize=8, color=INK)
    ax.set_title(f"ARI and NMI agree  ($r = {r:.3f}$)", fontsize=9,
                 fontweight="bold", color=INK, pad=4)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    tidy(ax)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left", labelcolor=INK)
    ax.text(0.97, 0.05, "circles HER2, squares cSCC", transform=ax.transAxes,
            ha="right", fontsize=6.4, color=MUTED)
    save(fig, "ari_vs_nmi")


def fig_patients(C, ds="HER2", name="HER2"):
    d = slidewise(ds)
    if d.empty:
        return
    d = d[d.ep == d.budget.map(EPOCH)]
    pat = {}
    for p in sorted(os.listdir(DI / ds)):
        for f in glob.glob(str(DI / ds / p / "*.npy")):
            pat[os.path.basename(f)[:-4]] = (p.replace("Patient_", "")
                                             .replace("cscc_patient", "P"))
    d["patient"] = d.slide.map(pat)
    g = d[d.sampler == "random"].groupby("patient").pcc
    order = g.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(9.0 * CM, 5.6 * CM))
    for i, p in enumerate(order):
        v = d[(d.sampler == "random") & (d.patient == p)].pcc.values
        ax.scatter(np.full(len(v), i), v, s=9, c=MUTED, alpha=0.45,
                   linewidths=0, zorder=2)
        ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=C["coreset"],
                lw=2.0, zorder=4, solid_capstyle="butt")
    ax.axhline(0, color=GRID, lw=0.8, zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=7)
    ax.set_xlabel(f"{name} patient", fontsize=8, color=INK)
    ax.set_ylabel("Per-slide PCC", fontsize=8, color=INK)
    ax.set_title(f"Between-patient spread, Random baseline — {name}",
                 fontsize=9, fontweight="bold", color=INK, pad=4)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    tidy(ax)
    lo, hi = g.mean().min(), g.mean().max()
    ax.text(0.02, 0.95, f"patient means span {lo:+.3f} to {hi:+.3f};\n"
                        f"strategy effects are ~0.02",
            transform=ax.transAxes, va="top", fontsize=6.6, color=MUTED,
            linespacing=1.4)
    save(fig, f"patient_spread_{name}")


def fig_depth(C, ds="HER2", name="HER2"):
    """PCC before and after projecting per-spot library size out of both sides."""
    rows = []
    depth = {}
    for p in sorted(os.listdir(DI / ds)):
        for f in glob.glob(str(DI / ds / p / "*.npy")):
            rec = np.load(f, allow_pickle=True)
            v = np.log1p(np.expm1(np.stack([r["expression_norm"] for r in rec]))
                         .sum(1))
            depth[os.path.basename(f)[:-4]] = (v - v.mean()) / (v.std() + 1e-12)

    def resid(M, dd):
        Z = M - M.mean(0)
        return Z - np.outer(dd, (Z * dd[:, None]).mean(0))

    def pcc(A, B):
        x, y = A - A.mean(0), B - B.mean(0)
        den = np.sqrt((x ** 2).sum(0) * (y ** 2).sum(0)) + 1e-9
        return float(((x * y).sum(0) / den).mean())

    for b, ep in EPOCH.items():
        for fold in range(4):
            pdir = ST / "weights_st" / ds / f"random_{b}" / f"fold{fold}" / \
                "pred" / f"epoch_{ep}"
            if not pdir.is_dir():
                continue
            for pf in pdir.glob("*_pred.npy"):
                s = pf.name.replace("_pred.npy", "")
                gf = pdir.parent.parent / "gt" / f"{s}_gt.npy"
                if not gf.exists() or s not in depth:
                    continue
                gt, pr = np.load(gf), np.load(pf)
                dd = depth[s]
                rows.append(dict(budget=b, raw=pcc(pr, gt),
                                 free=pcc(resid(pr, dd), resid(gt, dd))))
    if not rows:
        return
    d = pd.DataFrame(rows).groupby("budget")[["raw", "free"]].mean()
    fig, ax = plt.subplots(figsize=(8.0 * CM, 5.6 * CM))
    x = np.arange(len(d))
    w = 0.34
    ax.bar(x - w / 2 - 0.01, d.raw, w, color=MUTED, label="As reported", zorder=3)
    ax.bar(x + w / 2 + 0.01, d.free, w, color=C["coreset"],
           label="Sequencing depth removed", zorder=3)
    for i, (r, f) in enumerate(zip(d.raw, d.free)):
        ax.text(i + w / 2 + 0.01, f + 0.004, f"{f/r:.0%}", ha="center",
                fontsize=6.4, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b*100)}%" for b in d.index])
    ax.set_xlabel("Labelled spots", fontsize=8, color=INK)
    ax.set_ylabel("Per-slide PCC", fontsize=8, color=INK)
    ax.set_title(f"How much of the correlation is gene-specific? — {name}",
                 fontsize=9, fontweight="bold", color=INK, pad=4)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    tidy(ax)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", labelcolor=INK)
    save(fig, f"depth_confound_{name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", choices=["house", "safe"], default="house")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    C = PAL[args.palette]
    print(f"palette: {args.palette}")
    jobs = [(fig_cost, ()), (fig_oversmooth, ()), (fig_ari_nmi, ())]
    for ds, name in (("HER2", "HER2"), ("cscc", "cSCC")):
        jobs += [(fig_converge, (ds, name)), (fig_patients, (ds, name)),
                 (fig_depth, (ds, name))]
    for fn, a in jobs:
        try:
            fn(C, *a)
        except Exception as e:
            print(f"  {fn.__name__}{a} failed: {type(e).__name__}: {e}")
    print(f"-> figures/assets/extra/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
