"""Spot-level prediction maps: ground truth against each acquisition strategy.

One panel per arm, spots drawn at their array coordinates and coloured by the
predicted level of a single gene, with the measured map first for reference.
Per-panel numbers are the slide's PCC and MSE over all 300 targets, so they match
the values in the manuscript's tables rather than being figure-only quantities.

Three choices worth stating, because each could otherwise smuggle in a
conclusion.

The display gene is chosen by the mean correlation across all five arms on that
slide. It is symmetric -- no arm is privileged, random included -- so whatever
ranking appears between panels is the data's. Choosing on ground truth alone was
tried first and picked a gene no arm predicts (every r within +-0.04), which
yields six panels of flat colour: honest, but it shows nothing.

All panels share one colour scale, taken from the measured map. Predictions are
strongly shrunk toward the mean -- typically a fifth to a third of the true
spread -- so on a shared scale the model panels look washed out next to the
truth. That washing out is the result, not a rendering fault, and giving each
panel its own scale would hide it. The predicted s.d. is printed under each
panel so the shrinkage is legible as a number too.

Slide choice matters more than it looks. Per-slide PCC ranges from -0.08 to
+0.57 across HER2 at a fixed budget, and the spread between patients is several
times the spread between strategies. SPA125 sits near the bottom (Patient_F
averages +0.024); SPA143 near the top (Patient_B, +0.458). Both are rendered so
the difference is visible rather than implied.

    python scripts/make_pred_maps.py                 # SPA125 and SPA143 at 5%
    python scripts/make_pred_maps.py SPA143 0.3
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ST = Path(r"C:\Users\zheyu\workspace\stnet_ref\ST-Net")
DI = Path(r"C:\Users\zheyu\workspace\stnet_ref\data_infor")
OUT = ROOT / "figures" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

CM = 1 / 2.54
EPOCH = {0.05: 110, 0.1: 80, 0.3: 70, 0.5: 60}
ARMS = ["random", "coreset", "mc_dropout", "tod", "typiclust"]
LABEL = {"random": "Random", "coreset": "CoreSet", "mc_dropout": "MC-dropout",
         "tod": "TOD", "typiclust": "TypiClust"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.dpi": 400, "savefig.transparent": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def locate(slide):
    for p in sorted(os.listdir(DI / "HER2")):
        f = DI / "HER2" / p / f"{slide}.npy"
        if f.exists():
            return p, f
    raise SystemExit(f"{slide} not found under {DI / 'HER2'}")


def fold_of(slide, budget):
    for fold in range(4):
        if (ST / "weights_st" / "HER2" / f"random_{budget}" / f"fold{fold}" /
                "gt" / f"{slide}_gt.npy").exists():
            return fold
    raise SystemExit(f"no stored fold holds out {slide}")


def load(slide, budget):
    fold = fold_of(slide, budget)
    base = ST / "weights_st" / "HER2"
    gt = np.load(base / f"random_{budget}" / f"fold{fold}" / "gt" / f"{slide}_gt.npy")
    preds = {}
    for a in ARMS:
        f = (base / f"{a}_{budget}" / f"fold{fold}" / "pred" /
             f"epoch_{EPOCH[budget]}" / f"{slide}_pred.npy")
        if f.exists():
            preds[a] = np.load(f)
    return fold, gt, preds


def pick_gene(gt, preds):
    """Highest mean correlation across arms -- symmetric between strategies."""
    G = gt.shape[1]
    R = []
    for a, p in preds.items():
        r = np.full(G, np.nan)
        for g in range(G):
            if p[:, g].std() > 1e-9 and gt[:, g].std() > 1e-9:
                r[g] = np.corrcoef(p[:, g], gt[:, g])[0, 1]
        R.append(r)
    return int(np.nanargmax(np.nanmean(np.vstack(R), axis=0)))


def slide_pcc(p, gt):
    """Gene-wise Pearson averaged over genes -- the manuscript's definition."""
    x = p - p.mean(0, keepdims=True)
    y = gt - gt.mean(0, keepdims=True)
    den = np.sqrt((x ** 2).sum(0) * (y ** 2).sum(0)) + 1e-8
    return float(((x * y).sum(0) / den).mean())


def draw(slide, budget, gene=None):
    patient, recf = locate(slide)
    fold, gt, preds = load(slide, budget)
    rec = np.load(recf, allow_pickle=True)
    pos = np.array([r["position"] for r in rec], dtype=float)
    if len(pos) != len(gt):
        raise SystemExit(f"{slide}: {len(pos)} spots but {len(gt)} gt rows")

    names = [str(x) for x in np.load(
        ROOT / "processed_gene" / "gene_names" / "HER2_expression_norm_genes.npy",
        allow_pickle=True)[:gt.shape[1]]]
    if gene is None:
        g = pick_gene(gt, preds)
    elif gene in names:
        g = names.index(gene)
    else:
        # Two different failures, and the distinction decides whether anything
        # can be done about it. A gene outside the target panel exists in the
        # cohort but was never a prediction target, so no stored prediction
        # holds it -- only retraining with a different panel would help. A gene
        # missing from the cohort entirely was filtered upstream.
        full = [str(x) for x in np.load(
            ROOT / "processed_gene" / "gene_names" /
            "HER2_expression_norm_genes.npy", allow_pickle=True)]
        why = (f"present in the cohort at index {full.index(gene)} but outside "
               f"the {len(names)}-gene target panel, so it was never predicted"
               if gene in full else "not in this cohort's expression matrix at all")
        raise SystemExit(f"{gene}: {why}")

    vmin, vmax = gt[:, g].min(), gt[:, g].max()
    panels = [("Measured", gt)] + [(LABEL[a], preds[a]) for a in ARMS if a in preds]

    # `adjustable="datalim"` keeps every axes box exactly the subplot rectangle
    # and stretches the limits instead. The default, "box", resizes the box at
    # draw time to match the tissue's aspect, which makes transAxes mean a
    # different thing in each panel -- captions then sat at visibly different
    # distances from the spots between the two rows. Keeping the box fixed only
    # works if the box already has the tissue's proportions, so the figure is
    # sized from them rather than assumed square; otherwise the caption falls
    # into the whitespace below the tissue and collides with the next row.
    x0, x1 = pos[:, 0].min(), pos[:, 0].max()
    y0, y1 = -pos[:, 1].max(), -pos[:, 1].min()
    pad = 0.06 * max(x1 - x0, y1 - y0)
    ar = (x1 - x0 + 2 * pad) / (y1 - y0 + 2 * pad)

    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    pw = 5.0                                   # panel width, cm
    ph = pw / ar                               # panel height that fits the tissue
    cap, top = 1.05, 1.15                      # caption strip, suptitle strip
    fig_w, fig_h = ncol * pw / 0.86, nrow * (ph + cap) + top
    fig = plt.figure(figsize=(fig_w * CM, fig_h * CM))
    gs = fig.add_gridspec(nrow, ncol, left=0.015, right=0.855,
                          top=1 - top / fig_h, bottom=cap / fig_h * 0.35,
                          wspace=0.05, hspace=cap / ph)
    axes = np.array([fig.add_subplot(gs[i // ncol, i % ncol])
                     for i in range(nrow * ncol)])

    # one marker per array position, sized to the grid pitch: panel width in
    # points divided by the number of columns the array actually occupies
    nx = max(len(np.unique(pos[:, 0])), 1)
    ms = ((pw / 2.54 * 72) / nx * 0.82) ** 2

    for ax, (name, mat) in zip(axes, panels):
        sc = ax.scatter(pos[:, 0], -pos[:, 1], c=mat[:, g], s=ms, marker="o",
                        cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0)
        # Two scopes, labelled, because they are easy to conflate. The first
        # line is the displayed gene alone -- the convention the comparable
        # figures in the literature use. The second is over all targets and is
        # the number that appears in the manuscript's tables.
        if name == "Measured":
            sub = f"{names[g]} s.d. {gt[:, g].std():.2f}"
        else:
            r1 = np.corrcoef(mat[:, g], gt[:, g])[0, 1] if mat[:, g].std() > 0 else np.nan
            sub = (f"{names[g]}: MSE {((mat[:, g] - gt[:, g]) ** 2).mean():.3f}"
                   f"   r {r1:+.2f}   s.d. {mat[:, g].std():.2f}\n"
                   f"all {gt.shape[1]} genes: PCC {slide_pcc(mat, gt):.3f}"
                   f"   MSE {((mat - gt) ** 2).mean():.3f}")
        ax.set_title(name, fontsize=8.5, fontweight="bold", pad=3)
        ax.text(0.5, -0.02, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=6.4, color="0.35", linespacing=1.4)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(f"{slide} ({patient}, fold {fold}) — {names[g]} at "
                 f"{budget:.0%} of spots", fontsize=9.5, fontweight="bold",
                 y=1 - 0.30 * top / fig_h)
    # placed by hand: letting colorbar() steal space from the axes would undo
    # the gridspec proportions the caption spacing depends on
    cax = fig.add_axes([0.885, 0.22, 0.018, 0.52])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(f"{names[g]}  $\\log(1+x)$", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, width=0.7, length=2)
    cb.outline.set_visible(False)

    stem = f"predmap_{slide}_{names[g]}_{int(budget * 100)}"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)

    print(f"{slide} ({patient}, fold {fold}), gene {names[g]}, budget {budget:.0%}")
    print(f"  measured s.d. {gt[:, g].std():.3f}")
    for a in ARMS:
        if a in preds:
            print(f"  {LABEL[a]:11s} PCC {slide_pcc(preds[a], gt):+.4f}   "
                  f"MSE {((preds[a] - gt) ** 2).mean():.4f}   "
                  f"pred s.d. {preds[a][:, g].std():.3f}")
    print(f"  -> figures/assets/{stem}.png / .svg / .pdf\n")


def split(slide, budget, gene=None, cmap="viridis"):
    """The same six maps as separate transparent PNGs, for hand assembly.

    No titles or numbers are burned in: a panel that has to be moved, resized or
    relabelled in a layout tool is easier to work with clean, and the values are
    printed here instead. The colour scale is still shared and still taken from
    the measured map, so the panels remain comparable once placed side by side.
    """
    patient, recf = locate(slide)
    fold, gt, preds = load(slide, budget)
    rec = np.load(recf, allow_pickle=True)
    pos = np.array([r["position"] for r in rec], dtype=float)
    names = [str(x) for x in np.load(
        ROOT / "processed_gene" / "gene_names" / "HER2_expression_norm_genes.npy",
        allow_pickle=True)[:gt.shape[1]]]
    g = names.index(gene) if gene else pick_gene(gt, preds)

    vmin, vmax = gt[:, g].min(), gt[:, g].max()
    panels = [("measured", gt)] + [(a, preds[a]) for a in ARMS if a in preds]
    x0, x1 = pos[:, 0].min(), pos[:, 0].max()
    y0, y1 = -pos[:, 1].max(), -pos[:, 1].min()
    pad = 0.06 * max(x1 - x0, y1 - y0)
    nx = max(len(np.unique(pos[:, 0])), 1)
    pw = 6.0                                        # cm per panel, standalone

    print(f"{slide} ({patient}, fold {fold}), {names[g]}, {budget:.0%} -- six panels")
    stem = f"predmap_{slide}_{names[g]}_{int(budget * 100)}"
    for i, (name, mat) in enumerate(panels):
        fig = plt.figure(figsize=(pw * CM, pw * CM * (y1 - y0) / (x1 - x0)))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.scatter(pos[:, 0], -pos[:, 1], c=mat[:, g],
                   s=((pw / 2.54 * 72) / nx * 0.82) ** 2, marker="o", cmap=cmap,
                   vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
        f = OUT / f"{stem}_{i}_{name}.png"
        fig.savefig(f)
        plt.close(fig)
        if name == "measured":
            print(f"  {f.name:44s} s.d. {gt[:, g].std():.2f}")
        else:
            r1 = np.corrcoef(mat[:, g], gt[:, g])[0, 1] if mat[:, g].std() > 0 else np.nan
            print(f"  {f.name:44s} {names[g]} MSE "
                  f"{((mat[:, g] - gt[:, g]) ** 2).mean():.3f}  r {r1:+.2f}  |  "
                  f"all {gt.shape[1]}: PCC {slide_pcc(mat, gt):.3f}  "
                  f"MSE {((mat - gt) ** 2).mean():.3f}")

    # a standalone scale bar, since none of the panels carries one
    fig = plt.figure(figsize=(1.1 * CM, pw * CM * 0.75))
    cax = fig.add_axes([0.02, 0.04, 0.30, 0.92])
    fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin, vmax), cmap=cmap), cax=cax)
    cax.tick_params(labelsize=7, width=0.7, length=2.2)
    cax.set_ylabel(f"{names[g]}  $\\log(1+x)$", fontsize=7.5)
    cax.yaxis.set_label_position("left")
    for sp in cax.spines.values():
        sp.set_visible(False)
    fig.savefig(OUT / f"{stem}_colorbar.png")
    plt.close(fig)
    print(f"  {stem}_colorbar.png")


def main() -> int:
    a = [x for x in sys.argv[1:] if x != "--split"]
    if "--split" in sys.argv:
        split(a[0] if a else "SPA143",
              float(a[1]) if len(a) > 1 else 0.05,
              a[2] if len(a) > 2 else None)
    elif a:
        draw(a[0], float(a[1]) if len(a) > 1 else 0.05,
             a[2] if len(a) > 2 else None)
    else:
        for slide in ("SPA125", "SPA143"):
            draw(slide, 0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
