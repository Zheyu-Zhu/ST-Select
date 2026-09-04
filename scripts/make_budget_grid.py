"""The 5x5 cell images for the strategy-by-budget grid, one file per cell.

Rows are the five acquisition strategies, columns are the four budgets plus the
measured map. Every cell is written separately, without titles or axes, so the
grid can be assembled and labelled in a layout tool.

Three things are held constant so the cells are comparable once placed. All 25
share one colour scale, taken from the measured map, which is why the prediction
cells look flatter than the ground-truth column: that shrinkage is the result,
not a rendering choice. All share one spot size and one axis range, so a cell
can be swapped without the tissue changing size. And the ground-truth column is
the same image in every row, since the measurement does not depend on the
strategy; it is written five times only because the grid has five slots for it.

One cell has no data. TypiClust's 50% budget finished on fold 0 alone, and
SPA143 is held out in fold 1, so that combination was never predicted for this
slide. A placeholder is written for it rather than silently leaving a gap.

    python scripts/make_budget_grid.py
    python scripts/make_budget_grid.py --gene ERBB2 --slide SPA143
"""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ST = Path(r"C:\Users\zheyu\workspace\stnet_ref\ST-Net")
DI = Path(r"C:\Users\zheyu\workspace\stnet_ref\data_infor")
OUT = ROOT / "figures" / "assets" / "grid"
OUT.mkdir(parents=True, exist_ok=True)
CM = 1 / 2.54

EPOCH = {0.05: 110, 0.1: 80, 0.3: 70, 0.5: 60}
ARMS = ["random", "coreset", "mc_dropout", "tod", "typiclust"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.dpi": 400, "savefig.transparent": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def fold_of(slide):
    for f in range(4):
        if list((ST / "weights_st" / "HER2").glob(f"*/fold{f}/gt/{slide}_gt.npy")):
            return f
    raise SystemExit(f"{slide}: not held out in any stored fold")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", default="SPA143")
    ap.add_argument("--gene", default="COL1A1")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--width", type=float, default=4.4, help="cell width, cm")
    ap.add_argument("--missing", choices=["blank", "silhouette", "rings"],
                    default="blank",
                    help="how to render a cell that was never run. The first "
                         "attempt drew open rings, which reads as a broken "
                         "render rather than as absent data; blank leaves the "
                         "cell empty for the caption to explain, silhouette "
                         "keeps the tissue outline in flat grey.")
    args = ap.parse_args()

    fold = fold_of(args.slide)
    names = [str(x) for x in np.load(
        ROOT / "processed_gene" / "gene_names" / "HER2_expression_norm_genes.npy",
        allow_pickle=True)]
    gt = np.load(next((ST / "weights_st" / "HER2").glob(
        f"*/fold{fold}/gt/{args.slide}_gt.npy")))
    if args.gene not in names[:gt.shape[1]]:
        why = (f"outside the {gt.shape[1]}-gene target panel"
               if args.gene in names else "absent from this cohort")
        raise SystemExit(f"{args.gene}: {why}, so the model never predicts it")
    gi = names.index(args.gene)

    rec = np.load(glob.glob(str(DI / "HER2" / "*" / f"{args.slide}.npy"))[0],
                  allow_pickle=True)
    pos = np.array([r["position"] for r in rec], dtype=float)
    x, y = pos[:, 0], -pos[:, 1]
    pad = 0.06 * max(np.ptp(x), np.ptp(y))
    nx = max(len(np.unique(x)), 1)
    ms = ((args.width / 2.54 * 72) / nx * 0.82) ** 2
    vmin, vmax = float(gt[:, gi].min()), float(gt[:, gi].max())
    hgt = args.width * (np.ptp(y) + 2 * pad) / (np.ptp(x) + 2 * pad)

    def cell(values, name):
        fig = plt.figure(figsize=(args.width * CM, hgt * CM))
        ax = fig.add_axes([0, 0, 1, 1])
        if values is not None:
            ax.scatter(x, y, c=values, cmap=args.cmap, vmin=vmin, vmax=vmax,
                       s=ms, marker="o", linewidths=0)
        elif args.missing == "silhouette":
            ax.scatter(x, y, c="#E4E6EA", s=ms, marker="o", linewidths=0)
        elif args.missing == "rings":
            ax.scatter(x, y, facecolors="none", edgecolors="#D7DAE0",
                       s=ms, marker="o", linewidths=0.5)
        ax.set_xlim(x.min() - pad, x.max() + pad)
        ax.set_ylim(y.min() - pad, y.max() + pad)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
        fig.savefig(OUT / f"{name}.png")
        plt.close(fig)

    stem = f"grid_{args.slide}_{args.gene}"
    print(f"{args.slide}, fold {fold} held out, {args.gene}, "
          f"colour scale {vmin:.2f}–{vmax:.2f} shared by all cells\n")
    print(f"{'cell':34s} {'gene r':>7s} {'gene MSE':>9s} {'300-gene PCC':>13s} {'pred s.d.':>10s}")

    n_ok = n_missing = 0
    for a in ARMS:
        for b in (0.05, 0.1, 0.3, 0.5):
            f = (ST / "weights_st" / "HER2" / f"{a}_{b}" / f"fold{fold}" /
                 "pred" / f"epoch_{EPOCH[b]}" / f"{args.slide}_pred.npy")
            name = f"{stem}_{a}_{int(b*100)}"
            if not f.exists():
                cell(None, name)
                print(f"  {name[len(stem)+1:]:32s} {'not run for this fold':>43s}")
                n_missing += 1
                continue
            p = np.load(f)
            cell(p[:, gi], name)
            r = np.corrcoef(p[:, gi], gt[:, gi])[0, 1] if p[:, gi].std() > 0 else np.nan
            xx, yy = p - p.mean(0), gt - gt.mean(0)
            den = np.sqrt((xx ** 2).sum(0) * (yy ** 2).sum(0)) + 1e-9
            pcc = float(((xx * yy).sum(0) / den).mean())
            print(f"  {name[len(stem)+1:]:32s} {r:+7.3f} "
                  f"{((p[:, gi]-gt[:, gi])**2).mean():9.3f} {pcc:13.4f} "
                  f"{p[:, gi].std():10.3f}")
            n_ok += 1
        # the measured column: identical in every row, written once per slot
        cell(gt[:, gi], f"{stem}_{a}_gt")

    # a colour bar for the whole grid
    fig = plt.figure(figsize=(1.15 * CM, hgt * 0.85 * CM))
    cax = fig.add_axes([0.04, 0.05, 0.30, 0.90])
    fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin, vmax), cmap=args.cmap), cax=cax)
    cax.tick_params(labelsize=6.5, width=0.7, length=2.2)
    cax.set_ylabel(f"{args.gene}  $\\log(1+x)$", fontsize=7)
    cax.yaxis.set_label_position("left")
    for sp in cax.spines.values():
        sp.set_visible(False)
    fig.savefig(OUT / f"{stem}_colorbar.png")
    plt.close(fig)

    print(f"\nmeasured s.d. {gt[:, gi].std():.3f}")
    print(f"{n_ok} prediction cells, {n_missing} placeholder, "
          f"{len(ARMS)} copies of the measured column, 1 colour bar")
    print(f"-> figures/assets/grid/{stem}_*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
