"""The predictor as a standalone graphic, for placing in a figure by hand.

Three styles, because the right amount of detail depends on where it goes:

  wedge     a narrowing trunk and the regression head. The default, and what
            the paper's position calls for: the predictor is fixed and shared
            by every arm, so it should read as a component, not a contribution.
  blocks    the same, with DenseNet-121's four dense blocks drawn separately.
            Factually true of the architecture and useful where the panel has
            room, though it invites more attention than the model deserves here.
  box       a plain labelled rectangle, for when the figure only needs to say
            "a model happens here".

Drawn in matplotlib so the SVG is real geometry rather than a raster, on a
transparent ground. Labels can be switched off to be set in a layout tool.

    python scripts/make_model_icon.py
    python scripts/make_model_icon.py --style blocks
    python scripts/make_model_icon.py --style box --no-labels
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
CM = 1 / 2.54

INK = "#1C1B18"
BOX = "#6F86A8"
DEEP = "#4E6684"        # a darker step of the same hue, for alternating blocks

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.dpi": 400, "savefig.transparent": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def head(ax, x0, cy, labels):
    """The regression head: 1024 units, then the 300 outputs."""
    for k, (w, h, lab) in enumerate(((3.6, 11.0, "1024"),
                                     (3.6, 11.0, ""),
                                     (3.6, 16.0, "300"))):
        x = x0 + k * 4.8
        ax.add_patch(FancyBboxPatch((x, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.12,rounding_size=0.5",
                                    fc="white", ec=BOX, lw=1.1, zorder=6))
        if lab and labels:
            ax.text(x + w / 2, cy, lab, ha="center", va="center", zorder=7,
                    fontsize=5.6, color=BOX, rotation=90)
    return x0 + 2 * 4.8 + 3.6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style",
                    choices=["wedge", "blocks", "box", "layers", "cloud"],
                    default="wedge")
    ap.add_argument("--labels", dest="labels", action="store_true", default=True)
    ap.add_argument("--no-labels", dest="labels", action="store_false")
    ap.add_argument("--caption", action="store_true",
                    help="add the one-line architecture caption underneath")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    W, H = (58.0, 30.0) if args.style != "box" else (34.0, 24.0)
    if args.caption:
        H += 7.0
    fig = plt.figure(figsize=(W / 10 * CM * 1.6, H / 10 * CM * 1.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    cy = H - 15.0 if args.caption else H / 2

    if args.style == "wedge":
        ax.add_patch(Polygon([[3, cy - 11], [3, cy + 11],
                              [20, cy + 7], [20, cy - 7]], closed=True,
                             fc=BOX, ec="none", alpha=0.95, zorder=6))
        if args.labels:
            ax.text(11.5, cy, "DenseNet-121", ha="center", va="center",
                    zorder=7, fontsize=6.4, color="white", fontweight="bold")
        head(ax, 25.0, cy, args.labels)

    elif args.style == "blocks":
        # DenseNet-121 really is four dense blocks; the taper is the pooling
        xs = [3.0, 8.4, 13.8, 19.2]
        hs = [22.0, 19.0, 16.0, 13.0]
        for i, (x, h) in enumerate(zip(xs, hs)):
            ax.add_patch(FancyBboxPatch((x, cy - h / 2), 4.2, h,
                                        boxstyle="round,pad=0.1,rounding_size=0.4",
                                        fc=BOX if i % 2 == 0 else DEEP,
                                        ec="none", alpha=0.95, zorder=6))
        if args.labels:
            ax.text((xs[0] + xs[-1] + 4.2) / 2, cy - 13.5,
                    "four dense blocks", ha="center", va="top",
                    fontsize=5.4, color=INK)
        head(ax, 27.0, cy, args.labels)

    elif args.style == "layers":
        # Nodes and edges, but arranged as layers running left to right, which
        # reads as a feed-forward network. That is what this predictor is.
        cols = [(6.0, 5), (19.0, 6), (32.0, 4), (45.0, 3)]
        pts = []
        for x, n in cols:
            span = 20.0
            ys = ([cy] if n == 1 else
                  [cy - span / 2 + i * span / (n - 1) for i in range(n)])
            pts.append([(x, y) for y in ys])
        for a, b in zip(pts, pts[1:]):
            for x0, y0 in a:
                for x1, y1 in b:
                    ax.plot([x0, x1], [y0, y1], color=BOX, lw=0.45,
                            alpha=0.55, zorder=4, solid_capstyle="round")
        for k, col in enumerate(pts):
            fc = DEEP if k in (0, len(pts) - 1) else BOX
            for x, y in col:
                ax.add_patch(plt.Circle((x, y), 1.5, fc=fc, ec="white",
                                        lw=0.8, zorder=6))
        if args.labels:
            ax.text((cols[0][0] + cols[-1][0]) / 2, cy - 13.5,
                    "trunk features → 1024 → 300 genes", ha="center", va="top",
                    fontsize=5.4, color=INK)

    elif args.style == "cloud":
        # An unstructured node graph, matching the shape used in some pipeline
        # figures. Included because it was asked for, but it is the wrong
        # picture for this study: beside a spot map it reads as a graph BETWEEN
        # spots, and there is none here. Every patch passes through the network
        # on its own.
        import numpy as np
        rng = np.random.default_rng(4)
        n = 7
        ang = np.sort(rng.permutation(np.linspace(0, 2 * np.pi, n, endpoint=False)))
        rad = 8.5 + rng.uniform(-1.6, 1.6, n)
        xs = W / 2 + rad * np.cos(ang)
        ys = cy + rad * 0.72 * np.sin(ang)
        edges = [(0, 2), (0, 3), (1, 3), (1, 4), (2, 5), (3, 5), (3, 6),
                 (4, 6), (5, 6), (0, 1)]
        for i, j in edges:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=BOX, lw=0.8,
                    alpha=0.7, zorder=4, solid_capstyle="round")
        for k in range(n):
            ax.add_patch(plt.Circle((xs[k], ys[k]), 1.9,
                                    fc=DEEP if k % 3 == 0 else BOX,
                                    ec="white", lw=0.9, zorder=6))

    else:
        ax.add_patch(FancyBboxPatch((3, cy - 9), W - 6, 18,
                                    boxstyle="round,pad=0.2,rounding_size=1.2",
                                    fc=BOX, ec="none", alpha=0.95, zorder=6))
        if args.labels:
            ax.text(W / 2, cy, "Morphology-to-\nexpression model",
                    ha="center", va="center", zorder=7, fontsize=6.4,
                    color="white", fontweight="bold", linespacing=1.35)

    if args.caption:
        ax.text(W / 2, cy - 15.5,
                "Fixed predictor, identical across every arm\n"
                "trunk → linear 1024 → ReLU → dropout → linear 300",
                ha="center", va="top", fontsize=5.6, color=INK,
                linespacing=1.45)

    name = args.out or f"model_icon_{args.style}" + ("" if args.labels else "_bare")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.png / .svg / .pdf   style={args.style} "
          f"labels={args.labels} caption={args.caption}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
