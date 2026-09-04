"""Figure 1, prediction panel: many tiles in, a whole-slide gene map out.

A simpler alternative to the magnification-cascade version. The cascade answers
"where does a patch come from"; this one answers "what does the model do", which
is what the panel is for: a batch of H&E patches enters, a fixed network turns
each into a 300-gene vector, and the vectors reassemble into a prediction across
the whole section.

Both ends are real. The input tiles are actual patches from SPA143, chosen for
tissue content rather than at even spacing because an evenly spaced draw lands
on near-empty slide. The output is that slide's own held-out prediction painted
back onto its H&E at true spot size, not a schematic heatmap.

Only the middle is drawn: a trapezoid for the trunk and blocks for the head. The
paper's position is that the predictor is ordinary and held constant across
every arm, so the panel should not make it look like a contribution.

    python scripts/make_fig1_predict.py
    python scripts/make_fig1_predict.py --gene COL1A1 --tiles 20
"""

import argparse
import glob
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
CM = 1 / 2.54

INK = "#1C1B18"
BOX = "#6F86A8"
DASH = "#9AA3AE"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.dpi": 400, "savefig.transparent": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", default="SPA143")
    ap.add_argument("--gene", default="ERBB2")
    ap.add_argument("--tiles", type=int, default=16)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--frame", action="store_true", help="draw the outer border")
    args = ap.parse_args()

    # ---- input: a grid of real patches -----------------------------------
    with h5py.File(ROOT / "hest_data" / "patches" / f"{args.slide}.h5", "r") as h:
        imgs = h["img"]
        cand = np.linspace(0, imgs.shape[0] - 1, min(imgs.shape[0], 80)).astype(int)
        ink = []
        for i in cand:
            t = np.asarray(imgs[i]).astype(np.float32)
            sat = (t.max(2) - t.min(2)) / (t.max(2) + 1e-6)
            ink.append((float(sat.mean()), int(i)))
        ink.sort(reverse=True)
        idx = sorted(i for _, i in ink[:args.tiles])
        tiles = [np.asarray(imgs[i]) for i in idx]

    # ---- output: the prediction painted back onto the slide --------------
    base = Image.open(ROOT / "figures" / f"wsi_{args.slide}_clean.png").convert("RGBA")
    over = ROOT / "figures" / f"expr_{args.slide}_{args.gene}_pred.png"
    if not over.exists():
        raise SystemExit(f"missing {over.name}; run scripts/expr_overlay.py "
                         f"--slide {args.slide} --gene {args.gene} first")
    with Image.open(over) as o:
        o = o.convert("RGBA")
        if o.size != base.size:
            raise SystemExit(f"overlay {o.size} does not match slide {base.size}")
        comp = Image.alpha_composite(base, o)
    W_OUT = 900                       # matplotlib promotes to float64 RGBA
    comp = comp.resize((W_OUT, round(W_OUT * comp.size[1] / comp.size[0])),
                       Image.LANCZOS)

    # ---- layout ----------------------------------------------------------
    W, H = 172.0, 62.0
    fig = plt.figure(figsize=(W / 10 * CM * 1.3, H / 10 * CM * 1.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    if args.frame:
        ax.add_patch(FancyBboxPatch((1.0, 1.0), W - 2.0, H - 2.0,
                                    boxstyle="round,pad=0,rounding_size=1.4",
                                    fc="none", ec=DASH, lw=0.8, zorder=1))

    cy = 36.0
    rows = int(np.ceil(len(tiles) / args.cols))
    t = 7.4                                        # tile side
    gap = 0.7
    gw = args.cols * t + (args.cols - 1) * gap
    gh = rows * t + (rows - 1) * gap
    gx, gy = 5.0, cy - gh / 2
    for k, im in enumerate(tiles):
        x = gx + (k % args.cols) * (t + gap)
        y = gy + gh - t - (k // args.cols) * (t + gap)
        ax.imshow(im, extent=[x, x + t, y, y + t], zorder=10,
                  interpolation="bilinear")
        ax.add_patch(Rectangle((x, y), t, t, fill=False, lw=0.5, ec="white",
                               zorder=11))
    ax.text(gx + gw / 2, gy - 2.0,
            f"H&E patches, one per spot\n224 × 224 px  ·  {args.tiles} of "
            f"{imgs.shape[0]} shown",
            ha="center", va="top", fontsize=5.6, color=INK, linespacing=1.4)

    x0 = gx + gw + 4.0
    ax.add_patch(FancyArrow(x0, cy, 5.0, 0, width=0.55, head_width=2.1,
                            head_length=1.9, length_includes_head=True,
                            color=INK, zorder=5))

    mx = x0 + 6.8
    ax.add_patch(Polygon([[mx, cy - 10], [mx, cy + 10],
                          [mx + 15, cy + 6], [mx + 15, cy - 6]], closed=True,
                         fc=BOX, ec="none", alpha=0.92, zorder=6))
    ax.text(mx + 7.5, cy, "DenseNet-121", ha="center", va="center", zorder=7,
            fontsize=5.6, color="white", fontweight="bold")
    for k, (bw, lab) in enumerate(((3.6, "1024"), (3.6, ""), (3.6, "300"))):
        bx = mx + 18 + k * 4.7
        bh = (9.5, 9.5, 14.5)[k]
        ax.add_patch(FancyBboxPatch((bx, cy - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.1,rounding_size=0.4",
                                    fc="white", ec=BOX, lw=1.0, zorder=6))
        if lab:
            ax.text(bx + bw / 2, cy, lab, ha="center", va="center", zorder=7,
                    fontsize=5.0, color=BOX, rotation=90)
    ax.text(mx + 16, cy - 13.0,
            "Fixed predictor, identical across every arm\n"
            "trunk → linear 1024 → ReLU → dropout → linear 300",
            ha="center", va="top", fontsize=5.6, color=INK, linespacing=1.4)

    x1 = mx + 32.5
    ax.add_patch(FancyArrow(x1, cy, 5.0, 0, width=0.55, head_width=2.1,
                            head_length=1.9, length_includes_head=True,
                            color=INK, zorder=5))

    ox = x1 + 6.8
    ow = W - ox - 4.0
    oh = ow * comp.size[1] / comp.size[0]
    ax.imshow(np.asarray(comp), extent=[ox, ox + ow, cy - oh / 2, cy + oh / 2],
              zorder=9, interpolation="bilinear")
    ax.text(ox + ow / 2, cy - oh / 2 - 2.0,
            f"Predicted expression across the section\n"
            f"300 genes per spot ({args.gene} shown), held-out patient",
            ha="center", va="top", fontsize=5.6, color=INK, linespacing=1.4)

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"fig1_predict.{ext}")
    plt.close(fig)
    print(f"tiles: {len(tiles)} of {imgs.shape[0]} from {args.slide}, "
          f"rows {idx[:6]}...")
    print(f"output: {args.slide} {args.gene} prediction over its own H&E")
    print("-> figures/assets/fig1_predict.png / .svg / .pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
