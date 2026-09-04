"""Figure 1, panel 3: the morphology-to-expression model, drawn rather than borrowed.

Replaces a crop of another paper's architecture diagram. That crop was wrong on
three counts: it showed ASIGN, which this study does not use; reusing a
published figure needs permission whoever the co-authors are; and ASIGN's own
figures draw the HER2 cohort, so both papers would be showing the same slides in
the same style.

The composition follows the panel it replaces -- a magnification cascade along
the top, the network, and a strip of per-slide predictions in dashed boxes with
an ellipsis at each end -- because the rest of Figure 1 was laid out around that
shape. Everything in it is ours and everything outside the network block is
real. The cascade is four genuine crops of one location, each box drawn where
the next crop actually comes from, ending at the 224-pixel patch the model is
actually fed. The strip is real held-out predictions, one per slide of the
held-out patient.

Three things in the original are deliberately not reproduced. Its four stacked
layers are serial sections of a 3D volume, and this study is single-section.
Edge/weight estimation depicts building a graph between spots, and partial
imputation depicts filling unmeasured layers from a measured one; we do neither,
since each patch passes through the network independently and no expression
value from a held-out patient is ever shown to the model. Drawing any of them
would recreate the problem this panel is being redrawn to fix.

    python scripts/make_fig1_panel3.py
    python scripts/make_fig1_panel3.py --gene COL1A1 --budget 0.3
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr
from matplotlib.patches import (ConnectionPatch, FancyArrow, FancyBboxPatch,
                                Polygon, Rectangle)
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
ST = Path(r"C:\Users\zheyu\workspace\stnet_ref\ST-Net")
DI = Path(r"C:\Users\zheyu\workspace\stnet_ref\data_infor")
OUT = ROOT / "figures" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
CM = 1 / 2.54

INK = "#1C1B18"
BOX = "#6F86A8"
DASH = "#9AA3AE"
ACC = "#D2601A"
EPOCH = {0.05: 110, 0.1: 80, 0.3: 70, 0.5: 60}
PATCH = 224                       # full-res px, the model's actual input width

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.dpi": 400, "savefig.transparent": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


def crop_box(slide, level=1, mask_level=3, pad=8):
    """The crop clean_wsi.py produces, as (left, top, w, h) at `level`."""
    tf = tifffile.TiffFile(ROOT / "hest_data" / "wsis" / f"{slide}.tif")
    ml = min(mask_level, len(tf.pages) - 1)
    shape = tf.pages[ml].shape[:2]
    polys = []
    for f in json.load(open(ROOT / "hest_data" / "tissue_seg" /
                            f"{slide}_contours.geojson"))["features"]:
        gm = f["geometry"]
        rings = ([gm["coordinates"][0]] if gm["type"] == "Polygon"
                 else [q[0] for q in gm["coordinates"]])
        polys += [np.asarray(r, dtype=float) for r in rings]
    im = Image.new("L", (shape[1], shape[0]), 0)
    dr = ImageDraw.Draw(im)
    for r in polys:
        dr.polygon([tuple(v) for v in (r / 2.0 ** ml)], fill=255)
    mask = np.asarray(im) > 127
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    s = 2 ** (ml - level)
    H0, W0 = tf.pages[level].shape[:2]
    top, left = int(y0 * s), int(x0 * s)
    return left, top, min(int(x1 * s), W0) - left, min(int(y1 * s), H0) - top


def read_region(slide, cx, cy, half, target=520):
    """A square crop centred on (cx, cy), read from whichever pyramid level
    keeps it near `target` pixels. Reading page 0 for a 3200-px region would
    decode 30 MB, and this machine has under 2 GB free while a sweep runs."""
    tf = tifffile.TiffFile(ROOT / "hest_data" / "wsis" / f"{slide}.tif")
    lv = 0
    while lv + 1 < len(tf.pages) and (2 * half) / 2 ** (lv + 1) >= target:
        lv += 1
    store = tifffile.imread(ROOT / "hest_data" / "wsis" / f"{slide}.tif",
                            key=lv, aszarr=True)
    z = zarr.open(store, mode="r")
    f = 2 ** lv
    x0 = int(max((cx - half) / f, 0))
    y0 = int(max((cy - half) / f, 0))
    n = int(2 * half / f)
    sub = np.asarray(z[y0:y0 + n, x0:x0 + n])[..., :3]
    store.close()
    if sub.shape[0] > target:
        sub = np.asarray(Image.fromarray(sub).resize((target, target),
                                                     Image.LANCZOS))
    return sub


def spot_map(ax, slide, gi, budget, x0, y0, w, h, vmin, vmax, size=1.6):
    rec = np.load(glob.glob(str(DI / "HER2" / "*" / f"{slide}.npy"))[0],
                  allow_pickle=True)
    pos = np.array([r["position"] for r in rec], dtype=float)
    pred = np.load(ST / "weights_st" / "HER2" / f"random_{budget}" / "fold1" /
                   "pred" / f"epoch_{EPOCH[budget]}" / f"{slide}_pred.npy")
    px, py = pos[:, 0], -pos[:, 1]
    s = min(w / max(np.ptp(px), 1e-9), h / max(np.ptp(py), 1e-9))
    px = x0 + (px - px.min()) * s + (w - np.ptp(px) * s) / 2
    py = y0 + (py - py.min()) * s + (h - np.ptp(py) * s) / 2
    ax.scatter(px, py, c=pred[:, gi], cmap="viridis", vmin=vmin, vmax=vmax,
               s=size, linewidths=0, zorder=8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", default="SPA143")
    ap.add_argument("--gene", default="ERBB2")
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--strip", default="SPA143,SPA144,SPA145,SPA146")
    ap.add_argument("--parts", action="store_true",
                    help="also write each cascade step and a set of flat-colour "
                         "spot maps as separate transparent PNGs, for assembling "
                         "the panel by hand")
    args = ap.parse_args()

    names = [str(x) for x in np.load(
        ROOT / "processed_gene" / "gene_names" / "HER2_expression_norm_genes.npy",
        allow_pickle=True)]
    gi = names.index(args.gene)
    strip = [s for s in args.strip.split(",") if s]
    um = 0.6904                                     # micrometres per pixel

    with h5py.File(ROOT / "hest_data" / "st" / f"{args.slide}.h5ad", "r") as f:
        xy = np.asarray(f["obsm/spatial"][:], dtype=float)
    d, _ = cKDTree(xy).query(xy, k=13)
    anchor = int(np.argmin(d[:, 1:].mean(1)))       # the densest neighbourhood
    cx, cy = xy[anchor]

    # level 0 is the cleaned slide render; the rest are crops of one location
    wsi = Image.open(ROOT / "figures" / f"wsi_{args.slide}_clean.png").convert("RGB")
    left, top, cw, ch = crop_box(args.slide)
    THUMB = 620
    wsi = wsi.resize((THUMB, round(THUMB * wsi.size[1] / wsi.size[0])),
                     Image.LANCZOS)
    sc = THUMB / cw
    wpx, wpy = (xy[:, 0] / 2 - left) * sc, (xy[:, 1] / 2 - top) * sc

    halves = [1600, 620, PATCH // 2]
    imgs = [np.asarray(wsi)] + [read_region(args.slide, cx, cy, h) for h in halves]
    spans = [None] + [2 * h for h in halves]
    labels = [f"{args.slide}\nwhole section",
              f"{2*halves[0]*um/1000:.1f} mm",
              f"{2*halves[1]*um:.0f} µm",
              f"{PATCH} px = {PATCH*um:.0f} µm\nmodel input"]

    W, H = 202.0, 80.0
    fig = plt.figure(figsize=(W / 10 * CM * 1.3, H / 10 * CM * 1.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((1.2, 1.2), W - 2.4, H - 2.4,
                                boxstyle="round,pad=0,rounding_size=1.6",
                                fc="none", ec=DASH, lw=0.8, zorder=1))

    # ---- the magnification cascade ---------------------------------------
    ty, side = 47.0, 21.0
    rects = []
    x = 5.0
    for k, im in enumerate(imgs):
        h_units = side * im.shape[0] / im.shape[1]
        y = ty - h_units / 2
        ax.imshow(im, extent=[x, x + side, y, y + h_units], zorder=9,
                  interpolation="bilinear")
        ax.add_patch(Rectangle((x, y), side, h_units, fill=False, lw=0.7,
                               ec=ACC if k == len(imgs) - 1 else "#CFD4DA",
                               zorder=11))
        rects.append((x, y, side, h_units))
        ax.text(x + side / 2, y - 1.6, labels[k], ha="center", va="top",
                fontsize=5.2, color=INK, linespacing=1.4)
        x += side + 7.0

    # each box marks where the next crop is taken from
    for k in range(len(imgs) - 1):
        rx, ry, rw, rh = rects[k]
        if k == 0:
            # spans are full-resolution pixels but `sc` maps level-1 pixels to
            # the thumbnail, so the half-width needs the extra factor of two.
            # Without it the box came out twice its true size and ran off the
            # section.
            bw = (spans[1] / 2 / 2) * sc / wsi.size[0] * rw
            bcx = rx + (cx / 2 - left) * sc / wsi.size[0] * rw
            bcy = ry + (1 - (cy / 2 - top) * sc / wsi.size[1]) * rh
        else:
            bw = spans[k + 1] / spans[k] * rw / 2
            bcx, bcy = rx + rw / 2, ry + rh / 2
        bw = max(bw, 0.9)
        ax.add_patch(Rectangle((bcx - bw, bcy - bw), 2 * bw, 2 * bw,
                               fill=False, ec=ACC, lw=0.9, zorder=12))
        nx, ny, nw, nh = rects[k + 1]
        for ay, by in (((bcy + bw), ny + nh), ((bcy - bw), ny)):
            ax.add_artist(ConnectionPatch((bcx + bw, ay), (nx, by), "data",
                                          "data", axesA=ax, axesB=ax, color=ACC,
                                          lw=0.6, ls=(0, (2.2, 1.8)), zorder=4))

    # ---- the fixed predictor ---------------------------------------------
    mx = rects[-1][0] + side + 4.5
    ax.add_patch(FancyArrow(mx, ty, 4.4, 0, width=0.5, head_width=1.9,
                            head_length=1.7, length_includes_head=True,
                            color=INK, zorder=5))
    ax.add_patch(Polygon([[mx + 6.5, ty - 9], [mx + 6.5, ty + 9],
                          [mx + 19, ty + 5], [mx + 19, ty - 5]], closed=True,
                         fc=BOX, ec="none", alpha=0.92, zorder=6))
    ax.text(mx + 12.7, ty, "DenseNet-121", ha="center", va="center", zorder=7,
            fontsize=5.2, color="white", fontweight="bold")
    for k, (bw, lab) in enumerate(((3.2, "1024"), (3.2, ""), (3.2, "300"))):
        bx = mx + 22 + k * 4.2
        bh = (8.5, 8.5, 13.0)[k]
        ax.add_patch(FancyBboxPatch((bx, ty - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.1,rounding_size=0.4",
                                    fc="white", ec=BOX, lw=0.9, zorder=6))
        if lab:
            ax.text(bx + bw / 2, ty, lab, ha="center", va="center", zorder=7,
                    fontsize=4.6, color=BOX, rotation=90)
    ax.text(mx + 19, ty - 12.5, "Fixed predictor, identical across every arm\n"
                                "trunk → linear 1024 → ReLU → dropout → linear 300",
            ha="center", va="top", fontsize=5.4, color=INK, linespacing=1.4)

    ox = mx + 36.5
    ax.add_patch(FancyArrow(ox, ty, 4.4, 0, width=0.5, head_width=1.9,
                            head_length=1.7, length_includes_head=True,
                            color=INK, zorder=5))
    gt = np.load(ST / "weights_st" / "HER2" / f"random_{args.budget}" / "fold1" /
                 "gt" / f"{args.slide}_gt.npy")
    vmin, vmax = gt[:, gi].min(), gt[:, gi].max()
    spot_map(ax, args.slide, gi, args.budget, ox + 6.5, ty - 10.5,
             W - (ox + 6.5) - 5.0, 21, vmin, vmax, 2.2)
    ax.text((ox + 6.5 + W - 5.0) / 2, ty - 12.5,
            f"300 genes per spot ({args.gene} shown)",
            ha="center", va="top", fontsize=5.4, color=INK)

    # ---- per-slide strip --------------------------------------------------
    if strip:
        sy, sh = 6.0, 18.0
        ax.text(5.0, sy + sh + 3.0, "Slide-level prediction, held-out patient",
                ha="left", va="bottom", fontsize=5.6, color=INK,
                fontweight="bold")
        bw = (W - 30.0) / len(strip)
        x = 12.0
        ax.text(7.5, sy + sh / 2, "…", ha="center", va="center", fontsize=9,
                color=DASH)
        for s in strip:
            ax.add_patch(FancyBboxPatch((x + 1.0, sy), bw - 2.0, sh,
                                        boxstyle="round,pad=0,rounding_size=0.8",
                                        fc="none", ec=DASH, lw=0.7,
                                        ls=(0, (2.5, 2)), zorder=3))
            spot_map(ax, s, gi, args.budget, x + 2.4, sy + 1.4, bw - 4.8,
                     sh - 2.8, vmin, vmax)
            x += bw
        ax.text(W - 7.5, sy + sh / 2, "…", ha="center", va="center", fontsize=9,
                color=DASH)

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"fig1_panel3_model.{ext}")
    plt.close(fig)

    if args.parts:
        pdir = OUT / "panel3_parts"
        pdir.mkdir(exist_ok=True)
        um = 0.6904

        # The four magnification steps as plain images: written straight through
        # PIL rather than a matplotlib figure, so the file is exactly the pixels
        # of the crop, at its native size, with nothing drawn around it.
        #
        # Alpha comes from the tissue contour rasterised at each crop's own
        # scale, so anything off the section is transparent. On the tighter
        # crops the whole frame is inside the tissue and the mask is solid,
        # which is the right answer rather than a missing feature.
        names = ["1_wholeslide", "2_region", "3_neighbourhood", "4_patch"]
        spans_um = ["whole section", f"{2*halves[0]*um/1000:.1f} mm",
                    f"{2*halves[1]*um:.0f} um", f"{PATCH} px = {PATCH*um:.0f} um"]

        # The whole-slide step keeps its own background rather than being cut
        # out. At this scale the section's outline is the point of the picture,
        # and a knocked-out silhouette reads as a shape rather than as a slide.
        whole = Image.open(ROOT / "figures" /
                           f"wsi_{args.slide}_clean.png").convert("RGB")
        whole = whole.resize(wsi.size, Image.LANCZOS)

        def contour_alpha(cx0, cy0, side_px, out_px):
            """Tissue mask for a full-resolution square, at the output size."""
            im = Image.new("L", (out_px, out_px), 0)
            dr = ImageDraw.Draw(im)
            s = out_px / side_px
            for f_ in json.load(open(ROOT / "hest_data" / "tissue_seg" /
                                     f"{args.slide}_contours.geojson"))["features"]:
                gm = f_["geometry"]
                rings = ([gm["coordinates"][0]] if gm["type"] == "Polygon"
                         else [q[0] for q in gm["coordinates"]])
                for r in rings:
                    v = (np.asarray(r, dtype=float) - [cx0, cy0]) * s
                    dr.polygon([tuple(t) for t in v], fill=255)
            return im

        for i, (nm, im, lab) in enumerate(zip(names, imgs, spans_um)):
            if i == 0:
                out = whole                       # opaque, background kept
            else:
                half = halves[i - 1]
                a = contour_alpha(cx - half, cy - half, 2 * half, im.shape[1])
                out = Image.fromarray(im).convert("RGBA")
                out.putalpha(a)
            out.save(pdir / f"cascade_{nm}.png")
            arr = np.asarray(out)
            clear = (f"transparent {float((arr[:, :, 3] < 128).mean()):.0%}"
                     if arr.shape[2] == 4 else "opaque, background kept")
            print(f"  cascade_{nm}.png  {out.size[0]}x{out.size[1]}  "
                  f"{out.mode}  ({lab})  {clear}")

        # the spot map as a flat silhouette, one colour per file. Every dot in a
        # given file is the same colour: these are for schematic use, where the
        # colour marks which panel a map belongs to rather than an expression
        # value, so nothing in them should read as data.
        rec = np.load(glob.glob(str(DI / "HER2" / "*" / f"{args.slide}.npy"))[0],
                      allow_pickle=True)
        pos = np.array([r["position"] for r in rec], dtype=float)
        px_, py_ = pos[:, 0], -pos[:, 1]
        pad = 0.06 * max(np.ptp(px_), np.ptp(py_))
        nx = max(len(np.unique(px_)), 1)
        W_CM = 5.0
        ms = ((W_CM / 2.54 * 72) / nx * 0.82) ** 2
        hcm = W_CM * (np.ptp(py_) + 2 * pad) / (np.ptp(px_) + 2 * pad)
        # taken from the strategy palette already used elsewhere in the figures
        for tag, col in (("blue", "#6F86A8"), ("orange", "#D2601A"),
                         ("green", "#3F7A52"), ("pink", "#C0407A")):
            f = plt.figure(figsize=(W_CM * CM, hcm * CM))
            a = f.add_axes([0, 0, 1, 1])
            a.scatter(px_, py_, c=col, s=ms, marker="o", linewidths=0)
            a.set_xlim(px_.min() - pad, px_.max() + pad)
            a.set_ylim(py_.min() - pad, py_.max() + pad)
            a.set_aspect("equal", adjustable="datalim")
            a.axis("off")
            for ext in ("png", "svg"):
                f.savefig(pdir / f"spots_flat_{tag}.{ext}")
            plt.close(f)
            print(f"  spots_flat_{tag}.png  {col}  {len(px_)} dots, all one colour")
        print(f"  -> figures/assets/panel3_parts/")
    print(f"cascade anchored on spot {anchor} at ({cx:.0f}, {cy:.0f})")
    for lab, sp in zip(labels[1:], spans[1:]):
        print(f"  {sp:>5d} px = {sp*um:>7.0f} um   {lab.splitlines()[0]}")
    print(f"strip: {strip}, held out in fold 1, random {args.budget:.0%}, {args.gene}")
    print("-> figures/assets/fig1_panel3_model.png / .svg / .pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
