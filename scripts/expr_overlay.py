"""Predicted expression painted onto the slide, as a transparent layer.

Produces an RGBA PNG the same size as scripts/clean_wsi.py's output for the
same slide, so dropping it over wsi_<slide>_clean_rgba.png gives a whole-slide
gene prediction map. Two are written and they belong together:

    pred   what the model produces from H&E alone
    true   what was measured at the same spots

They share a colour scale, taken from the measured map, so the pair shows what
the reported correlation actually looks like. A model can score 0.3 and still
produce a visibly flatter map than the truth, and the two side by side make that
concrete in a way the scalar does not.

No inference happens here. The predictions are read from the sweep's stored
output, so the slide must be one that was held out in the fold whose checkpoint
produced them; the script refuses otherwise. Registration reuses clean_wsi.py's
contour and crop rather than re-deriving them, and spots are drawn at their real
diameter, not an arbitrary marker size.

    python scripts/expr_overlay.py --slide SPA143 --gene ERBB2
    python scripts/expr_overlay.py --slide SPA143 --gene COL1A1 --budget 0.05
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import tifffile
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
ST = Path(r"C:\Users\zheyu\workspace\stnet_ref\ST-Net")
DI = Path(r"C:\Users\zheyu\workspace\stnet_ref\data_infor")
EPOCH = {0.05: 110, 0.1: 80, 0.3: 70, 0.5: 60}


def crop_box(slide, level, mask_level, pad):
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


def fold_of(slide):
    for f in range(4):
        if list((ST / "weights_st" / "HER2").glob(f"*/fold{f}/gt/{slide}_gt.npy")):
            return f
    raise SystemExit(f"{slide}: no stored fold holds it out; the map would be "
                     f"a memory, not a prediction")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", default="SPA143")
    ap.add_argument("--gene", default="ERBB2")
    ap.add_argument("--strategy", default="random")
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--mask-level", type=int, default=3)
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--radius", type=float, default=0.0,
                    help="spot radius in output px; 0 uses the platform's real "
                         "spot diameter from the HEST metadata")
    ap.add_argument("--alpha", type=int, default=235)
    ap.add_argument("--ring", type=float, default=0.10,
                    help="white halo as a fraction of the radius, for contrast "
                         "against dark stroma; 0 to disable")
    args = ap.parse_args()

    fold = fold_of(args.slide)
    names = [str(x) for x in np.load(
        ROOT / "processed_gene" / "gene_names" / "HER2_expression_norm_genes.npy",
        allow_pickle=True)]
    base = ST / "weights_st" / "HER2" / f"{args.strategy}_{args.budget}" / f"fold{fold}"
    gt = np.load(base / "gt" / f"{args.slide}_gt.npy")
    if args.gene not in names[:gt.shape[1]]:
        why = (f"outside the {gt.shape[1]}-gene target panel"
               if args.gene in names else "absent from this cohort")
        raise SystemExit(f"{args.gene}: {why}, so the model never predicts it")
    gi = names.index(args.gene)
    pred = np.load(base / "pred" / f"epoch_{EPOCH[args.budget]}" /
                   f"{args.slide}_pred.npy")

    left, top, w, h = crop_box(args.slide, args.level, args.mask_level, args.pad)
    ref = ROOT / "figures" / f"wsi_{args.slide}_clean.png"
    if ref.exists():                       # match the saved render exactly
        with Image.open(ref) as im:
            W, H = im.size
    else:
        W, H = w, h
    sc = W / w

    with h5py.File(ROOT / "hest_data" / "st" / f"{args.slide}.h5ad", "r") as f:
        xy = np.asarray(f["obsm/spatial"][:], dtype=float)
    with h5py.File(ROOT / "hest_data" / "patches" / f"{args.slide}.h5", "r") as f:
        ps = float(f["img"].attrs["pixel_size"])
    px = (xy[:, 0] / 2 ** args.level - left) * sc
    py = (xy[:, 1] / 2 ** args.level - top) * sc
    if len(px) != len(gt):
        raise SystemExit(f"{len(px)} spots but {len(gt)} stored rows")

    # 100 um capture spot on this platform, expressed in output pixels
    r = args.radius or (100.0 / ps / 2 ** args.level * sc / 2)

    cmap = matplotlib.colormaps[args.cmap]
    lo, hi = float(gt[:, gi].min()), float(gt[:, gi].max())
    ring = max(int(args.ring * r), 0)

    for tag, mat in (("pred", pred), ("true", gt)):
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(canvas)
        v = mat[:, gi]
        for x, y, q in zip(px, py, v):
            t = float(np.clip((q - lo) / (hi - lo + 1e-9), 0, 1))
            c = tuple(int(255 * u) for u in cmap(t)[:3]) + (args.alpha,)
            if ring:
                dr.ellipse([x - r - ring, y - r - ring, x + r + ring, y + r + ring],
                           fill=(255, 255, 255, args.alpha))
            dr.ellipse([x - r, y - r, x + r, y + r], fill=c)
        out = ROOT / "figures" / f"expr_{args.slide}_{args.gene}_{tag}.png"
        canvas.save(out)
        canvas.resize((1500, int(1500 * H / W)), Image.LANCZOS).save(
            out.with_name(out.stem + "_preview.png"))
        print(f"  {out.name}  {W}x{H}  s.d. {v.std():.3f}")

    x_, y_ = pred[:, gi], gt[:, gi]
    rr = float(np.corrcoef(x_, y_)[0, 1]) if x_.std() > 0 else float("nan")
    print(f"{args.slide}: fold {fold} held out, {args.strategy} "
          f"{args.budget:.0%}, {args.gene}")
    print(f"  spot radius {r:.0f} px (100 µm at {ps:.3f} µm/px)")
    print(f"  colour scale {lo:.2f}–{hi:.2f} from the measured map, shared")
    print(f"  predicted vs measured r = {rr:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
