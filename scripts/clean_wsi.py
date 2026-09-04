"""Turn a HEST whole-slide image into a clean PNG of just the tissue.

The raw WSI carries more than the section: a grey slide background and, on
Visium, the fiducial frame - a border of printed alignment dots that reads as
clutter in any figure. This segments the section, crops to it, and clears
everything outside to white.

Two details that a first attempt got wrong and are worth keeping:

  Mask resolution. Built at pyramid level 2 rather than a coarser one. At level
  4 a single mask pixel covers 16x16 full-resolution pixels, so tubule lumens
  and vessel interiors fall below the saturation threshold, get labelled
  background, and come back as blocky white squares punched through the tissue.

  Hole filling. Everything enclosed by the section outline is tissue, including
  lumens, vessels and tears. Without binary_fill_holes those interior
  low-saturation regions are cleared even when the mask is fine enough to
  resolve them.

The edge is feathered and alpha-composited rather than hard-thresholded, so the
section boundary does not come out as a staircase.

    python scripts/clean_wsi.py --slide SPA135
    python scripts/clean_wsi.py --slide NCBI712 --level 1     # smaller output
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageFilter
from scipy.ndimage import (binary_closing, binary_fill_holes, binary_opening,
                           label)

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent


def contour_mask(slide, shape, level):
    """HEST's own tissue polygon, rasterised at `level`.

    Preferred over thresholding because several slides carry a pale stain wash
    beside the section - a wide, faint pink field that is saturated enough to
    pass a colour threshold and comes back as a blocky slab attached to the
    tissue. HEST's contours exclude it. On SPA135 the polygon is y 2977-8294
    while the threshold started at y 0, which turned a square section into a
    2:3 one and produced a shape that was measured, reported, and wrong.
    """
    import json
    from PIL import ImageDraw as _ID
    g = ROOT / "hest_data" / "tissue_seg" / f"{slide}_contours.geojson"
    if not g.exists():
        return None
    polys = []
    for f in json.load(open(g))["features"]:
        gm = f["geometry"]
        rings = [gm["coordinates"][0]] if gm["type"] == "Polygon"             else [q[0] for q in gm["coordinates"]]
        polys += [np.asarray(r, dtype=float) for r in rings]
    if not polys:
        return None
    scale = 2.0 ** level
    im = Image.new("L", (shape[1], shape[0]), 0)
    dr = _ID.Draw(im)
    for r in polys:
        dr.polygon([tuple(v) for v in (r / scale)], fill=255)
    return np.asarray(im) > 127


def tissue_mask(rgb, sat_thr=0.12, val_thr=40):
    """H&E is saturated; slide glass and fiducial dots are not."""
    a = rgb.astype(np.float32)
    mx, mn = a.max(2), a.min(2)
    m = ((mx - mn) / (mx + 1e-6) > sat_thr) & (mx > val_thr)
    m = binary_closing(binary_opening(m, np.ones((5, 5))), np.ones((25, 25)))
    lab, n = label(m)
    if n:
        m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return binary_fill_holes(m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--level", type=int, default=0,
                    help="pyramid level to render; 0 is full resolution")
    ap.add_argument("--mask-level", type=int, default=2)
    ap.add_argument("--pad", type=int, default=8, help="margin in mask pixels")
    ap.add_argument("--transparent", action="store_true",
                    help="write RGBA with the off-tissue area alpha=0 instead of "
                         "white. The same feathered mask becomes the alpha "
                         "channel, so the section edge stays soft rather than "
                         "stepping. Roughly a third larger on disk - PNG cannot "
                         "predict a channel that is constant over most of the "
                         "frame as cheaply as it can a white one.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = ROOT / "hest_data" / "wsis" / f"{args.slide}.tif"
    if not src.exists():
        print(f"{src} not found - download it from MahmoodLab/hest first")
        return 1

    tf = tifffile.TiffFile(src)
    ml = min(args.mask_level, len(tf.pages) - 1)
    mask = contour_mask(args.slide, tf.pages[ml].shape[:2], ml)
    src_of_mask = "HEST contour"
    if mask is None or mask.sum() < 100:
        mask = tissue_mask(tf.pages[ml].asarray())
        src_of_mask = "colour threshold"
    print(f"  mask from {src_of_mask}")
    ys, xs = np.where(mask)
    if not len(ys):
        print("no tissue found")
        return 1

    s = 2 ** (ml - args.level)
    y0, y1 = max(0, ys.min() - args.pad), ys.max() + args.pad
    x0, x1 = max(0, xs.min() - args.pad), xs.max() + args.pad
    img = tf.pages[args.level].asarray()
    crop = img[int(y0 * s):min(int(y1 * s), img.shape[0]),
               int(x0 * s):min(int(x1 * s), img.shape[1])].copy()
    del img

    sub = mask[y0:y1, x0:x1]
    m = Image.fromarray(sub.astype(np.uint8) * 255).resize(
        (crop.shape[1], crop.shape[0]), Image.BILINEAR)
    m = m.filter(ImageFilter.GaussianBlur(2))
    alpha = np.asarray(m, dtype=np.uint8)
    a = alpha.astype(np.float32)[..., None] / 255.0
    if args.transparent:
        # Premultiplying into white first would leave a white halo wherever the
        # feathered edge is partly transparent; keep the original pixels and let
        # alpha do the work.
        crop = np.dstack([crop, alpha])
    else:
        crop = (crop * a + 255 * (1 - a)).astype(np.uint8)

    default = f"wsi_{args.slide}_clean{'_rgba' if args.transparent else ''}.png"
    out = Path(args.out) if args.out else ROOT / "figures" / default
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop).save(out)
    h, w = crop.shape[:2]
    prev = out.with_name(out.stem + "_preview.png")
    im = Image.fromarray(crop)
    im.thumbnail((1500, 1500), Image.LANCZOS)
    im.save(prev)

    print(f"{args.slide}: tissue {w}x{h} px, aspect w:h = {w/h:.4f} "
          f"({w/h*3:.2f}:3)")
    print(f"  {out}  {out.stat().st_size / 2**20:.0f} MB")
    print(f"  {prev}  {im.size[0]}x{im.size[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
