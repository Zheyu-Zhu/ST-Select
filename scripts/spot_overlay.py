"""Transparent spot overlay that registers pixel-for-pixel onto a cleaned WSI.

Produces an RGBA PNG the same size as scripts/clean_wsi.py's output for the same
slide, with everything transparent except the spots. Dropped on top of the tissue
image in a slide deck it shows where the assay actually samples.

Registration is the whole problem here, and it is solved by recomputing rather
than guessing: the tissue mask, the crop box and the pyramid level are derived
with the same code path clean_wsi.py uses, so the crop origin and scale are known
exactly instead of being eyeballed. Spot coordinates come from the slide's own
h5ad in full-resolution pixels and are mapped through that same transform.

    python scripts/spot_overlay.py --slide TENX106 --level 1 --mask-level 3
    python scripts/spot_overlay.py --slide TENX106 --mode budget --budget 0.1

Modes:
    expression   every spot, coloured by a value
    budget       the selected fraction coloured, the rest drawn as empty rings
                 - the picture of "we only sequence some of them"

The colours are mock unless --real-expression is passed: a smooth spatial field
plus noise, which reads like expression because neighbouring spots agree, but is
not measured. Anything using it must say so.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import (binary_closing, binary_fill_holes, binary_opening,
                           label)

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent


def contour_mask(slide, shape, level):
    """HEST's tissue polygon, rasterised at `level`.

    Must be the same mask clean_wsi.py uses, or the two crops differ and the
    overlay lands a few pixels off - which is the one thing this file exists to
    prevent. Measured on SPA125 the colour threshold gave a 2174x2456 crop
    against the contour's 2184x2475.
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
    im = Image.new("L", (shape[1], shape[0]), 0)
    dr = _ID.Draw(im)
    for r in polys:
        dr.polygon([tuple(v) for v in (r / 2.0 ** level)], fill=255)
    return np.asarray(im) > 127


def tissue_mask(rgb, sat_thr=0.12, val_thr=40):
    a = rgb.astype(np.float32)
    mx, mn = a.max(2), a.min(2)
    m = ((mx - mn) / (mx + 1e-6) > sat_thr) & (mx > val_thr)
    m = binary_closing(binary_opening(m, np.ones((5, 5))), np.ones((25, 25)))
    lab, n = label(m)
    if n:
        m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return binary_fill_holes(m)


def mock_field(xy, seed=0, scale=0.28):
    """A smooth spatial field, so neighbouring spots agree the way real
    expression does. Sum of a few random low-frequency waves, squashed to
    [0,1]; not measured, and labelled as such wherever it is used."""
    rng = np.random.default_rng(seed)
    p = (xy - xy.min(0)) / max((xy.max(0) - xy.min(0)).max(), 1)
    v = np.zeros(len(xy))
    for _ in range(5):
        k = rng.normal(0, 3.0, 2)
        v += rng.normal() * np.sin(2 * np.pi * (p @ k) + rng.uniform(0, 6.28))
    v += rng.normal(0, scale * v.std(), len(xy))
    v = (v - v.min()) / max(float(np.ptp(v)), 1e-9)
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--mask-level", type=int, default=3)
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--mode", choices=["expression", "budget"], default="expression")
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--cols", type=int, default=7,
                    help="circles across the tissue at its widest. This is a "
                         "schematic, not the assay's real spot density - the "
                         "slide carries 2730 spots at a 365 px pitch, which at "
                         "slide-deck size renders as texture rather than as "
                         "something a reader can see. A coarse lattice on the "
                         "same footprint reads as 'the tissue is sampled on a "
                         "grid', which is the point being made.")
    ap.add_argument("--radius", type=float, default=0.28,
                    help="filled disc radius as a fraction of the lattice pitch")
    ap.add_argument("--ring", type=float, default=0.34,
                    help="white halo width as a fraction of the disc radius. The "
                         "halo is what keeps the dots readable over H&E: a bare "
                         "disc on pink tissue has no edge, and at slide-deck size "
                         "the field reads as a stain rather than as markers. 0 "
                         "turns it off.")
    ap.add_argument("--ring-color", default="255,255,255",
                    help="halo colour as R,G,B")
    ap.add_argument("--edge", default=None,
                    help="stroke colour on the disc itself, e.g. '#2A3441'. Use "
                         "instead of --ring for the dense look: at a tight pitch "
                         "a white halo eats the gaps and the field turns into a "
                         "pale sheet, while a dark stroke keeps every disc "
                         "separate and still lets the tissue show between them.")
    ap.add_argument("--edge-width", type=float, default=0.16,
                    help="stroke width as a fraction of the disc radius")
    ap.add_argument("--palette", default=None,
                    help="'st' gives the muted blue/grey/orange of a spot map, "
                         "where most discs sit at the blue end and the warm ones "
                         "read as the few regions that differ")
    ap.add_argument("--color", default=None,
                    help="one flat colour for every dot (e.g. '#D2601A'); "
                         "default is to colour by the mock field")
    ap.add_argument("--real-spots", action="store_true",
                    help="use the slide's own 2730 spot positions instead")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt

    src = ROOT / "hest_data" / "wsis" / f"{args.slide}.tif"
    h5 = ROOT / "hest_data" / "st" / f"{args.slide}.h5ad"
    for f in (src, h5):
        if not f.exists():
            print(f"{f} not found - download it from MahmoodLab/hest first")
            return 1

    # --- exactly the geometry clean_wsi.py produces -----------------------
    tf = tifffile.TiffFile(src)
    ml = min(args.mask_level, len(tf.pages) - 1)
    mask = contour_mask(args.slide, tf.pages[ml].shape[:2], ml)
    if mask is None or mask.sum() < 100:
        mask = tissue_mask(tf.pages[ml].asarray())
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - args.pad), ys.max() + args.pad
    x0, x1 = max(0, xs.min() - args.pad), xs.max() + args.pad
    s = 2 ** (ml - args.level)                       # mask px -> render px
    H0, W0 = tf.pages[args.level].shape[:2]
    top, left = int(y0 * s), int(x0 * s)
    h = min(int(y1 * s), H0) - top
    w = min(int(x1 * s), W0) - left

    # --- spot positions ---------------------------------------------------
    if args.real_spots:
        import anndata
        ad = anndata.read_h5ad(h5)
        xy = np.asarray(ad.obsm["spatial"], dtype=float)
        f = 2 ** args.level
        px, py = xy[:, 0] / f - left, xy[:, 1] / f - top
        keep = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        px, py = px[keep], py[keep]
        d = np.linalg.norm(np.c_[px, py][:400, None, :] - np.c_[px, py][None, :400, :], axis=2)
        np.fill_diagonal(d, 1e9)
        pitch = float(np.median(d.min(1)))
        print(f"{args.slide}: {keep.sum()}/{len(keep)} real spots inside the crop, "
              f"pitch {pitch:.0f} px")
    else:
        # A lattice sized to the tissue, kept only where there is tissue, so the
        # dot field takes the section's outline instead of a bounding box.
        sub = mask[y0:y1, x0:x1]
        occupied = np.asarray(
            Image.fromarray(sub.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR)) > 127
        pitch = w / args.cols
        gx = np.arange(pitch / 2, w, pitch)
        gy = np.arange(pitch / 2, h, pitch * np.sqrt(3) / 2)     # hex rows
        pts = []
        for j, yy in enumerate(gy):
            off = 0.0 if j % 2 == 0 else pitch / 2
            for xx in gx + off:
                if xx >= w:
                    continue
                iy, ix = int(yy), int(xx)
                if occupied[iy, ix]:
                    pts.append((xx, yy))
        px, py = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        print(f"{args.slide}: canvas {w}x{h}, schematic lattice "
              f"{args.cols} across -> pitch {pitch:.0f} px, {len(px)} circles on tissue")

    if len(px) == 0:
        print("  !! no spots placed"); return 1
    r = args.radius * pitch

    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(canvas)
    if args.palette == "st":
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "st", ["#6F86A8", "#8FA2BC", "#B9BEC4", "#E3B48C", "#DE9455"])
    else:
        cmap = plt.get_cmap(args.cmap)
    val = mock_field(np.c_[px, py], seed=args.seed)

    ring = tuple(int(v) for v in args.ring_color.split(",")) + (255,)
    rw = args.ring * r
    flat = None
    if args.color:
        c = args.color.lstrip("#")
        flat = tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) + (255,) if len(c) == 6             else tuple(int(v) for v in args.color.split(",")) + (255,)

    edge = None
    if args.edge:
        c = args.edge.lstrip("#")
        edge = tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) + (255,)
    ew = max(int(args.edge_width * r), 1)

    def draw(x, y, fill):
        # Halo first, disc on top: drawing the ring as an outline instead would
        # centre it on the disc edge and eat half the fill.
        if rw > 0 and not edge:
            dr.ellipse([x - r - rw, y - r - rw, x + r + rw, y + r + rw], fill=ring)
        box = [x - r, y - r, x + r, y + r]
        if edge:
            dr.ellipse(box, fill=fill, outline=edge, width=ew)
        else:
            dr.ellipse(box, fill=fill)

    if args.mode == "expression":
        for x, y, v in zip(px, py, val):
            draw(x, y, flat or tuple(int(255 * q) for q in cmap(float(v))[:3]) + (255,))
    else:
        rng = np.random.default_rng(args.seed)
        k = max(int(round(args.budget * len(px))), 1)
        sel = set(rng.choice(len(px), k, replace=False).tolist())
        for i, (x, y) in enumerate(zip(px, py)):
            if i in sel:
                draw(x, y, flat or tuple(int(255 * q) for q in cmap(float(val[i]))[:3]) + (255,))
            else:
                # Unselected: an outline and nothing else, so the tissue shows
                # through. The halo has to follow the same rule the filled discs
                # do - without the `not edge` guard it was still being drawn,
                # and on a transparent background every unselected spot came out
                # as a thick white donut instead of a ring.
                box = [x - r, y - r, x + r, y + r]
                if rw > 0 and not edge:
                    dr.ellipse([x - r - rw, y - r - rw, x + r + rw, y + r + rw], fill=ring)
                    dr.ellipse(box, fill=(0, 0, 0, 0))
                oc = tuple(list(edge[:3]) + [150]) if edge else (150, 150, 150, 200)
                dr.ellipse(box, outline=oc, width=ew if edge else max(int(r * 0.22), 1))
        print(f"  budget {args.budget:.0%}: {k} of {len(px)} dots filled")

    tag = args.mode if args.mode == "expression" else f"budget{int(args.budget*100)}"
    out = Path(args.out) if args.out else ROOT / "figures" / f"spots_{args.slide}_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    prev = out.with_name(out.stem + "_preview.png")
    im = canvas.copy(); im.thumbnail((1500, 1500), Image.LANCZOS); im.save(prev)
    print(f"  {out}  {w}x{h}  {out.stat().st_size / 2**20:.1f} MB")
    print(f"  {prev}  {im.size[0]}x{im.size[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
