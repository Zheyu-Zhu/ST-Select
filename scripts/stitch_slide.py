"""Reassemble one slide's H&E patches into a single image.

The dataset ships 224x224 patches, one per spot, not the whole-slide images they
were cut from. This puts them back on the acquisition grid using each spot's
(row, col) position, which recovers a picture of the section at spot resolution -
not the original WSI, but the same tissue, and everything the models actually see.

Gaps are real and stay white. About 36% of the lattice cells hold no spot, and
the tissue under them was never captured - filling them in would be invention.
Nor can neighbouring patches be stretched to cover them: measured on NCBI712,
a patch's right edge matches its right-hand neighbour's left edge no better than
it matches a randomly chosen patch (NCC +0.013 vs +0.046), so the crops do not
overlap and there is nothing to bleed across.

    python scripts/stitch_slide.py --dataset Kidney --slide NCBI712
    python scripts/stitch_slide.py --dataset Kidney            # largest slide
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
GENE = ROOT / "processed_gene"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Kidney", choices=["Kidney", "HER2", "BC"])
    ap.add_argument("--slide", default=None, help="slide id; default = most spots")
    ap.add_argument("--tile", type=int, default=64,
                    help="pixels per spot in the output (224 = full resolution)")
    ap.add_argument("--overlap", type=float, default=1.0,
                    help="draw each patch this many lattice cells wide. Leave at "
                         "1.0. Values >1 were meant to close the lattice gaps by "
                         "letting neighbours bleed into them, on the assumption "
                         "that adjacent patches overlap on the slide. They do "
                         "not: matching each patch's right edge against its "
                         "neighbour's left edge scores NCC +0.013, against "
                         "+0.046 for randomly paired patches - no better than "
                         "chance. The patches are independent, non-overlapping "
                         "crops, so enlarging them duplicates tissue and the "
                         "seams stop lining up.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted((GENE / "dataset_log1p_hvg300" / args.dataset).glob("*.npy"))
    if not files:
        print(f"no slides under {GENE / 'dataset_log1p_hvg300' / args.dataset}")
        return 1

    if args.slide:
        hit = [f for f in files if f.stem == args.slide]
        if not hit:
            print(f"{args.slide} not found; have {[f.stem for f in files][:5]}...")
            return 1
        path = hit[0]
    else:
        path = max(files, key=lambda f: len(np.load(f, allow_pickle=True)))

    spots = np.load(path, allow_pickle=True)
    sid = spots[0]["sample_id"]
    pos = np.array([s["position"] for s in spots])
    r0, c0 = pos[:, 0].min(), pos[:, 1].min()
    rows = int(np.ptp(pos[:, 0])) + 1
    cols = int(np.ptp(pos[:, 1])) + 1

    t = args.tile
    d = max(int(round(t * args.overlap)), t)
    pad = (d - t) // 2 + 1
    canvas = np.full((rows * t + 2 * pad, cols * t + 2 * pad, 3), 255, dtype=np.uint8)
    pdir = GENE / "patches" / sid
    placed = 0
    # Farthest-from-centre first, so the centre of the section is drawn last and
    # its own patch wins wherever two overlap.
    ctr = pos.mean(0)
    order = np.argsort(-np.linalg.norm(pos - ctr, axis=1))
    for i in order:
        s, (r, c) = spots[i], pos[i]
        p = pdir / f"{s['patch_id']}.png"
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        if im.size != (d, d):
            im = im.resize((d, d), Image.LANCZOS)
        y = (r - r0) * t + pad - (d - t) // 2
        x = (c - c0) * t + pad - (d - t) // 2
        canvas[y:y + d, x:x + d] = np.asarray(im)
        placed += 1

    out = Path(args.out) if args.out else ROOT / "figures" / f"slide_{sid}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out, optimize=True)
    mb = out.stat().st_size / 2**20
    print(f"{sid}: {placed}/{len(spots)} patches placed on a {rows}x{cols} grid "
          f"({placed / (rows * cols):.0%} filled)")
    print(f"wrote {out}  {canvas.shape[1]}x{canvas.shape[0]} px  {mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
