"""A pile of H&E patches as a standalone transparent asset.

Real 224x224 patches from a slide, stacked with an offset so the pile reads as
several images rather than one. Composed with PIL rather than matplotlib so the
output is exactly the pixels drawn, at a predictable size, with a genuinely
transparent background instead of a figure canvas that happens to be white.

Patches are chosen by tissue content. An evenly spaced draw through the slide
lands on near-empty glass, which in a pile of four reads as a rendering fault
rather than as a sample of the input.

    python scripts/make_patch_pile.py
    python scripts/make_patch_pile.py --n 6 --dx 26 --dy -18 --border 3
    python scripts/make_patch_pile.py --layout grid --n 9
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def pick(slide, n, seed=0):
    """The n most tissue-rich patches, spread across the slide."""
    with h5py.File(ROOT / "hest_data" / "patches" / f"{slide}.h5", "r") as h:
        imgs = h["img"]
        cand = np.linspace(0, imgs.shape[0] - 1, min(imgs.shape[0], 60)).astype(int)
        score = []
        for i in cand:
            t = np.asarray(imgs[i]).astype(np.float32)
            sat = (t.max(2) - t.min(2)) / (t.max(2) + 1e-6)
            score.append((float(sat.mean()), int(i)))
        score.sort(reverse=True)
        idx = sorted(i for _, i in score[:n])
        return idx, [Image.fromarray(np.asarray(imgs[i])).convert("RGBA")
                     for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", default="SPA143")
    ap.add_argument("--n", type=int, default=5, help="patches in the pile")
    ap.add_argument("--size", type=int, default=224,
                    help="side of each patch in the output")
    ap.add_argument("--dx", type=int, default=30, help="x offset per patch")
    ap.add_argument("--dy", type=int, default=-22, help="y offset per patch")
    ap.add_argument("--border", type=int, default=4,
                    help="white edge around each patch, px; 0 for none")
    ap.add_argument("--shadow", type=int, default=6,
                    help="soft dark edge behind each patch, px; 0 for none")
    ap.add_argument("--layout", choices=["pile", "grid"], default="pile")
    ap.add_argument("--cols", type=int, default=3, help="grid layout only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    idx, tiles = pick(args.slide, args.n)
    s = args.size
    b = args.border
    tile = s + 2 * b

    if args.layout == "grid":
        rows = int(np.ceil(args.n / args.cols))
        gap = max(int(s * 0.05), 4)
        W = args.cols * tile + (args.cols - 1) * gap
        H = rows * tile + (rows - 1) * gap
        places = [((k % args.cols) * (tile + gap), (k // args.cols) * (tile + gap))
                  for k in range(args.n)]
    else:
        W = tile + abs(args.dx) * (args.n - 1)
        H = tile + abs(args.dy) * (args.n - 1)
        places = []
        for k in range(args.n):
            x = k * args.dx if args.dx >= 0 else (args.n - 1 - k) * (-args.dx)
            y = k * args.dy if args.dy >= 0 else (args.n - 1 - k) * (-args.dy)
            places.append((x, y))

    pad = args.shadow * 2
    canvas = Image.new("RGBA", (W + pad, H + pad), (0, 0, 0, 0))

    # back to front, so the last patch sits on top
    for (x, y), im in zip(places, tiles):
        im = im.resize((s, s), Image.LANCZOS)
        card = Image.new("RGBA", (tile, tile), (255, 255, 255, 255) if b
                         else (0, 0, 0, 0))
        card.paste(im, (b, b))
        px, py = x + args.shadow, y + args.shadow
        if args.shadow:
            # a soft edge behind the card, built by blurring its own silhouette
            from PIL import ImageFilter
            sh = Image.new("RGBA", (tile, tile), (0, 0, 0, 90))
            sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0)) \
                .copy()
            block = Image.new("RGBA", (tile, tile), (30, 30, 34, 110))
            sh.paste(block, (px + 2, py + 3))
            sh = sh.filter(ImageFilter.GaussianBlur(args.shadow * 0.7))
            canvas.alpha_composite(sh)
        canvas.alpha_composite(card, (px, py))

    name = args.out or f"patch_pile_{args.slide}_{args.layout}{args.n}"
    canvas.save(OUT / f"{name}.png")
    prev = canvas.copy()
    prev.thumbnail((900, 900), Image.LANCZOS)
    prev.save(OUT / f"{name}_preview.png")

    a = np.asarray(canvas)
    print(f"{args.slide}: patches {idx}")
    print(f"  each {s}x{s} px" + (f" with a {b} px white edge" if b else "")
          + (f", {args.shadow} px soft shadow" if args.shadow else ""))
    print(f"  canvas {canvas.size[0]}x{canvas.size[1]}, "
          f"{float((a[:, :, 3] < 8).mean()):.0%} fully transparent")
    print(f"  -> figures/assets/{name}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
