"""Colour-blind separation check for the categorical palette.

The bundled validator is a Node script and node is not installed here, so the
same quantities are computed directly: OKLab coordinates, Vienot dichromat
simulation for protanopia, deuteranopia and tritanopia, and pairwise distances
in OKLab (x100). Judging a palette by eye is exactly what this replaces.

Thresholds follow the guidance being applied: adjacent-pair CVD separation of 8
is the target and 6 is a floor that needs a second encoding channel; a
normal-vision separation below 15 is a hard failure because full-colour readers
cannot tell the pair apart either.

    python scripts/check_palette.py
    python scripts/check_palette.py "#1C1B18,#D2601A" --pairs all
"""

import argparse
import itertools
import sys

import numpy as np

PALETTE = {
    "Random": "#1C1B18",
    "CoreSet": "#D2601A",
    "MC-dropout": "#4B3F8F",
    "TOD": "#3F7A52",
    "TypiClust": "#C0407A",
}


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def to_oklab(rgb):
    """sRGB (0-1) to OKLab, Bjorn Ottosson's matrices."""
    l = srgb_to_linear(rgb)
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    lms = m1 @ l
    lms = np.cbrt(lms)
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return m2 @ lms


# Vienot 1999 dichromat simulation, in linear RGB
_CVD = {
    "protanopia": np.array([[0.11238, 0.88762, 0.0],
                            [0.11238, 0.88762, 0.0],
                            [0.00401, -0.00401, 1.0]]),
    "deuteranopia": np.array([[0.29275, 0.70725, 0.0],
                              [0.29275, 0.70725, 0.0],
                              [-0.02234, 0.02234, 1.0]]),
    "tritanopia": np.array([[1.0, 0.14461, -0.14461],
                            [0.0, 0.85837, 0.14163],
                            [0.0, 0.85837, 0.14163]]),
}


def simulate(rgb, kind):
    return linear_to_srgb(_CVD[kind] @ srgb_to_linear(rgb))


def dist(a, b):
    return float(np.linalg.norm(to_oklab(a) - to_oklab(b)) * 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("colours", nargs="?", default=None)
    ap.add_argument("--pairs", choices=["adjacent", "all"], default="all")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.colours:
        names = {f"c{i+1}": h.strip() for i, h in
                 enumerate(args.colours.split(","))}
    else:
        names = PALETTE
    items = list(names.items())

    print("=== lightness and chroma (OKLab) ===")
    for n, h in items:
        L, a, b = to_oklab(hex_to_rgb(h))
        C = float(np.hypot(a, b))
        flag = ""
        if not 0.25 <= L <= 0.85:
            flag = "  <-- outside the usual 0.25-0.85 band"
        print(f"  {n:12s} {h}  L={L:.3f}  C={C:.3f}{flag}")

    pairs = (list(itertools.combinations(range(len(items)), 2))
             if args.pairs == "all"
             else [(i, i + 1) for i in range(len(items) - 1)])

    print(f"\n=== pairwise separation, OKLab x100 ({args.pairs} pairs) ===")
    print(f"  {'pair':28s} {'normal':>7s} {'prot':>7s} {'deut':>7s} "
          f"{'trit':>7s}  verdict")
    worst = []
    for i, j in pairs:
        (na, ha), (nb, hb) = items[i], items[j]
        ra, rb = hex_to_rgb(ha), hex_to_rgb(hb)
        d0 = dist(ra, rb)
        ds = {k: dist(simulate(ra, k), simulate(rb, k)) for k in _CVD}
        m = min(ds.values())
        if d0 < 15:
            v = "FAIL normal <15"
        elif m >= 8:
            v = "pass"
        elif m >= 6:
            v = "floor - needs 2nd channel"
        else:
            v = "FAIL cvd <6"
        worst.append((m, f"{na} vs {nb}", v))
        print(f"  {na + ' vs ' + nb:28s} {d0:7.1f} {ds['protanopia']:7.1f} "
              f"{ds['deuteranopia']:7.1f} {ds['tritanopia']:7.1f}  {v}")

    worst.sort()
    bad = [w for w in worst if "FAIL" in w[2] or "floor" in w[2]]
    print(f"\n  tightest pair: {worst[0][1]} at {worst[0][0]:.1f} under CVD")
    print(f"  {len(bad)} of {len(pairs)} pairs need a second encoding channel"
          if bad else f"  all {len(pairs)} pairs clear the target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
