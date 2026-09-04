"""Figure 1 legend: what a filled dot and an empty ring mean.

The budget panel draws selected spots filled and unselected ones as empty rings,
but nothing on the figure says so. The marker styles here are copied from
scripts/spot_overlay.py's budget mode rather than approximated, so the legend
matches the picture it explains.

    python scripts/make_fig1_legend.py
    python scripts/make_fig1_legend.py --layout vertical
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
CM = 1 / 2.54
INK = "#1C1B18"

# spot_overlay.py's "st" ramp, sampled at both ends and the middle, plus the
# grey it outlines unselected spots with. Selection is fill-versus-ring; the
# hue along the ramp is the measured expression value.
FILL_LO = "#6F86A8"
FILL_MID = "#B9BEC4"
FILL_HI = "#DE9455"
RING_EC = "#B9BEC4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.dpi": 400, "savefig.transparent": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


FS = 6.8
MS = 6.4


def right_aligned() -> int:
    """Two rows, flush right: label right edges line up, markers hang left.

    The legend machinery cannot do this. Its rows are marker-then-label with the
    markers in a shared column, so the ragged edge lands on the right - which is
    the edge we want straight when the block sits against a panel's right
    margin. So the rows are placed by hand here, but the marker x still comes
    from measuring the drawn text rather than from a guessed offset: the two
    labels differ by nine characters and any fixed offset puts one marker
    through its own text.
    """
    fig_h = 1.3                                   # cm
    fig = plt.figure(figsize=(6.2 * CM, fig_h * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Row separation set in points against the type size, not as a fraction of
    # a figure height that is cropped away later. 1.35 line heights reads as one
    # block of two rows; the 2.4 it worked out to before read as two captions
    # that happened to be near each other.
    sep = (1.35 * FS) / (fig_h / 2.54 * 72)
    rows = [(0.5 + sep / 2, ["Selected (expression measured)"],
             [FILL_LO, FILL_MID, FILL_HI]),
            (0.5 - sep / 2, ["Not selected (masked)"], [None])]

    fig.canvas.draw()
    inv = ax.transData.inverted()
    drawn = []
    for y, (label,), fills in rows:
        t = ax.text(1.0, y, label, ha="right", va="center", fontsize=FS,
                    color=INK)
        drawn.append(t)
        fig.canvas.draw()
        # left edge of the text, in axes coordinates
        x0 = inv.transform(t.get_window_extent(fig.canvas.get_renderer()))[0][0]
        dx = 0.030                      # spacing between the three dots
        gap = 0.035                     # text-to-marker gap
        for k, c in enumerate(reversed(fills)):
            x = x0 - gap - k * dx
            if c is None:
                drawn += ax.plot([x], [y], marker="o", ms=MS, mfc="none",
                                 mec=RING_EC, mew=1.5, clip_on=False)
            else:
                drawn += ax.plot([x], [y], marker="o", ms=MS, mfc=c,
                                 mec="white", mew=0.8, clip_on=False)

    # Crop to what was actually drawn. A "tight" save keeps the whole axes, and
    # the axes is deliberately wider than the content so the right-aligned text
    # has room to grow leftwards - which would ship the empty half with it.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = Bbox.union([a.get_window_extent(r) for a in drawn])
    box = box.transformed(fig.dpi_scale_trans.inverted()).expanded(1.04, 1.30)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"fig1_legend_right.{ext}", bbox_inches=box)
    plt.close(fig)
    print("-> figures/assets/fig1_legend_right.png / .svg / .pdf")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["horizontal", "vertical", "right"],
                    default="horizontal")
    args = ap.parse_args()

    if args.layout == "right":
        return right_aligned()

    # Laid out by the legend machinery rather than by hand: hard-coded x
    # offsets overlapped as soon as a label ran longer than the gap allowed.
    #
    # The selected handle is three dots, not one. In the panel a selected spot
    # is filled with its expression value off the "st" ramp, so selected dots
    # run from blue through neutral to orange; only the fill-versus-ring
    # distinction carries selection. A single orange dot said "selected means
    # orange", which is wrong for every blue dot in the picture.
    vert = args.layout == "vertical"
    sel = tuple(
        plt.Line2D([], [], marker="o", linestyle="none", markersize=6.4,
                   markerfacecolor=c, markeredgecolor="white",
                   markeredgewidth=0.8)
        for c in (FILL_LO, FILL_MID, FILL_HI))
    unsel = plt.Line2D([], [], marker="o", linestyle="none", markersize=6.4,
                       markerfacecolor="none", markeredgecolor=RING_EC,
                       markeredgewidth=1.5)
    handles = [sel, unsel]
    labels = ["Selected (expression measured)", "Not selected (masked)"]

    fig = plt.figure(figsize=((9.6 if not vert else 5.4) * CM,
                              (0.8 if not vert else 1.4) * CM))
    fig.legend(handles=handles, labels=labels, loc="center", frameon=False,
               fontsize=6.8, ncol=2 if not vert else 1, handletextpad=0.5,
               columnspacing=1.6, labelspacing=0.9, labelcolor=INK,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)})

    name = f"fig1_legend_{args.layout}"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"-> figures/assets/{name}.png / .svg / .pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
