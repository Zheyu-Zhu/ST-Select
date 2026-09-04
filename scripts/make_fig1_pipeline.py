"""Figure 1 — complete, paper-ready pipeline figure (graphics + all typography).

Image-generation models were tried three times and failed on exactly the thing
that matters here: they garbled text ("Proposed AL Franwork"), filed Coreset
under "Random Baseline", printed the word FLAT inside an axes when asked for a
flat curve, and drew budget curves that contradicted the measurements.

Everything is therefore drawn and typeset here. Two details are load-bearing:

  * The canvas works in CENTIMETRES with an equal aspect ratio. An earlier
    version used a 0-1 box across a 17.5 cm wide figure, which stretched every
    spot into an ellipse.
  * Panel (a)'s colours come from a smooth 2-D field quantised into discrete
    zones, so neighbouring spots visibly share a value. That IS the redundancy
    argument — a random palette would illustrate its opposite.

Panel (d) plots the measured budget curves straight from results_pg/, so the
three shapes (flat / rising / rising-late) are whatever the experiments produced.

    python scripts/make_fig1_pipeline.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

W, H = 17.8, 8.8                       # cm, two-column journal width
TEAL, ROSE, SLATE, AMBER = "#1F6F78", "#C0407A", "#4B5D75", "#C98A2E"
PURPLE, INK, MUTED, HAIR = "#4B3F8F", "#1C1B18", "#6E6A61", "#C9C4BA"
TISSUE_FILL, TISSUE_EDGE = "#F7EAF0", "#B98BA6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 300, "savefig.dpi": 500,
})

TITLE, LABEL, SMALL, TINY = 7.6, 6.4, 5.8, 5.2


def blob(cx, cy, rx, ry, seed=3, n=240):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = np.arange(1, 6)
    amp = rng.normal(0, 1, len(k)) * 0.045 / k
    pha = rng.uniform(0, 2 * np.pi, len(k))
    r = 1 + sum(a * np.sin(kk * t + p) for a, kk, p in zip(amp, k, pha))
    return np.c_[cx + rx * r * np.cos(t), cy + ry * r * np.sin(t)]


def hex_spots(poly, step):
    pth = MPath(poly)
    xs, ys = poly[:, 0], poly[:, 1]
    pts, row, y = [], 0, ys.min()
    while y <= ys.max():
        x = xs.min() + ((step / 2) if row % 2 else 0)
        while x <= xs.max():
            pts.append((x, y)); x += step
        y += step * 0.866; row += 1
    pts = np.array(pts)
    return pts[pth.contains_points(pts, radius=-step * 0.45)]


def field(pts, seed=1, n_centres=6, width=0.55, levels=0):
    rng = np.random.default_rng(seed)
    lo, hi = pts.min(0), pts.max(0)
    c = rng.uniform(lo, hi, (n_centres, 2))
    w = rng.normal(0, 1, n_centres)
    span = np.hypot(*(hi - lo))
    d2 = ((pts[:, None, :] - c[None]) ** 2).sum(-1)
    v = (np.exp(-d2 / (2 * (width * span / 2) ** 2)) * w).sum(1)
    v = (v - v.min()) / (np.ptp(v) + 1e-9)
    if levels:                       # quantise -> visible same-value zones
        v = np.floor(v * levels).clip(0, levels - 1) / (levels - 1)
    return v


def tissue(ax, poly, lw=1.0):
    ax.add_patch(Polygon(poly, closed=True, facecolor=TISSUE_FILL,
                         edgecolor=TISSUE_EDGE, lw=lw, zorder=1))


def main():
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    T = dict(ha="center", va="center")

    PW, GAP, PY, PH = 4.03, 0.40, 0.16, 8.05
    X = [0.14 + i * (PW + GAP) for i in range(4)]
    for x in X:
        ax.add_patch(FancyBboxPatch((x, PY), PW, PH,
                                    boxstyle="round,pad=0.05,rounding_size=0.16",
                                    facecolor="white", edgecolor=HAIR, lw=0.9, zorder=0))
    for i in range(3):
        ax.add_patch(FancyArrowPatch((X[i] + PW + 0.06, PY + PH * 0.52),
                                     (X[i + 1] - 0.06, PY + PH * 0.52),
                                     arrowstyle="-|>", mutation_scale=10,
                                     lw=1.5, color=SLATE, zorder=6))

    # One left-aligned string per panel. Centring the title while left-aligning
    # the letter made them collide once the title outgrew the panel width.
    TITLES = ["Dense ST acquisition", "Spot selection under budget",
              "Expression prediction", "Budget is cohort-specific"]
    for i, (x, t) in enumerate(zip(X, TITLES)):
        ax.text(x + 0.20, PY + PH - 0.30, f"({chr(97+i)})  {t}", fontsize=TITLE,
                fontweight="bold", ha="left", va="center", color=INK)
        ax.plot([x + 0.18, x + PW - 0.18], [PY + PH - 0.58] * 2,
                color=HAIR, lw=0.7, zorder=2)

    cmap = plt.get_cmap("Spectral_r")

    # ------------------------------------------------------------- (a)
    cx = X[0] + PW / 2
    p1 = blob(cx, 4.90, 1.42, 1.62, seed=3)
    tissue(ax, p1)
    s1 = hex_spots(p1, 0.205)
    v1 = field(s1, seed=6, n_centres=5, width=0.40, levels=6)
    ax.add_collection(PatchCollection([Circle(p, 0.088) for p in s1],
                                      facecolor=cmap(v1), edgecolor="white",
                                      linewidths=0.25, zorder=3))
    ax.text(cx, 6.86, "every spot sequenced", fontsize=SMALL, color=MUTED, **T)
    # callout on a redundant zone
    ax.add_patch(Circle((cx - 0.72, 4.30), 0.46, facecolor="none",
                        edgecolor=INK, lw=0.8, ls=(0, (2.2, 1.6)), zorder=5))
    ax.annotate("", xy=(cx - 0.30, 4.02), xytext=(cx + 0.62, 3.16),
                arrowprops=dict(arrowstyle="-", lw=0.6, color=INK), zorder=5)
    ax.text(cx + 0.66, 3.02, "neighbouring spots\nshare expression", fontsize=TINY,
            ha="left", va="top", color=INK, linespacing=1.25)
    for xo in (-0.44, 0, 0.44):
        for dy, fc in ((0.0, "#D9C68A"), (0.10, "#F0E4B8")):
            ax.add_patch(Circle((cx + xo, 1.62 + dy), 0.20, facecolor=fc,
                                edgecolor="#A98B3F", lw=0.8, zorder=3 + (dy > 0)))
    ax.text(cx, 1.06, "High cost", fontsize=LABEL, fontweight="bold", color=INK, **T)
    ax.text(cx, 0.62, "dense grids oversample\nrepetitive morphology", fontsize=TINY,
            color=MUTED, linespacing=1.25, **T)

    # ------------------------------------------------------------- (b)
    cx = X[1] + PW / 2 - 0.34
    p2 = blob(cx, 5.30, 1.20, 1.34, seed=3)
    tissue(ax, p2)
    s2 = hex_spots(p2, 0.205)
    ax.add_collection(PatchCollection([Circle(p, 0.082) for p in s2],
                                      facecolor="white", edgecolor="#CFC9BE",
                                      linewidths=0.45, zorder=2))
    rng = np.random.default_rng(11)
    fam = [TEAL, ROSE, SLATE, AMBER]
    fam_lab = ["Random", "Uncertainty", "Diversity", "Spatial"]
    pick = rng.choice(len(s2), size=int(0.15 * len(s2)), replace=False)
    ax.add_collection(PatchCollection([Circle(s2[i], 0.095) for i in pick],
                                      facecolor=[fam[j % 4] for j in range(len(pick))],
                                      edgecolor="white", linewidths=0.4, zorder=4))
    ax.text(cx, 6.86, "a subset is sequenced", fontsize=SMALL, color=MUTED, **T)
    for j, (c, lb) in enumerate(zip(fam, fam_lab)):
        yy = 6.10 - j * 0.40
        ax.add_patch(Rectangle((X[1] + PW - 1.44, yy), 0.24, 0.24,
                               facecolor=c, edgecolor="none", zorder=4))
        ax.text(X[1] + PW - 1.12, yy + 0.12, lb, fontsize=TINY, ha="left",
                va="center", color=INK)
    ax.text(X[1] + PW - 1.46, 6.52, "20 strategies, 6 families", fontsize=TINY,
            fontweight="bold", ha="left", va="center", color=INK)
    bw, x0 = 0.315, X[1] + 0.50
    for j, f in enumerate([0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]):
        ax.add_patch(Rectangle((x0 + j * (bw + 0.115), 1.30), bw,
                               0.18 + 0.98 * f, facecolor=cmap(0.12 + 0.72 * f),
                               edgecolor="none", zorder=3))
        ax.text(x0 + j * (bw + 0.115) + bw / 2, 1.14,
                f"{int(f*100)}", fontsize=TINY - 0.2, ha="center", va="top", color=MUTED)
    ax.text(X[1] + PW / 2, 2.72, "labelling budget", fontsize=LABEL,
            fontweight="bold", color=INK, **T)
    ax.text(X[1] + PW / 2, 0.66, "% of spots measured per slide", fontsize=TINY,
            color=MUTED, **T)

    # ------------------------------------------------------------- (c)
    cx = X[2] + PW / 2
    ax.text(cx, 7.22, "selected H&E patches", fontsize=SMALL, color=MUTED, **T)
    for j in range(4):
        px = X[2] + 0.44 + j * 0.80
        ax.add_patch(Rectangle((px, 6.32), 0.60, 0.60, facecolor="#EBD7E4",
                               edgecolor=fam[j], lw=1.1, zorder=3))
        rr = np.random.default_rng(20 + j)
        ax.scatter(rr.uniform(px + .06, px + .54, 28),
                   rr.uniform(6.38, 6.86, 28), s=1.5, c="#8E5E86",
                   zorder=4, linewidths=0)
    ax.add_patch(FancyArrowPatch((cx, 6.24), (cx, 5.86), arrowstyle="-|>",
                                 mutation_scale=7, lw=1.0, color=SLATE, zorder=4))
    hs = [1.00, 0.76, 0.55, 0.38, 0.26]
    for j, h in enumerate(hs):
        ax.add_patch(Rectangle((cx - 1.42 + j * 0.44, 5.30 - h / 2), 0.30, h,
                               facecolor=plt.get_cmap("Blues")(0.32 + 0.11 * j),
                               edgecolor="white", lw=0.6, zorder=3))
    ax.add_patch(Rectangle((cx + 0.86, 5.30 - 0.22), 0.38, 0.44,
                           facecolor=TEAL, edgecolor="white", lw=0.6, zorder=3))
    ax.text(cx - 0.30, 5.98, "UNI ViT-L, 304 M params, end-to-end", fontsize=TINY,
            ha="center", va="center", color=INK)
    ax.text(cx + 1.05, 4.88, "head", fontsize=TINY, ha="center", va="center", color=MUTED)
    ax.add_patch(FancyArrowPatch((cx, 4.74), (cx, 4.40), arrowstyle="-|>",
                                 mutation_scale=7, lw=1.0, color=SLATE, zorder=4))
    p3 = blob(cx, 3.46, 1.16, 0.76, seed=3)
    tissue(ax, p3, lw=0.8)
    s3 = hex_spots(p3, 0.148)
    v3 = field(s3, seed=6, n_centres=5, width=0.40)
    ax.add_collection(PatchCollection([Circle(p, 0.064) for p in s3],
                                      facecolor=plt.get_cmap("viridis")(v3),
                                      edgecolor="none", zorder=3))
    for j in range(24):
        ax.add_patch(Rectangle((cx + 1.28, 2.90 + j * 0.047), 0.14, 0.047,
                               facecolor=plt.get_cmap("viridis")(j / 23),
                               edgecolor="none", zorder=3))
    ax.text(cx, 2.44, "expression predicted at every unmeasured spot",
            fontsize=TINY, color=MUTED, **T)
    ax.text(cx, 1.96, "evaluated on held-out patients", fontsize=LABEL,
            fontweight="bold", color=INK, **T)
    icx, icy = [X[2] + 0.80, cx, X[2] + PW - 0.80], 1.32
    rr = np.random.default_rng(7)
    q = np.c_[rr.uniform(-1, 1, 12), rr.uniform(-1, 1, 12)] * 0.26
    for a_ in range(12):
        for b_ in range(a_ + 1, 12):
            if np.hypot(*(q[a_] - q[b_])) < 0.26:
                ax.plot([icx[0] + q[a_][0], icx[0] + q[b_][0]],
                        [icy + q[a_][1], icy + q[b_][1]], color=MUTED, lw=0.4, zorder=3)
    ax.scatter(icx[0] + q[:, 0], icy + q[:, 1], s=7, c=TEAL, zorder=4, linewidths=0)
    for dx, dy, c in [(-.15, .08, ROSE), (.13, .11, TEAL), (0, -.12, SLATE),
                      (.19, -.05, PURPLE), (-.10, -.10, AMBER)]:
        ax.add_patch(Circle((icx[1] + dx, icy + dy), 0.135, facecolor=c,
                            edgecolor="white", lw=0.55, alpha=0.88, zorder=3))
    for g in range(6):
        ax.plot([icx[2] - .32, icx[2] + .32], [icy - .32 + g * .128] * 2,
                color=HAIR, lw=.4, zorder=2)
        ax.plot([icx[2] - .32 + g * .128] * 2, [icy - .32, icy + .32],
                color=HAIR, lw=.4, zorder=2)
    th = np.linspace(0, 2 * np.pi, 120)
    ax.plot(icx[2] + .25 * np.cos(th) * (1 + .26 * np.sin(3 * th)),
            icy + .25 * np.sin(th) * (1 + .26 * np.cos(2 * th)),
            color=TEAL, lw=1.2, zorder=4)
    for xx, lb in zip(icx, ["Gene\nexpression", "Cell-type\ncomposition",
                            "Spatial\nfidelity"]):
        ax.text(xx, 0.74, lb, fontsize=TINY, color=INK, linespacing=1.2, **T)

    # ------------------------------------------------------------- (d)
    R = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    sat = {"her2": 5, "kidney": 50, "bc": 75}
    col = {"her2": TEAL, "kidney": ROSE, "bc": PURPLE}
    nice = {"her2": "HER2", "kidney": "Kidney", "bc": "BC"}
    for j, tag in enumerate(("her2", "kidney", "bc")):
        f = ROOT / f"results_pg/budget/controlled/pg_{tag}_hvg300/results.json"
        sub = fig.add_axes([(X[3] + 0.92) / W, (5.28 - j * 2.06) / H,
                            (PW - 1.32) / W, 1.44 / H])
        for sp in ("top", "right"):
            sub.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            sub.spines[sp].set_linewidth(0.55)
        if not f.exists():
            continue
        d = json.load(open(f))
        M = np.array([[r["pcc_per_gene_by_slide"][i] for i in range(7)]
                      for r in d["detailed"]["random"]])
        m, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
        x = np.array(R) * 100
        sub.axhline(m[-1], ls="--", lw=0.6, color="0.45")
        sub.fill_between(x, m - se, m + se, color=col[tag], alpha=0.18, lw=0)
        sub.plot(x, m, "-o", color=col[tag], lw=1.3, ms=2.3)
        s = sat[tag]
        sub.plot([s], [m[R.index(s / 100)]], marker="*", ms=9, color=col[tag],
                 mec="white", mew=0.6, zorder=5)
        sub.set_xscale("log"); sub.set_xticks([5, 20, 100])
        sub.set_xticklabels(["5", "20", "100"], fontsize=TINY)
        sub.tick_params(labelsize=TINY, width=0.55, length=1.9, pad=1.2)
        lo, hi = (m - se).min(), (m + se).max(); pad = (hi - lo) * 0.34
        sub.set_ylim(lo - pad, hi + pad)
        sub.set_title(f"{nice[tag]} — {s}% is enough", fontsize=SMALL,
                      color=col[tag], fontweight="bold", pad=2.5)
        sub.set_ylabel("per-slide PCC", fontsize=TINY, labelpad=1.5)
        if j == 2:
            sub.set_xlabel("labelled spots (%)", fontsize=TINY, labelpad=1.0)
        if j == 0:
            sub.text(0.97, 0.06, "full supervision", transform=sub.transAxes,
                     ha="right", va="bottom", fontsize=TINY - 0.4, color="0.45")
    ax.text(X[3] + PW / 2, 0.52,
            "3 cohorts · 127 slides · 70,219 spots · 53 patients\n"
            "patient-level 4-fold CV · no strategy beat random",
            fontsize=TINY, color=MUTED, linespacing=1.3, **T)

    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig1_pipeline.{ext}", facecolor="white",
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("  wrote figures/fig1_pipeline.pdf / .svg / .png")


if __name__ == "__main__":
    main()
    sys.exit(0)
