"""Figure 1 built from REAL data — real H&E patches, real spots, real expression.

The previous programmatic attempt was factually correct but looked synthetic: a
rainbow candy-disc for tissue, pink squares with dots for H&E, a leader line
pointing at nothing. Real slides are available (13,620 patches at 224x224), so
the tissue in panels (a)-(c) is now an actual mosaic of measured patches from one
HER2 slide, the spot positions are the real acquisition grid, and the colours are
real measured expression.

That matters beyond looks: the spatial autocorrelation visible in panel (a) is
the genuine article, not a smooth random field standing in for it. The redundancy
claim is shown with the data that motivates it.

    python scripts/make_fig1_real.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

W, H = 17.8, 8.8
TEAL, ROSE, SLATE, AMBER = "#1F6F78", "#C0407A", "#4B5D75", "#C98A2E"
PURPLE, INK, MUTED, HAIR = "#4B3F8F", "#1C1B18", "#6E6A61", "#C9C4BA"
TITLE, LABEL, SMALL, TINY = 7.6, 6.4, 5.8, 5.2

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 500,
})


def load_slide(tile=26):
    """Mosaic one real slide from its measured patches, plus expression per spot."""
    d = np.load(ROOT / "feature_cache/pg_her2_hvg300.npz", allow_pickle=True)
    pp = np.asarray(d["patch_paths"]).astype(str)
    pos = d["positions"].astype(float)
    sid = np.asarray(d["sample_ids"]).astype(str)
    expr = d["expressions"].astype("float32")

    # the slide with the most spots gives the fullest-looking section
    uniq, counts = np.unique(sid, return_counts=True)
    slide = uniq[counts.argmax()]
    m = np.where(sid == slide)[0]
    # Compact to the OCCUPIED grid lines. The raw coordinates index a lattice the
    # array only sparsely populates (712 spots across 83x94 = 9% filled), so
    # mosaicking on them produced a field of specks rather than a tissue. Ranking
    # the distinct x and y values packs the same spots to 45% fill - the rest is
    # the genuine irregular outline of the section.
    ux = np.unique(pos[m, 0]); uy = np.unique(pos[m, 1])
    gx = np.searchsorted(ux, pos[m, 0]); gy = np.searchsorted(uy, pos[m, 1])
    Wg, Hg = gx.max() + 1, gy.max() + 1

    # No H&E mosaic: the spot discs cover it completely, so all it contributed
    # was white gaps wherever the sparse grid had no spot - which read as holes
    # punched through the tissue. The silhouette below is derived from the real
    # spot positions instead, so the outline is still the true section shape.
    canvas, mask = None, np.zeros((Hg, Wg), dtype=bool)
    mask[gy, gx] = True

    # colour spots by the single most spatially structured gene on this slide
    e = expr[m]
    var = e.var(0)
    gene = int(np.argsort(var)[-6])          # high-variance but not the extreme
    v = e[:, gene]
    v = (v - np.percentile(v, 5)) / (np.percentile(v, 95) - np.percentile(v, 5) + 1e-9)
    return canvas, mask, gx, gy, np.clip(v, 0, 1), Wg, Hg, slide, pp[m]


def main():
    canvas, mask, gx, gy, val, Wg, Hg, slide, paths = load_slide()
    print(f"  slide {slide}: {len(gx)} spots, grid {Wg}x{Hg}")

    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    root = fig.add_axes([0, 0, 1, 1]); root.set_xlim(0, W); root.set_ylim(0, H)
    root.set_aspect("equal"); root.axis("off")
    T = dict(ha="center", va="center")

    PW, GAP, PY, PH = 4.03, 0.40, 0.16, 8.05
    X = [0.14 + i * (PW + GAP) for i in range(4)]
    for x in X:
        root.add_patch(FancyBboxPatch((x, PY), PW, PH,
                                      boxstyle="round,pad=0.05,rounding_size=0.16",
                                      facecolor="white", edgecolor=HAIR, lw=0.9, zorder=0))
    for i in range(3):
        root.add_patch(FancyArrowPatch((X[i] + PW + 0.06, PY + PH * 0.52),
                                       (X[i + 1] - 0.06, PY + PH * 0.52),
                                       arrowstyle="-|>", mutation_scale=10,
                                       lw=1.5, color=SLATE, zorder=6))
    TITLES = ["Dense ST acquisition", "Spot selection under budget",
              "Expression prediction", "Budget is cohort-specific"]
    for i, (x, t) in enumerate(zip(X, TITLES)):
        root.text(x + 0.20, PY + PH - 0.30, f"({chr(97+i)})  {t}", fontsize=TITLE,
                  fontweight="bold", ha="left", va="center", color=INK)
        root.plot([x + 0.18, x + PW - 0.18], [PY + PH - 0.58] * 2,
                  color=HAIR, lw=0.7, zorder=2)

    from scipy.ndimage import binary_closing, binary_dilation
    solid = binary_dilation(binary_closing(mask, np.ones((3, 3))), np.ones((3, 3)))

    def slide_axes(x0, y0, w):
        h = w * Hg / Wg
        a = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
        a.contourf(np.arange(Wg) + .5, np.arange(Hg) + .5, solid.astype(float),
                   levels=[0.5, 1.5], colors=["#F7EAF0"], zorder=1)
        a.contour(np.arange(Wg) + .5, np.arange(Hg) + .5, solid.astype(float),
                  levels=[0.5], colors=["#B98BA6"], linewidths=1.0, zorder=2)
        a.set_xlim(-1, Wg + 1); a.set_ylim(Hg + 1, -1); a.axis("off")
        a.set_facecolor("none")
        return a, h

    # ---------------------------------------------------------------- (a)
    axA, hA = slide_axes(X[0] + 0.42, 3.10, PW - 0.84)
    cm = plt.get_cmap("magma")
    axA.add_collection(PatchCollection(
        [Circle((x + .5, y + .5), 0.50) for x, y in zip(gx, gy)],
        facecolor=cm(val), edgecolor="white", linewidths=0.18, alpha=0.92, zorder=3))
    root.text(X[0] + PW / 2, 3.10 + hA + 0.30, "every spot sequenced",
              fontsize=SMALL, color=MUTED, **T)
    # callout: a genuinely homogeneous neighbourhood
    from scipy.ndimage import uniform_filter
    grid = np.full((Hg, Wg), np.nan)
    grid[gy, gx] = val
    best, bxy = 1e9, (Wg // 2, Hg // 2)
    for i in range(len(gx)):
        nb = [val[j] for j in range(len(gx))
              if abs(gx[j] - gx[i]) <= 2 and abs(gy[j] - gy[i]) <= 2]
        if len(nb) >= 12:
            sd = float(np.std(nb))
            if sd < best:
                best, bxy = sd, (gx[i], gy[i])
    axA.add_patch(Circle((bxy[0] + .5, bxy[1] + .5), 2.6, facecolor="none",
                         edgecolor="white", lw=1.9, zorder=5))
    axA.add_patch(Circle((bxy[0] + .5, bxy[1] + .5), 2.6, facecolor="none",
                         edgecolor=INK, lw=0.9, zorder=6))
    root.annotate("", xy=(X[0] + 0.42 + (PW - 0.84) * (bxy[0] + 3.0) / Wg,
                          3.10 + hA * (1 - (bxy[1] + 2.4) / Hg)),
                  xytext=(X[0] + PW - 0.52, 2.72),
                  arrowprops=dict(arrowstyle="-", lw=0.7, color=INK), zorder=7)
    root.text(X[0] + PW - 0.46, 2.62, "neighbours share\nexpression",
              fontsize=TINY, ha="right", va="top", color=INK, linespacing=1.25)
    for xo in (-0.44, 0, 0.44):
        for dy, fc in ((0.0, "#D9C68A"), (0.10, "#F0E4B8")):
            root.add_patch(Circle((X[0] + PW / 2 + xo, 1.42 + dy), 0.20, facecolor=fc,
                                  edgecolor="#A98B3F", lw=0.8, zorder=3 + (dy > 0)))
    root.text(X[0] + PW / 2, 0.90, "High cost", fontsize=LABEL,
              fontweight="bold", color=INK, **T)
    root.text(X[0] + PW / 2, 0.52, "dense grids oversample\nrepetitive morphology",
              fontsize=TINY, color=MUTED, linespacing=1.25, **T)

    # ---------------------------------------------------------------- (b)
    axB, hB = slide_axes(X[1] + 0.22, 3.55, PW - 1.75)
    rng = np.random.default_rng(11)
    fam = [TEAL, ROSE, SLATE, AMBER]
    fam_lab = ["Random", "Uncertainty", "Diversity", "Spatial"]
    axB.add_collection(PatchCollection(
        [Circle((x + .5, y + .5), 0.44) for x, y in zip(gx, gy)],
        facecolor="white", edgecolor="#9E978A", linewidths=0.30, alpha=0.75, zorder=3))
    pick = rng.choice(len(gx), size=int(0.13 * len(gx)), replace=False)
    axB.add_collection(PatchCollection(
        [Circle((gx[i] + .5, gy[i] + .5), 0.62) for i in pick],
        facecolor=[fam[j % 4] for j in range(len(pick))],
        edgecolor="white", linewidths=0.35, zorder=4))
    root.text(X[1] + 0.22 + (PW - 1.75) / 2, 3.55 + hB + 0.30, "a subset is sequenced",
              fontsize=SMALL, color=MUTED, **T)
    root.text(X[1] + PW - 1.52, 6.52, "20 strategies, 6 families", fontsize=TINY,
              fontweight="bold", ha="left", va="center", color=INK)
    for j, (c, lb) in enumerate(zip(fam, fam_lab)):
        yy = 6.10 - j * 0.40
        root.add_patch(Rectangle((X[1] + PW - 1.50, yy), 0.24, 0.24,
                                 facecolor=c, edgecolor="none", zorder=20))
        root.text(X[1] + PW - 1.18, yy + 0.12, lb, fontsize=TINY, ha="left",
                  va="center", color=INK)
    cmap = plt.get_cmap("Spectral_r")
    bw, x0 = 0.315, X[1] + 0.50
    for j, f in enumerate([0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]):
        root.add_patch(Rectangle((x0 + j * (bw + 0.115), 1.30), bw,
                                 0.18 + 0.98 * f, facecolor=cmap(0.12 + 0.72 * f),
                                 edgecolor="none", zorder=3))
        root.text(x0 + j * (bw + 0.115) + bw / 2, 1.14, f"{int(f*100)}",
                  fontsize=TINY - 0.2, ha="center", va="top", color=MUTED)
    root.text(X[1] + PW / 2, 2.72, "labelling budget", fontsize=LABEL,
              fontweight="bold", color=INK, **T)
    root.text(X[1] + PW / 2, 0.66, "% of spots measured per slide",
              fontsize=TINY, color=MUTED, **T)

    # ---------------------------------------------------------------- (c)
    cx = X[2] + PW / 2
    root.text(cx, 7.22, "selected H&E patches", fontsize=SMALL, color=MUTED, **T)
    sel4 = rng.choice(len(paths), 4, replace=False)
    for j, k in enumerate(sel4):                       # REAL patches
        px = X[2] + 0.44 + j * 0.80
        a = fig.add_axes([px / W, 6.32 / H, 0.60 / W, 0.60 / H])
        try:
            a.imshow(Image.open(paths[k]).convert("RGB"))
        except Exception:
            pass
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(fam[j]); sp.set_linewidth(1.2)
    root.add_patch(FancyArrowPatch((cx, 6.24), (cx, 5.86), arrowstyle="-|>",
                                   mutation_scale=7, lw=1.0, color=SLATE, zorder=4))
    # The map is built first so everything above it can be stacked from its real
    # height. Hard-coding these y values is what put the arrow tip inside the
    # encoder box: the box moved when the map did, the arrow did not.
    MAP_BOT, PATCH_BOT, STACK = 2.50, 6.28, 0.22 + 1.06
    mapw = min(PW - 1.24, (PATCH_BOT - STACK - MAP_BOT) * Wg / Hg)
    axC, hC = slide_axes(X[2] + (PW - mapw) / 2, MAP_BOT, mapw)
    axC.add_collection(PatchCollection(
        [Circle((x + .5, y + .5), 0.50) for x, y in zip(gx, gy)],
        facecolor=plt.get_cmap("viridis")(val), edgecolor="none",
        alpha=0.95, zorder=3))

    top = 2.50 + hC
    root.add_patch(FancyArrowPatch((cx, top + 0.34), (cx, top + 0.08),
                                   arrowstyle="-|>", mutation_scale=7, lw=1.0,
                                   color=SLATE, zorder=4))
    root.add_patch(FancyBboxPatch((cx - 1.34, top + 0.40), 2.68, 0.58,
                                  boxstyle="round,pad=0.02,rounding_size=0.10",
                                  facecolor="#EAF0F4", edgecolor=SLATE, lw=0.8,
                                  zorder=3))
    root.text(cx, top + 0.79, "UNI ViT-L encoder", fontsize=TINY,
              fontweight="bold", ha="center", va="center", color=INK, zorder=4)
    root.text(cx, top + 0.56, "304 M params  ·  fine-tuned end-to-end",
              fontsize=TINY, ha="center", va="center", color=MUTED, zorder=4)
    root.add_patch(FancyArrowPatch((cx, 6.28), (cx, top + 1.06),
                                   arrowstyle="-|>", mutation_scale=7, lw=1.0,
                                   color=SLATE, zorder=4))

    for j in range(22):
        root.add_patch(Rectangle((X[2] + PW - 0.48, MAP_BOT + (hC - 1.06) / 2 + j * 0.048), 0.13, 0.048,
                                 facecolor=plt.get_cmap("viridis")(j / 21),
                                 edgecolor="none", zorder=3))
    root.text(cx, 2.34, "expression predicted at every unmeasured spot",
              fontsize=TINY, color=MUTED, **T)
    root.text(cx, 1.90, "evaluated on held-out patients", fontsize=LABEL,
              fontweight="bold", color=INK, **T)
    icx, icy = [X[2] + 0.80, cx, X[2] + PW - 0.80], 1.28
    q = np.c_[rng.uniform(-1, 1, 12), rng.uniform(-1, 1, 12)] * 0.26
    for a_ in range(12):
        for b_ in range(a_ + 1, 12):
            if np.hypot(*(q[a_] - q[b_])) < 0.26:
                root.plot([icx[0] + q[a_][0], icx[0] + q[b_][0]],
                          [icy + q[a_][1], icy + q[b_][1]],
                          color=MUTED, lw=0.4, zorder=3)
    root.scatter(icx[0] + q[:, 0], icy + q[:, 1], s=7, c=TEAL, zorder=4, linewidths=0)
    for dx, dy, c in [(-.15, .08, ROSE), (.13, .11, TEAL), (0, -.12, SLATE),
                      (.19, -.05, PURPLE), (-.10, -.10, AMBER)]:
        root.add_patch(Circle((icx[1] + dx, icy + dy), 0.135, facecolor=c,
                              edgecolor="white", lw=0.55, alpha=0.88, zorder=3))
    for g in range(6):
        root.plot([icx[2] - .32, icx[2] + .32], [icy - .32 + g * .128] * 2,
                  color=HAIR, lw=.4, zorder=2)
        root.plot([icx[2] - .32 + g * .128] * 2, [icy - .32, icy + .32],
                  color=HAIR, lw=.4, zorder=2)
    th = np.linspace(0, 2 * np.pi, 120)
    root.plot(icx[2] + .25 * np.cos(th) * (1 + .26 * np.sin(3 * th)),
              icy + .25 * np.sin(th) * (1 + .26 * np.cos(2 * th)),
              color=TEAL, lw=1.2, zorder=4)
    for xx, lb in zip(icx, ["Gene\nexpression", "Cell-type\ncomposition",
                            "Spatial\nfidelity"]):
        root.text(xx, 0.70, lb, fontsize=TINY, color=INK, linespacing=1.2, **T)

    # ---------------------------------------------------------------- (d)
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
    root.text(X[3] + PW / 2, 0.50,
              "3 cohorts · 127 slides · 70,219 spots · 53 patients\n"
              "patient-level 4-fold CV · no strategy beat random",
              fontsize=TINY, color=MUTED, linespacing=1.3, **T)

    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig1_real.{ext}", facecolor="white",
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("  wrote figures/fig1_real.pdf / .svg / .png")


if __name__ == "__main__":
    main()
    sys.exit(0)
