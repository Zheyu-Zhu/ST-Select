"""The individual pieces of the workflow figure, as transparent layers.

Everything is written twice, SVG and PNG, on a transparent background, so the
pieces can be dropped into a slide or an Illustrator layout and placed freely.

Where a piece carries data it uses measured data. That matters for the two
charts: the version they are replacing shows HER2 rising steeply to a plateau
and saturating at 5%, and shows every coloured strategy curve sitting above the
black random line beneath a caption saying no strategy beats random. Neither is
what the runs produced. Measured on ST-Net end-to-end over HER2:

    budget      5%      10%     30%     50%
    random    0.1295  0.1940  0.2361  0.2367     saturates at 30%, not 5%
    coreset  -0.0412 -0.0189 +0.0081 +0.0055     AL is worse when the budget
    mc_drop  -0.0286 -0.0129 +0.0017 +0.0072     is small, better when large

    python scripts/make_assets.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
PCC = Path(r"C:\Users\zheyu\workspace\stnet_ref\ST-Net\her2_pcc_by_fold.csv")
CM = 1 / 2.54

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.dpi": 400, "savefig.transparent": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

INK, MUTED = "#1C1B18", "#6E6A61"
TEAL, ORANGE, PURPLE = "#1F6F78", "#D2601A", "#4B3F8F"


def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.svg / .png")


# ------------------------------------------------------------------ 1. money
def money():
    fig, ax = plt.subplots(figsize=(3.4 * CM, 2.4 * CM))
    ax.set_xlim(0, 34); ax.set_ylim(0, 24); ax.axis("off"); ax.set_aspect("equal")
    for i, (dx, dy) in enumerate(((5.0, 8.4), (2.5, 4.2), (0, 0))):
        ax.add_patch(FancyBboxPatch((2 + dx, 2 + dy), 24, 11,
                                    boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor="white", edgecolor=INK, lw=1.6, zorder=i))
    ax.add_patch(Circle((14, 7.5), 3.1, facecolor="none", edgecolor=INK, lw=1.6, zorder=9))
    ax.text(14, 7.5, "$", ha="center", va="center", fontsize=11,
            fontweight="bold", color=INK, zorder=10)
    for x in (5.6, 22.4):
        ax.add_patch(Circle((x, 7.5), 0.55, facecolor=INK, zorder=9))
    save(fig, "money")


# ----------------------------------------------------------------- 2. budget
def budget_bar(ramp="blue"):
    """The wedge on its own - no ticks, no labels, no title.

    Neutral grey, deliberately. Blue and orange already carry expression on the
    spot maps, where dark blue means low expression; reusing that hue for a high
    budget puts one colour on two variables pointing opposite ways. A budget
    scale is structure, like an axis or an arrow, not a measurement.

    Without labels the wedge says only "left is less, right is more", which is
    all a legend strip has to say. Whatever budgets end up being quoted can be
    typed alongside it in the layout, and they stay editable there instead of
    being baked into a raster.
    """
    stops = {"blue": ["#EDEDEA", "#A8A8A2", "#3C3C38"]}[ramp]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("budget", stops)
    fig, ax = plt.subplots(figsize=(7.0 * CM, 0.85 * CM))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.imshow(np.linspace(0, 1, 512).reshape(1, -1), extent=[0, 1, 0, 1],
              aspect="auto", cmap=cmap, zorder=1)
    clip = Polygon([[0, 0.0], [1, 0.0], [1, 1.0], [0, 0.48]], closed=True,
                   transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    for im in ax.images:
        im.set_clip_path(clip)
    save(fig, "budget_bar")


# ---------------------------------------------------------------- 3/4. patches
def patches(slide="TENX106", n=6, size=1400, seed=5):
    """Cut from this slide's own WSI at its own spot coordinates.

    The earlier version pulled from SPA125's patch archive, which is a different
    slide in a different cohort - fine as generic H&E, wrong if the panel is
    meant to show what went into this model.
    """
    import anndata
    import tifffile
    wsi = ROOT / "hest_data" / "wsis" / f"{slide}.tif"
    h5 = ROOT / "hest_data" / "st" / f"{slide}.h5ad"
    if not (wsi.exists() and h5.exists()):
        print(f"  [skip] patches: need {wsi.name} and {h5.name}"); return
    from PIL import Image as PILImage
    xy = np.asarray(anndata.read_h5ad(h5).obsm["spatial"], dtype=float)
    page = tifffile.TiffFile(wsi).pages[0]
    H, W = page.shape[:2]
    img = page.asarray()

    rng = np.random.default_rng(seed)
    half = size // 2
    ok = ((xy[:, 0] > half) & (xy[:, 0] < W - half) &
          (xy[:, 1] > half) & (xy[:, 1] < H - half))
    # Spot coordinates sit on tissue, but a 1400 px window around one near the
    # section edge is mostly slide. Score candidates by how much of the window
    # is stained and keep the full ones, or half the row comes out blank.
    cand = rng.permutation(np.where(ok)[0])[:120]
    scored = []
    for i in cand:
        x, y = int(xy[i, 0]), int(xy[i, 1])
        c = img[y - half:y + half:8, x - half:x + half:8].astype(np.float32)
        mxv, mnv = c.max(2), c.min(2)
        scored.append((float(((mxv - mnv) / (mxv + 1e-6) > 0.12).mean()), int(i)))
    scored.sort(reverse=True)
    pick = [i for _, i in scored[:n * 3]]
    pick = rng.choice(pick, n, replace=False)
    imgs = []
    for i in pick:
        x, y = int(xy[i, 0]), int(xy[i, 1])
        crop = img[y - half:y + half, x - half:x + half]
        imgs.append(np.asarray(PILImage.fromarray(crop).resize((448, 448),
                                                               PILImage.LANCZOS)))
    del img
    print(f"  patches cut from {slide} at {size} px around real spots")

    fig, axes = plt.subplots(1, n, figsize=(n * 1.9 * CM, 1.9 * CM))
    for ax, im, c in zip(np.atleast_1d(axes), imgs,
                         [TEAL, ORANGE, PURPLE, "#C0407A", "#3F7A52", "#B4622A"]):
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(c); sp.set_linewidth(2.0)
    fig.subplots_adjust(wspace=0.12)
    save(fig, f"patches_row_{slide}")

    fig, ax = plt.subplots(figsize=(3.6 * CM, 3.2 * CM))
    ax.set_xlim(0, 36); ax.set_ylim(0, 32); ax.axis("off"); ax.set_aspect("equal")
    for i, im in enumerate(imgs[:4][::-1]):
        x, y = 2 + i * 2.6, 2 + i * 2.6
        ax.imshow(im, extent=[x, x + 22, y, y + 22], zorder=i, interpolation="bilinear")
        ax.add_patch(Rectangle((x, y), 22, 22, facecolor="none",
                               edgecolor="white", lw=1.6, zorder=i + 0.5))
    save(fig, f"patches_stack_{slide}")


# ------------------------------------------------- 5. prediction map, TENX106
def prediction_map(slide="TENX106"):
    """The model's output as a transparent layer registered to the tissue PNG.

    Same canvas and same transform as scripts/spot_overlay.py, so it drops onto
    wsi_<slide>_clean_rgba.png without scaling. The earlier version was a
    free-standing scatter on its own axes - it could not be laid over anything.

    The field is smooth and low-frequency so the hexes merge into regions, which
    is what a prediction over a whole section looks like. The values are mock.
    """
    import anndata
    import tifffile
    from PIL import ImageDraw
    from PIL import Image as PILImage
    from scipy.ndimage import (binary_closing, binary_fill_holes,
                               binary_opening, label)

    wsi = ROOT / "hest_data" / "wsis" / f"{slide}.tif"
    h5 = ROOT / "hest_data" / "st" / f"{slide}.h5ad"
    if not (wsi.exists() and h5.exists()):
        print(f"  [skip] prediction map: need {wsi.name} and {h5.name}"); return

    level, mask_level, pad = 1, 3, 8
    tf = tifffile.TiffFile(wsi)
    a = tf.pages[mask_level].asarray().astype(np.float32)
    mx, mn = a.max(2), a.min(2)
    m = ((mx - mn) / (mx + 1e-6) > 0.12) & (mx > 40)
    m = binary_closing(binary_opening(m, np.ones((5, 5))), np.ones((25, 25)))
    lab, n = label(m)
    if n:
        m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    m = binary_fill_holes(m)
    ys, xs = np.where(m)
    y0, y1 = max(0, ys.min() - pad), ys.max() + pad
    x0, x1 = max(0, xs.min() - pad), xs.max() + pad
    sc = 2 ** (mask_level - level)
    H0, W0 = tf.pages[level].shape[:2]
    top, left = int(y0 * sc), int(x0 * sc)
    h = min(int(y1 * sc), H0) - top
    w = min(int(x1 * sc), W0) - left

    occ = np.asarray(PILImage.fromarray(m[y0:y1, x0:x1].astype(np.uint8) * 255)
                     .resize((w, h), PILImage.BILINEAR)) > 127
    cols = 26                      # fine enough that the discs read as regions
    pitch = w / cols
    pts = []
    for j, yy in enumerate(np.arange(pitch / 2, h, pitch * np.sqrt(3) / 2)):
        off = 0.0 if j % 2 == 0 else pitch / 2
        for xx in np.arange(pitch / 2, w, pitch) + off:
            if xx < w and occ[int(yy), int(xx)]:
                pts.append((xx, yy))
    P = np.array(pts)

    # A blurred random field, not a sum of sines. Sines gave clean diagonal
    # stripes - regular in a way tissue never is. Smoothed noise produces
    # irregular contiguous patches, which is what a prediction over a section
    # actually looks like.
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(11)
    gh, gw = 240, max(int(240 * w / h), 24)
    fld = gaussian_filter(rng.normal(size=(gh, gw)), sigma=(14, 14), mode="reflect")
    fld = (fld - fld.min()) / max(float(np.ptp(fld)), 1e-9)
    gi = np.clip((P[:, 1] / h * (gh - 1)).astype(int), 0, gh - 1)
    gj = np.clip((P[:, 0] / w * (gw - 1)).astype(int), 0, gw - 1)
    v = fld[gi, gj]
    v = (v - v.min()) / max(float(np.ptp(v)), 1e-9)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "pred", ["#2E5F8A", "#6F9BC0", "#EDD9A3", "#E8A05C", "#7FA86B"])
    canvas = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(canvas)
    r = pitch * 0.52               # slight overlap so regions read as continuous
    for (x, y), vv in zip(P, v):
        c = tuple(int(255 * t) for t in cmap(float(vv))[:3]) + (255,)
        dr.ellipse([x - r, y - r, x + r, y + r], fill=c)

    out = OUT / f"prediction_map_{slide}.png"
    canvas.save(out)
    prev = out.with_name(out.stem + "_preview.png")
    im = canvas.copy(); im.thumbnail((1200, 1200), PILImage.LANCZOS); im.save(prev)
    print(f"  prediction_map_{slide}.png  {w}x{h}  {len(P)} hexes  (transparent, "
          f"registers on wsi_{slide}_clean_rgba.png)")


# ------------------------------------------------------------- 6. the curves
def curves():
    if not PCC.exists():
        print(f"  [skip] curves: {PCC} missing"); return
    d = pd.read_csv(PCC).dropna(subset=["pcc"])
    m = d.groupby(["sampler", "budget"]).pcc.mean().reset_index()
    x = np.array([5, 10, 30, 50])

    # (a) the budget curve, measured
    r = m[m.sampler == "random"].sort_values("budget").pcc.values
    fig, ax = plt.subplots(figsize=(6.4 * CM, 4.4 * CM))
    ax.axhline(r[-1], ls="--", lw=0.9, color="0.45")
    # PCHIP through the measured points: it passes through every one of them
    # and cannot overshoot between them, so the curve is smooth without
    # inventing a bump the four budgets do not support.
    from scipy.interpolate import PchipInterpolator
    xs = np.linspace(x.min(), x.max(), 300)
    ax.plot(xs, PchipInterpolator(x, r)(xs), color=TEAL, lw=2.4,
            solid_capstyle="round", zorder=2)
    ax.plot(x, r, "o", color=TEAL, ms=5, zorder=3)
    ax.annotate("saturates at 30%", (30, r[2]), (0.42, 0.34),
                textcoords="axes fraction", fontsize=7, color=MUTED,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED))
    ax.set_xticks(x); ax.set_xticklabels([f"{v}%" for v in x], fontsize=7.5)
    ax.set_xlabel("Labelled spots", fontsize=8.5)
    ax.set_ylabel("Per-slide PCC", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "curve_budget")

    # (b) absolute per-slide PCC, one line per sampler.
    # A budget is drawn only where the sampler has all four folds. typiclust's
    # 0.3 has three, and its three-fold mean (0.2592) is not comparable with the
    # others' four-fold means - fold 3 is missing and the folds differ by 4-7x,
    # so including it would put typiclust above every other line for a reason
    # that has nothing to do with the sampler.
    piv = d.pivot_table(index=["budget", "fold"], columns="sampler", values="pcc")
    cols = {"random": "black", "coreset": ORANGE, "mc_dropout": PURPLE,
            "tod": "#3F7A52", "typiclust": "#C0407A"}
    fig, ax = plt.subplots(figsize=(7.4 * CM, 5.2 * CM))
    from scipy.interpolate import PchipInterpolator
    for s_, c in cols.items():
        if s_ not in piv.columns:
            continue
        xs, ys = [], []
        for b in sorted(piv.index.get_level_values(0).unique()):
            g = piv.loc[b][s_].dropna()
            if len(g) < 4:
                continue
            xs.append(b * 100); ys.append(g.mean())
        if len(xs) < 2:
            continue
        xs_, ys_ = np.array(xs), np.array(ys)
        lw = 2.6 if s_ == "random" else 1.7
        if len(xs_) >= 3:
            gg = np.linspace(xs_.min(), xs_.max(), 300)
            ax.plot(gg, PchipInterpolator(xs_, ys_)(gg), color=c, lw=lw,
                    zorder=4 if s_ == "random" else 3)
        else:
            ax.plot(xs_, ys_, color=c, lw=lw, zorder=3)
        ax.plot(xs_, ys_, "o", color=c, ms=4, label=s_,
                zorder=5 if s_ == "random" else 4)
    ax.set_xticks(x); ax.set_xticklabels([f"{v}%" for v in x], fontsize=7.5)
    ax.set_xlabel("Labelled spots", fontsize=8.5)
    ax.set_ylabel("Per-slide PCC", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(frameon=False, fontsize=6.8, loc="lower right", handlelength=1.4,
              labelspacing=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "curve_strategies")


# ------------------------------------------------------------- 7. the people
def people():
    def person(ax, x, y, col, s=1.0):
        ax.add_patch(Circle((x, y + 1.55 * s), 0.62 * s, facecolor=col, lw=0))
        ax.add_patch(FancyBboxPatch((x - 0.78 * s, y - 0.55 * s), 1.56 * s, 1.7 * s,
                                    boxstyle=f"round,pad=0,rounding_size={0.62*s}",
                                    facecolor=col, lw=0))

    fig, ax = plt.subplots(figsize=(8.6 * CM, 3.4 * CM))
    ax.set_xlim(0, 26); ax.set_ylim(0, 10); ax.axis("off"); ax.set_aspect("equal")
    for i, c in enumerate([TEAL, TEAL, ORANGE]):
        person(ax, 2.2 + i * 2.3, 4.6, c, 1.15)
    ax.text(4.5, 1.6, "Many spots\nfrom a few patients", ha="center", va="center",
            fontsize=7.5, color=INK)
    for i, c in enumerate([TEAL, ORANGE, "#8FBFC4", "#3F7A52", PURPLE]):
        person(ax, 13.4 + i * 2.0, 4.6, c, 0.92)
    ax.text(17.4, 1.6, "Fewer spots across\nmore patients", ha="center", va="center",
            fontsize=7.5, color=INK)
    ax.annotate("", (24.4, 8.0), (24.4, 2.6),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color=INK,
                                mutation_scale=16))
    save(fig, "people_breadth_depth")


if __name__ == "__main__":
    print("writing transparent assets to figures/assets/")
    jobs = [money, budget_bar, patches, prediction_map, curves, people]
    for fn in jobs:
        try:
            fn()
        except Exception as exc:
            print(f"  [skip] {fn.__name__}: {type(exc).__name__}: {exc}")
    sys.exit(0)
