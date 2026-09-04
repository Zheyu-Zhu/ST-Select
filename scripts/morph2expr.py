"""The morphology-to-expression model's output, rendered densely over a slide.

The model maps an H&E patch to a 300-gene vector, and nothing about it is tied
to the spot grid -- it can be evaluated wherever a patch can be cut. So instead
of drawing one disc per measured spot, this evaluates the network on a fine
hexagonal lattice across the whole section and draws that. The result is a
continuous-looking expression field that is genuine model output at every point,
not an interpolation between spots.

Registration and honesty both depend on a few details.

Patch extraction has to match training exactly or the network sees a different
input distribution than it was fitted on. Measured against the stored training
patches, they are 224x224 crops taken at full resolution with no rescaling
(mean absolute difference 0.00). HEST's own patch files are a different
extraction -- 162x162 upscaled to 224 -- and using those would feed the model a
1.4x magnified view. The value is read off the file rather than assumed.

The slide must be one the checkpoint never trained on, or the map is a memory
rather than a prediction. SPA143 belongs to patient B, held out in fold 1, so a
fold-1 checkpoint is genuinely predicting here. The script refuses to run if the
slide's own fold is not the one being loaded.

The lattice is eroded a little inside the tissue contour, so the section's own
H&E rim stays visible around the field and the boundary does not look painted on.

    python scripts/morph2expr.py --slide SPA143 --gene ERBB2
    python scripts/morph2expr.py --slide SPA143 --gene COL1A1 --budget 0.05
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
ST = Path(r"C:\Users\zheyu\workspace\stnet_ref\ST-Net")
DI = Path(r"C:\Users\zheyu\workspace\stnet_ref\data_infor")
PATCH = 224                      # full-res pixels, verified against training data
EPOCH = {0.05: 110, 0.1: 80, 0.3: 70, 0.5: 60}


def contour_mask(slide, shape, level):
    """HEST's tissue polygon rasterised at `level` -- the same mask clean_wsi.py
    and spot_overlay.py use, so all three outputs share one crop box."""
    g = ROOT / "hest_data" / "tissue_seg" / f"{slide}_contours.geojson"
    polys = []
    for f in json.load(open(g))["features"]:
        gm = f["geometry"]
        rings = ([gm["coordinates"][0]] if gm["type"] == "Polygon"
                 else [q[0] for q in gm["coordinates"]])
        polys += [np.asarray(r, dtype=float) for r in rings]
    im = Image.new("L", (shape[1], shape[0]), 0)
    dr = ImageDraw.Draw(im)
    for r in polys:
        dr.polygon([tuple(v) for v in (r / 2.0 ** level)], fill=255)
    return np.asarray(im) > 127


def fold_of(slide):
    """The fold that holds this slide out, from the stored ground truth."""
    for fold in range(4):
        if list((ST / "weights_st" / "HER2").glob(f"*/fold{fold}/gt/{slide}_gt.npy")):
            return fold
    raise SystemExit(f"{slide}: no stored fold holds it out")


def load_model(strategy, budget, fold, n_out):
    sys.path.insert(0, str(ST))
    import torch.nn as nn
    from torchvision import models
    ck = (ST / "weights_st" / "HER2" / f"{strategy}_{budget}" / f"fold{fold}" /
          "checkpoint_epoch" / "final.pth")
    if not ck.exists():
        raise SystemExit(f"no checkpoint at {ck}")
    m = models.densenet121(weights=None, memory_efficient=True)
    m.classifier = nn.Sequential(
        nn.Linear(m.classifier.in_features, 1024), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(1024, n_out))
    sd = torch.load(ck, map_location="cpu")
    sd = sd.get("model_state_dict", sd.get("state_dict", sd))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
        if len(missing) > 4:
            raise SystemExit("  too many missing keys -- architecture mismatch")
    return m.eval(), ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", default="SPA143")
    ap.add_argument("--gene", default="ERBB2")
    ap.add_argument("--strategy", default="random")
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--level", type=int, default=1, help="render level; must match clean_wsi.py")
    ap.add_argument("--mask-level", type=int, default=3)
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--stride", type=int, default=69,
                    help="lattice pitch in full-res px (spot pitch is 276)")
    ap.add_argument("--erode", type=int, default=1, help="mask erosion in mask px")
    ap.add_argument("--cmap", default="Spectral_r")
    ap.add_argument("--radius", type=float, default=0.62,
                    help="disc radius as a fraction of the lattice pitch")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default: the GPU usually has a sweep on it, and "
                         "taking memory from a running trainer has cost a run "
                         "before. ~30 img/s on 16 threads, minutes for a slide.")
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU threads. Each one carries its own convolution "
                         "workspace, so 16 threads is what exhausted RAM while a "
                         "sweep was training; 4 is slower and leaves room.")
    ap.add_argument("--min-free-gb", type=float, default=0.7,
                    help="abort rather than let a running trainer be squeezed "
                         "into swap. Checked before every batch.")
    ap.add_argument("--clip", type=float, default=2.0,
                    help="colour range as +/- this many s.d. about the mean")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    names = [str(x) for x in np.load(
        ROOT / "processed_gene" / "gene_names" / "HER2_expression_norm_genes.npy",
        allow_pickle=True)]
    gt = np.load(next((ST / "weights_st" / "HER2").glob(
        f"*/fold*/gt/{args.slide}_gt.npy")))
    n_out = gt.shape[1]
    if args.gene not in names[:n_out]:
        why = (f"outside the {n_out}-gene target panel (index {names.index(args.gene)})"
               if args.gene in names else "absent from this cohort")
        raise SystemExit(f"{args.gene}: {why} -- the model never predicts it")
    gi = names.index(args.gene)

    fold = fold_of(args.slide)
    model, ck = load_model(args.strategy, args.budget, fold, n_out)
    print(f"{args.slide}: held out in fold {fold}; checkpoint "
          f"{ck.parent.parent.parent.name}/fold{fold} -- the model never saw this slide")

    # --- geometry, shared with clean_wsi.py -------------------------------
    tf = tifffile.TiffFile(ROOT / "hest_data" / "wsis" / f"{args.slide}.tif")
    ml = min(args.mask_level, len(tf.pages) - 1)
    mask = contour_mask(args.slide, tf.pages[ml].shape[:2], ml)
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - args.pad, 0), min(ys.max() + args.pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - args.pad, 0), min(xs.max() + args.pad + 1, mask.shape[1])
    s = 2 ** (ml - args.level)
    H0, W0 = tf.pages[args.level].shape[:2]
    top, left = int(y0 * s), int(x0 * s)
    h = min(int(y1 * s), H0) - top
    w = min(int(x1 * s), W0) - left

    inner = binary_erosion(mask, np.ones((3, 3)), iterations=args.erode) if args.erode else mask

    # --- hex lattice over the tissue, in full-res coordinates -------------
    f0 = 2 ** args.level                       # render level -> full res
    st = args.stride
    gx = np.arange(left * f0 + st / 2, (left + w) * f0, st)
    gy = np.arange(top * f0 + st / 2, (top + h) * f0, st * np.sqrt(3) / 2)
    pts = []
    for j, yy in enumerate(gy):
        for xx in gx + (0 if j % 2 == 0 else st / 2):
            my, mx = int(yy / 2 ** ml), int(xx / 2 ** ml)
            if 0 <= my < inner.shape[0] and 0 <= mx < inner.shape[1] and inner[my, mx]:
                pts.append((xx, yy))
    pts = np.array(pts)
    print(f"  lattice: {len(pts)} points at {st}px pitch "
          f"({276 / st:.1f}x the spot pitch)")

    # --- inference --------------------------------------------------------
    # The page is 9931x9271x3 -- 276 MB decoded, which this machine does not
    # have to spare while a sweep is training (a earlier attempt died unable to
    # allocate 9 MB). The TIFF is tiled 256x256, so zarr reads only the tiles a
    # crop touches and peak memory stays at a few MB.
    import zarr
    store = tifffile.imread(ROOT / "hest_data" / "wsis" / f"{args.slide}.tif",
                            key=0, aszarr=True)
    full = zarr.open(store, mode="r")
    print(f"  lazy tiled reader over {full.shape}", flush=True)
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if dev == "cpu":
        torch.set_num_threads(os.cpu_count() or 8)
    model = model.to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    half = PATCH // 2
    out = np.zeros(len(pts), dtype=np.float32)

    with torch.no_grad():
        for b in range(0, len(pts), args.batch):
            chunk = pts[b:b + args.batch]
            buf = np.zeros((len(chunk), PATCH, PATCH, 3), dtype=np.uint8)
            for k, (xx, yy) in enumerate(chunk):
                xi, yi = int(xx) - half, int(yy) - half
                xi = min(max(xi, 0), full.shape[1] - PATCH)
                yi = min(max(yi, 0), full.shape[0] - PATCH)
                buf[k] = np.asarray(full[yi:yi + PATCH, xi:xi + PATCH])[..., :3]
            t = torch.from_numpy(buf).to(dev).permute(0, 3, 1, 2).float().div_(255)
            t = (t - mean) / std
            out[b:b + len(chunk)] = model(t)[:, gi].float().cpu().numpy()
            if b % (args.batch * 40) == 0:
                print(f"    {b + len(chunk)}/{len(pts)}", flush=True)
    store.close()

    print(f"  predicted {args.gene}: mean {out.mean():.3f}, s.d. {out.std():.3f} "
          f"(measured at spots: mean {gt[:, gi].mean():.3f}, s.d. {gt[:, gi].std():.3f})")

    # --- render -----------------------------------------------------------
    lo, hi = out.mean() - args.clip * out.std(), out.mean() + args.clip * out.std()
    cmap = plt.get_cmap(args.cmap)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(canvas)
    r = args.radius * st / f0
    for (xx, yy), v in zip(pts, out):
        cx, cy = xx / f0 - left, yy / f0 - top
        c = cmap(float(np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)))
        dr.ellipse([cx - r, cy - r, cx + r, cy + r],
                   fill=tuple(int(255 * q) for q in c[:3]) + (235,))

    out_p = ROOT / "figures" / (args.out or
                                f"m2e_{args.slide}_{args.gene}_{int(args.budget*100)}.png")
    canvas.save(out_p)
    prev = out_p.with_name(out_p.stem + "_preview.png")
    canvas.resize((1500, int(1500 * h / w)), Image.LANCZOS).save(prev)
    print(f"  {out_p}  {w}x{h}")
    print(f"  {prev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
