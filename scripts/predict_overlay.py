"""A real model prediction, rendered as a transparent layer over the tissue.

Not a mock field. The checkpoint is the UNI ViT-L end-to-end ceiling model from
results_pg/uni_ceiling, and the slide is chosen so the model never saw it: on
pg_her2_hvg300 the fold-0 split holds out patients B and F, and SPA125 belongs
to patient F, so its predictions here are genuine held-out output.

Two maps are written and they should be read together:

    pred   what the model produces from the H&E patch alone
    true   what was measured at the same spots

Side by side they show what the number in the results table means. A per-slide
PCC near 0.3 does not look like a good match, and the pair makes that concrete
in a way the scalar does not.

Geometry comes from the same contour mask and crop that scripts/clean_wsi.py
uses, so the output registers pixel-for-pixel on wsi_<slide>_clean_rgba.png.

    python scripts/predict_overlay.py --slide SPA125 --gene ERBB2
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def contour_crop(slide, level, mask_level=3, pad=8):
    """The crop clean_wsi.py produces, as (left, top, w, h) at `level`."""
    import tifffile
    tf = tifffile.TiffFile(ROOT / "hest_data" / "wsis" / f"{slide}.tif")
    ml = min(mask_level, len(tf.pages) - 1)
    shape = tf.pages[ml].shape[:2]
    g = ROOT / "hest_data" / "tissue_seg" / f"{slide}_contours.geojson"
    polys = []
    for f in json.load(open(g))["features"]:
        gm = f["geometry"]
        rings = [gm["coordinates"][0]] if gm["type"] == "Polygon" \
            else [q[0] for q in gm["coordinates"]]
        polys += [np.asarray(r, dtype=float) for r in rings]
    im = Image.new("L", (shape[1], shape[0]), 0)
    dr = ImageDraw.Draw(im)
    for r in polys:
        dr.polygon([tuple(v) for v in (r / 2.0 ** ml)], fill=255)
    m = np.asarray(im) > 127
    ys, xs = np.where(m)
    y0, y1 = max(0, ys.min() - pad), ys.max() + pad
    x0, x1 = max(0, xs.min() - pad), xs.max() + pad
    s = 2 ** (ml - level)
    H0, W0 = tf.pages[level].shape[:2]
    top, left = int(y0 * s), int(x0 * s)
    return left, top, min(int(x1 * s), W0) - left, min(int(y1 * s), H0) - top


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", default="SPA125")
    ap.add_argument("--gene", default="ERBB2")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--radius", type=float, default=0.46)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cmap", default="RdYlBu_r")
    ap.add_argument("--mode", choices=["gene", "domains"], default="gene",
                    help="'gene' colours one gene's value; 'domains' clusters the "
                         "full 300-dim vector per spot and colours by cluster, "
                         "which is what produces contiguous regions. A single "
                         "gene is one noisy channel and renders as speckle; the "
                         "domains are the same quantity the ARI/NMI metric "
                         "scores, so the picture matches what is measured.")
    ap.add_argument("--k", type=int, default=6, help="clusters for --mode domains")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from src.models import UNIRegressor

    cache = np.load(ROOT / "feature_cache" / "pg_her2_hvg300.npz", allow_pickle=True)
    sel = cache["sample_ids"] == args.slide
    if not sel.any():
        print(f"{args.slide} not in the cache"); return 1
    genes = [str(g) for g in cache["gene_names"]]
    if args.gene not in genes:
        print(f"{args.gene} not among the 300 targets"); return 1
    gi = genes.index(args.gene)
    paths = [str(p) for p in cache["patch_paths"][sel]]
    truth = cache["expressions"][sel][:, gi]

    ckpt = ROOT / "results_pg" / "uni_ceiling" / "models" / "pg_her2_hvg300" / \
        f"full_supervision_fold{args.fold}_seed42.pt"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    held = str(cache["patient_ids"][sel][0])
    print(f"{args.slide}: {sel.sum()} spots, patient {held}, gene {args.gene}")
    print(f"  checkpoint fold {ck['fold']} of {ck['dataset_name']} ({ck['method']})")

    dev = args.device
    if dev == "auto":
        free = 0
        if torch.cuda.is_available():
            free = (torch.cuda.mem_get_info()[0]) / 2**30
        dev = "cuda" if free > 4.0 else "cpu"
        print(f"  {free:.1f} GB free on the GPU -> running on {dev}")

    model = UNIRegressor(n_genes=300, frozen_backbone=False, grad_checkpointing=False)
    model.load_state_dict(ck["state_dict"], assign=True)
    del ck
    gc.collect()
    model = model.to(dev).eval()
    print(f"  weights loaded, running inference on {dev}", flush=True)

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
    out = []
    with torch.no_grad():
        for s in range(0, len(paths), args.batch):
            imgs = [np.asarray(Image.open(p).convert("RGB").resize((224, 224)))
                    for p in paths[s:s + args.batch]]
            x = torch.from_numpy(np.stack(imgs)).to(dev)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            out.append(model((x - MEAN) / STD).cpu().numpy())
            if (s // args.batch) % 10 == 0:
                print(f"    {s + len(imgs)}/{len(paths)}", flush=True)
    P = np.concatenate(out)                      # (n_spots, 300)
    T = cache["expressions"][sel]
    np.savez(ROOT / "figures" / f"_pred_{args.slide}_fold{args.fold}.npz",
             pred=P, true=T, genes=np.array(genes))
    pred, truth = P[:, gi], T[:, gi]
    print(f"  {args.gene} on this slide: PCC(pred, measured) = "
          f"{float(np.corrcoef(pred, truth)[0, 1]):+.4f}")
    pg = np.array([np.corrcoef(P[:, j], T[:, j])[0, 1] if T[:, j].std() > 1e-9 else np.nan
                   for j in range(P.shape[1])])
    print(f"  across all 300 genes: median {np.nanmedian(pg):+.4f}, "
          f"best {np.nanmax(pg):+.4f} ({genes[int(np.nanargmax(pg))]}), "
          f"mean {np.nanmean(pg):+.4f}")

    xy = cache["positions"][sel].astype(float)
    import anndata
    h5 = ROOT / "hest_data" / "st" / f"{args.slide}.h5ad"
    xy = np.asarray(anndata.read_h5ad(h5).obsm["spatial"], dtype=float)
    left, top, w, h = contour_crop(args.slide, args.level)
    f = 2 ** args.level
    px, py = xy[:, 0] / f - left, xy[:, 1] / f - top

    d = np.linalg.norm(np.c_[px, py][:300, None, :] - np.c_[px, py][None, :300, :], axis=2)
    np.fill_diagonal(d, 1e9)
    rad = args.radius * float(np.median(d.min(1)))

    if args.mode == "domains":
        # Cluster both sides with one model fitted on the measured data, so the
        # cluster ids mean the same thing in the two pictures. Fitting each
        # separately would give the same regions different colours and the
        # comparison would be unreadable.
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(T)
        km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(sc.transform(T))
        labs = {"true": km.labels_, "pred": km.predict(sc.transform(P))}
        pal = [tuple(int(255 * t) for t in plt.get_cmap("Spectral")(i / (args.k - 1))[:3]) + (255,)
               for i in range(args.k)]
        agree = float((labs["true"] == labs["pred"]).mean())
        print(f"  {args.k} domains; predicted domain matches measured at "
              f"{agree:.0%} of spots")
        series = [(t, [pal[i] for i in labs[t]]) for t in ("true", "pred")]
    else:
        lo, hi = float(min(pred.min(), truth.min())), float(max(pred.max(), truth.max()))
        cmap = plt.get_cmap(args.cmap)
        series = [(t, [tuple(int(255 * q) for q in cmap(float((x - lo) / max(hi - lo, 1e-9)))[:3]) + (255,)
                       for x in v]) for t, v in (("true", truth), ("pred", pred))]

    for tag, colours in series:
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dr = ImageDraw.Draw(canvas)
        for x, y, c in zip(px, py, colours):
            dr.ellipse([x - rad, y - rad, x + rad, y + rad], fill=c,
                       outline=(42, 52, 65, 255), width=max(int(rad * 0.16), 1))
        name = f"expr_{args.slide}_{args.gene}_{tag}" if args.mode == "gene"             else f"domains_{args.slide}_k{args.k}_{tag}"
        o = ROOT / "figures" / f"{name}.png"
        canvas.save(o)
        print(f"  {o.name}  {w}x{h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
