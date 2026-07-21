"""Build a feature cache from downloaded HEST samples using the package's own components.

For each sample:
  - patches/<id>.h5  : img (N,224,224,3) uint8 + coords (N,2) + barcode (N,1)
  - st/<id>.h5ad     : raw counts + obsm['spatial']

Steps (mirrors tutorial §1-§2.9.4):
  1. Joint HVG selection across all slides via the fixed HVGSelector (seurat),
     selecting a large pool (default 1000) and keeping the dispersion-ranked
     top-N (default 300) columns for reporting a top-N PCC sweep downstream.
  2. Align patch barcodes to h5ad spots BY BARCODE (not by position).
  3. Extract frozen features from patches (get_feature_extractor).
  4. Save <feature_cache_dir>/<dataset>.npz with
     features, expressions, positions, patient_ids, sample_ids, gene_names.

Usage:
  python scripts/prepare_hest_cache.py \
      --dataset ncbi_mouse --sample-ids NCBI326 NCBI331 NCBI333 NCBI335
  # or derive sample ids from the files present under hest_data/st/:
  python scripts/prepare_hest_cache.py --dataset her2_breast
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import scanpy as sc
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.hvg_selection import HVGSelector  # noqa: E402
from src.models import get_feature_extractor  # noqa: E402

DATA = ROOT / "hest_data"
CACHE_DIR = ROOT / "feature_cache"


def read_patches(sid):
    with h5py.File(DATA / "patches" / f"{sid}.h5", "r") as f:
        imgs = f["img"][:]                       # (N,224,224,3) uint8
        coords = f["coords"][:]                  # (N,2)
        barcodes = [b[0].decode() if isinstance(b[0], bytes) else str(b[0])
                    for b in f["barcode"][:]]
    return imgs, coords, list(barcodes)


def discover_sample_ids():
    """Sample ids = stems of every h5ad under hest_data/st/ that also has patches."""
    st_dir = DATA / "st"
    patch_dir = DATA / "patches"
    ids = []
    for p in sorted(st_dir.glob("*.h5ad")):
        if (patch_dir / f"{p.stem}.h5").exists():
            ids.append(p.stem)
    return ids


def align_expression_to_patches(patch_barcodes, slide_entry):
    """Join expression rows to patch rows BY BARCODE.

    Returns (row_index_into_expression, keep_mask_over_patches). Every patch
    barcode must be present in the slide's expression barcodes, otherwise we
    error loudly rather than silently truncating / mis-pairing (the old
    positional min(len) join could desync features and labels).
    """
    expr_barcodes = [str(b) for b in slide_entry["barcodes"]]
    bc_to_row = {bc: i for i, bc in enumerate(expr_barcodes)}

    rows, keep = [], []
    missing = []
    for bc in patch_barcodes:
        r = bc_to_row.get(str(bc))
        if r is None:
            missing.append(bc)
            keep.append(False)
        else:
            rows.append(r)
            keep.append(True)

    if missing:
        raise ValueError(
            f"{len(missing)}/{len(patch_barcodes)} patch barcodes have no matching "
            f"expression row (e.g. {missing[:3]}). Refusing to build a mis-aligned "
            f"cache. Check that patch and h5ad barcodes share the same format."
        )
    return np.asarray(rows, dtype=int), np.asarray(keep, dtype=bool)


def _load_patient_map(path, sample_ids):
    """slide id -> patient id. From a HEST metadata CSV (columns `id`,`patient`)
    or a JSON dict; falls back to one-patient-per-slide. Any slide absent from
    the map keeps its own id as patient (safe, no accidental merging)."""
    mapping = {sid: sid for sid in sample_ids}
    if not path:
        return mapping
    p = Path(path)
    if p.suffix == ".json":
        import json
        mapping.update({k: str(v) for k, v in json.loads(p.read_text()).items()})
    elif p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
        if not {"id", "patient"} <= set(df.columns):
            raise ValueError(f"{path} must have 'id' and 'patient' columns; got {list(df.columns)}")
        csv_map = dict(zip(df["id"].astype(str), df["patient"].astype(str)))
        for sid in sample_ids:
            if sid in csv_map:
                mapping[sid] = csv_map[sid]
    else:
        raise ValueError(f"Unsupported --patient-map format: {path} (use .csv or .json)")
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Build a HEST feature cache.")
    parser.add_argument("--dataset", required=True, help="Output cache name, e.g. her2_breast")
    parser.add_argument(
        "--sample-ids", nargs="+", default=None,
        help="Slide ids to include; if omitted, discovered from hest_data/st/*.h5ad",
    )
    parser.add_argument("--n-hvgs", type=int, default=1000, help="HVG pool size to select")
    parser.add_argument(
        "--top-n", type=int, default=300,
        help="Dispersion-ranked HVG columns to keep (supports a top-N PCC sweep up to this)",
    )
    parser.add_argument("--feature-extractor", default="densenet121")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--patient-map", default=None,
        help="slide id -> patient id mapping. Either a HEST metadata CSV "
             "(uses its `id` and `patient` columns) or a .json dict. "
             "Default: one patient per slide.",
    )
    args = parser.parse_args()

    sample_ids = args.sample_ids or discover_sample_ids()
    if not sample_ids:
        raise SystemExit(
            f"No sample ids given and none discovered under {DATA/'st'}. "
            f"Pass --sample-ids explicitly."
        )
    print(f"Dataset '{args.dataset}': {len(sample_ids)} slides -> {sample_ids}")

    patient_map = _load_patient_map(args.patient_map, sample_ids)
    n_patients = len({patient_map[s] for s in sample_ids})
    print(f"  patient grouping: {n_patients} distinct patients "
          f"({'from ' + args.patient_map if args.patient_map else 'one-per-slide default'})")

    h5ad_paths = [str(DATA / "st" / f"{sid}.h5ad") for sid in sample_ids]

    # 1) Joint HVG selection over a large pool; keep ranked top-N columns.
    print(f"Selecting {args.n_hvgs} HVGs jointly, keeping ranked top-{args.top_n}...")
    selector = HVGSelector(n_top_genes=args.n_hvgs, flavor="seurat")
    # h5ad var names may not be unique; make them unique first (shared, per-slide).
    tmp_dir = DATA / "_uniq"
    tmp_dir.mkdir(exist_ok=True)
    selector_paths = []
    for sid, p in zip(sample_ids, h5ad_paths):
        a = sc.read_h5ad(p)
        a.var_names_make_unique()
        up = tmp_dir / f"{sid}.h5ad"
        a.write_h5ad(up)
        selector_paths.append(str(up))
    selector.fit(selector_paths, slide_ids=sample_ids)
    expr_map = selector.extract_expression_matrices(sample_ids, top_n=args.top_n)
    gene_names = np.asarray(selector.get_hvg_names(args.top_n))
    print("  top HVGs:", list(gene_names[:5]), "...")

    extractor = get_feature_extractor(args.feature_extractor, device=args.device)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    all_feats, all_expr, all_pos, all_patient, all_sample = [], [], [], [], []

    for sid in sample_ids:
        imgs, coords, patch_barcodes = read_patches(sid)

        # Barcode join: expression row per patch (order follows patch order).
        rows, keep = align_expression_to_patches(patch_barcodes, expr_map[sid])
        imgs = imgs[keep]
        coords = coords[keep]
        expr = expr_map[sid]["expression"][rows]     # (n_kept, top_n), aligned by barcode
        n = len(imgs)

        # Normalize patches to ImageNet stats and extract features in batches.
        x = torch.from_numpy(imgs).float().permute(0, 3, 1, 2) / 255.0
        x = (x - imagenet_mean) / imagenet_std

        feats = []
        for i in range(0, len(x), args.batch_size):
            feats.append(extractor.extract_batch(x[i:i + args.batch_size]))
        feats = np.concatenate(feats, axis=0)        # (n, D)

        all_feats.append(feats)
        all_expr.append(expr)
        all_pos.append(coords.astype("float32"))
        all_patient.append(np.array([patient_map[sid]] * n))  # patient-level grouping
        all_sample.append(np.array([sid] * n))
        print(f"  {sid}: {n} spots -> features {feats.shape}, expr {expr.shape}")

    features = np.concatenate(all_feats).astype("float32")
    expressions = np.concatenate(all_expr).astype("float32")
    positions = np.concatenate(all_pos).astype("float32")
    patient_ids = np.concatenate(all_patient)
    sample_ids_arr = np.concatenate(all_sample)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{args.dataset}.npz"
    np.savez(
        out,
        features=features, expressions=expressions, positions=positions,
        patient_ids=patient_ids, sample_ids=sample_ids_arr, gene_names=gene_names,
    )
    print(f"\nSaved cache: {out}")
    print(f"  features {features.shape}, expressions {expressions.shape}, "
          f"feature_dim={features.shape[1]}, n_genes={expressions.shape[1]}")
    print(f"  slides: {sorted(set(sample_ids_arr.tolist()))}")


if __name__ == "__main__":
    main()
