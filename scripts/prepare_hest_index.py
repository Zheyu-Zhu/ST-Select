"""Build a lightweight *index* cache for end-to-end (image-input) training.

Mirrors prepare_hest_cache.py exactly for the parts that matter (joint HVG
selection + barcode-aligned expression), but does NOT extract frozen features.
Instead it records, per spot, which slide and which row of that slide's patch
h5 the image lives at (`patch_pos`), so PatchImageDataset can fetch raw pixels
lazily and a backbone can be fine-tuned end-to-end.

Output: <feature_cache_dir>/<dataset>_img.npz with arrays
    expressions  (N, top_n)  float32  log1p HVG expression (barcode-aligned)
    positions    (N, 2)      float32  level-0 (x, y)
    patient_ids  (N,)        str      patient/donor per spot
    sample_ids   (N,)        str      slide id per spot
    gene_names   (top_n,)    str      dispersion-ranked HVG names
    patch_pos    (N,)        int      row index into <slide>.h5 `img`

Usage:
  python scripts/prepare_hest_index.py --dataset her2_breast \
      --patient-map hest_data/HEST_v1_1_0.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.hvg_selection import HVGSelector  # noqa: E402
# Reuse the vetted barcode-join / discovery / patient-map helpers.
from scripts.prepare_hest_cache import (  # noqa: E402
    read_patches,
    discover_sample_ids,
    align_expression_to_patches,
    _load_patient_map,
    DATA,
    CACHE_DIR,
)


def main():
    parser = argparse.ArgumentParser(description="Build a HEST image-index cache.")
    parser.add_argument("--dataset", required=True, help="Output cache name, e.g. her2_breast")
    parser.add_argument("--sample-ids", nargs="+", default=None,
                        help="Slide ids; if omitted, discovered from hest_data/st/*.h5ad")
    parser.add_argument("--n-hvgs", type=int, default=1000, help="HVG pool size")
    parser.add_argument("--top-n", type=int, default=300, help="Ranked HVG columns to keep")
    parser.add_argument("--patient-map", default=None,
                        help="slide->patient CSV (cols id,patient) or .json; default one-per-slide")
    args = parser.parse_args()

    sample_ids = args.sample_ids or discover_sample_ids()
    if not sample_ids:
        raise SystemExit(f"No sample ids given and none discovered under {DATA/'st'}.")
    print(f"Dataset '{args.dataset}': {len(sample_ids)} slides -> {sample_ids}")

    patient_map = _load_patient_map(args.patient_map, sample_ids)
    n_patients = len({patient_map[s] for s in sample_ids})
    print(f"  patient grouping: {n_patients} distinct patients")

    h5ad_paths = [str(DATA / "st" / f"{sid}.h5ad") for sid in sample_ids]

    # Joint HVG selection over a large pool; keep ranked top-N columns.
    print(f"Selecting {args.n_hvgs} HVGs jointly, keeping ranked top-{args.top_n}...")
    selector = HVGSelector(n_top_genes=args.n_hvgs, flavor="seurat")
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

    all_expr, all_pos, all_patient, all_sample, all_patchpos = [], [], [], [], []
    for sid in sample_ids:
        _imgs, coords, patch_barcodes = read_patches(sid)
        rows, keep = align_expression_to_patches(patch_barcodes, expr_map[sid])
        patch_pos = np.where(keep)[0].astype("int64")  # rows into this slide's patch h5
        coords = coords[keep]
        expr = expr_map[sid]["expression"][rows]        # (n_kept, top_n)
        n = len(patch_pos)

        all_expr.append(expr)
        all_pos.append(coords.astype("float32"))
        all_patient.append(np.array([patient_map[sid]] * n))
        all_sample.append(np.array([sid] * n))
        all_patchpos.append(patch_pos)
        print(f"  {sid}: {n} spots kept -> expr {expr.shape}")

    expressions = np.concatenate(all_expr).astype("float32")
    positions = np.concatenate(all_pos).astype("float32")
    patient_ids = np.concatenate(all_patient)
    sample_ids_arr = np.concatenate(all_sample)
    patch_pos_arr = np.concatenate(all_patchpos)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{args.dataset}_img.npz"
    np.savez(
        out,
        expressions=expressions, positions=positions,
        patient_ids=patient_ids, sample_ids=sample_ids_arr,
        gene_names=gene_names, patch_pos=patch_pos_arr,
    )
    print(f"\nSaved index cache: {out}")
    print(f"  expressions {expressions.shape}, n_genes={expressions.shape[1]}, "
          f"{len(set(sample_ids_arr.tolist()))} slides, {n_patients} patients")


if __name__ == "__main__":
    main()
