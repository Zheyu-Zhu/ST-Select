"""Build cell-type-composition caches from processed_gene, reusing existing features.

The pitch's Evaluation Task #4 ("cell-type composition estimation") and the review's
metric #4 ("cell-type composition similarity") need a second prediction target:
the per-spot abundance of each deconvolved cell type. The package ships this as
`cell_type_counts` (67 types for HER2/BC, 88 for Kidney), which sums to `c_count`,
the number of cells in the spot.

Target = **proportions** (counts / c_count), i.e. composition rather than abundance,
matching what "composition estimation" means. Spots with c_count == 0 have no
defined composition; they are kept (as all-zero rows) so the spot set stays
identical to the gene-expression task, and are masked out at evaluation time via
the `valid_mask` array.

Features/positions/ids are copied from the already-built pg_<ds>_hvg300.npz, so no
backbone re-extraction is needed.

Usage:  python scripts/make_celltype_cache.py --dataset HER2 [--backbone uni]
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PG = ROOT / "processed_gene"
SPOT_DIR = PG / "dataset_log1p_hvg300"
CACHE = ROOT / "feature_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["HER2", "BC", "Kidney"])
    ap.add_argument("--backbone", default="densenet121", choices=["densenet121", "uni"])
    args = ap.parse_args()

    ds = args.dataset.lower()
    suffix = "" if args.backbone == "densenet121" else "_uni"
    src = CACHE / f"pg_{ds}_hvg300{suffix}.npz"
    if not src.exists():
        raise SystemExit(f"source cache missing: {src} (build it first)")
    d = np.load(src, allow_pickle=True)

    # Re-read the spot dicts in the same slide order the source cache used
    # (sorted glob), so rows line up with the cached features.
    files = sorted(glob.glob(str(SPOT_DIR / args.dataset / "*.npy")))
    counts, ccount = [], []
    for f in files:
        for s in np.load(f, allow_pickle=True):
            counts.append(s["cell_type_counts"])
            ccount.append(s["c_count"])
    counts = np.stack(counts).astype("float32")
    ccount = np.asarray(ccount, dtype="float32")

    if len(counts) != len(d["expressions"]):
        raise SystemExit(f"row mismatch: {len(counts)} spots vs cache {len(d['expressions'])}")
    # Sanity: the spot ordering must match — check ids align by construction.
    tot = counts.sum(axis=1)
    if not np.allclose(tot, ccount, atol=0.05):
        raise SystemExit("cell_type_counts do not sum to c_count — ordering/parse error")

    valid = ccount > 0
    props = np.zeros_like(counts)
    props[valid] = counts[valid] / tot[valid][:, None]

    ct_names = np.array([f"celltype_{i}" for i in range(counts.shape[1])])
    out = CACHE / f"pg_{ds}_celltype{suffix}.npz"
    payload = dict(
        expressions=props,               # the regression target: composition
        positions=d["positions"], patient_ids=d["patient_ids"],
        sample_ids=d["sample_ids"], gene_names=ct_names,
        patch_paths=d["patch_paths"], valid_mask=valid, c_count=ccount,
    )
    if "features" in d:
        payload["features"] = d["features"]
    np.savez(out, **payload)
    print(f"Saved {out}")
    print(f"  {len(props)} spots · {props.shape[1]} cell types · "
          f"{int(valid.sum())} with cells ({100*valid.mean():.1f}%) · "
          f"mean cells/spot {ccount[valid].mean():.1f}")


if __name__ == "__main__":
    main()
