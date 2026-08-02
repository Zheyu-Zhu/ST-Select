"""Build a cache from the collaborator's `processed_gene/` package.

That package stores, per slide, an object-array .npy of spot dicts plus 224x224
PNG patches — a different layout from HEST (`patches/*.h5`), so this is the
adapter. One pass writes BOTH:
  * frozen backbone features (so the fast MLP path works), and
  * per-spot PNG paths (so end-to-end image training works from the same cache).

Two prediction targets are available (README of the package):
  --target hvg300 : `expression`, the HVG-300 **raw counts** -> we apply log1p.
                    This gene list is variance-ranked and, for HER2, leads with
                    immunoglobulin genes (IGKC, IGHG3, ...).
  --target norm   : `expression_norm`, ~907-967 genes already library-size
                    normalized + log1p (intersected with a single-cell reference,
                    alphabetical order).

Output: feature_cache/pg_<dataset>_<target>.npz with
    features (N,D) | expressions (N,G) | positions (N,2) | patient_ids (N,)
    sample_ids (N,) | gene_names (G,) | patch_paths (N,)

Usage:
  python scripts/prepare_processed_gene.py --dataset HER2 --target hvg300 \
      --feature-extractor densenet121 --device cuda
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import get_feature_extractor  # noqa: E402

PG = ROOT / "processed_gene"
SPOT_DIR = PG / "dataset_log1p_hvg300"
PATCH_DIR = PG / "patches"
GENE_DIR = PG / "gene_names"
CACHE_DIR = ROOT / "feature_cache"

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_spots(dataset: str):
    """All spots of a dataset, in slide order. Returns (spots, slide_ids)."""
    files = sorted(glob.glob(str(SPOT_DIR / dataset / "*.npy")))
    if not files:
        raise SystemExit(f"No .npy under {SPOT_DIR / dataset}")
    spots, slides = [], []
    for f in files:
        arr = np.load(f, allow_pickle=True)
        spots.extend(arr.tolist())
        slides.append(Path(f).stem)
    return spots, slides


def main():
    ap = argparse.ArgumentParser(description="Cache builder for processed_gene/")
    ap.add_argument("--dataset", required=True, choices=["HER2", "BC", "Kidney"])
    ap.add_argument("--target", default="hvg300", choices=["hvg300", "norm"],
                    help="hvg300 = log1p(HVG-300 raw counts); norm = expression_norm as-is")
    ap.add_argument("--feature-extractor", default="densenet121",
                    help="densenet121 / uni / none (none = image-only cache)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    spots, slides = load_spots(args.dataset)
    print(f"{args.dataset}: {len(slides)} slides, {len(spots)} spots")

    # ---- labels ----------------------------------------------------------
    if args.target == "hvg300":
        # Raw counts -> log1p (the package keeps `expression` un-transformed).
        expr = np.log1p(np.stack([s["expression"] for s in spots]).astype("float32"))
        genes = np.load(GENE_DIR / f"{args.dataset}_hvg300_genes.npy", allow_pickle=True)
    else:
        expr = np.stack([s["expression_norm"] for s in spots]).astype("float32")
        genes = np.load(GENE_DIR / f"{args.dataset}_expression_norm_genes.npy", allow_pickle=True)
    if expr.shape[1] != len(genes):
        raise SystemExit(f"gene count mismatch: expr {expr.shape[1]} vs names {len(genes)}")

    positions = np.stack([np.asarray(s["position"], dtype="float32") for s in spots])
    patient_ids = np.array([str(s["patient_id"]) for s in spots])
    sample_ids = np.array([str(s["sample_id"]) for s in spots])
    patch_paths = np.array([str(PATCH_DIR / s["sample_id"] / f"{s['patch_id']}.png")
                            for s in spots])

    missing = [p for p in patch_paths[:50] if not Path(p).exists()]
    if missing:
        raise SystemExit(f"missing patches, e.g. {missing[:3]}")

    print(f"  target={args.target}  expressions {expr.shape}  "
          f"patients={len(set(patient_ids))}  genes[:5]={list(genes[:5])}")

    # ---- frozen features -------------------------------------------------
    features = None
    if args.feature_extractor != "none":
        extractor = get_feature_extractor(args.feature_extractor, device=args.device)
        feats = []
        bs = args.batch_size
        for i in range(0, len(patch_paths), bs):
            batch_paths = patch_paths[i:i + bs]
            imgs = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in batch_paths])
            x = torch.from_numpy(imgs).float().permute(0, 3, 1, 2) / 255.0
            x = (x - _MEAN) / _STD
            feats.append(extractor.extract_batch(x))
            if (i // bs) % 20 == 0:
                print(f"    features {i + len(batch_paths)}/{len(patch_paths)}", flush=True)
        features = np.concatenate(feats).astype("float32")
        print(f"  features {features.shape}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Suffix non-default backbones so a UNI cache never overwrites a DenseNet one.
    suffix = "" if args.feature_extractor in ("densenet121", "none") else f"_{args.feature_extractor}"
    out = CACHE_DIR / f"pg_{args.dataset.lower()}_{args.target}{suffix}.npz"
    payload = dict(
        expressions=expr, positions=positions, patient_ids=patient_ids,
        sample_ids=sample_ids, gene_names=genes, patch_paths=patch_paths,
    )
    if features is not None:
        payload["features"] = features
    np.savez(out, **payload)
    print(f"\nSaved {out}")
    print(f"  {len(spots)} spots, {expr.shape[1]} genes, "
          f"{len(set(sample_ids))} slides, {len(set(patient_ids))} patients")


if __name__ == "__main__":
    main()
