"""Rebuild the benchmark caches with a depth-normalised target.

The problem this fixes. Every result in results_pg/ was trained against
`expressions` = log1p(raw HVG-300 counts), with no library-size normalisation.
Per-spot totals span 108 to 2471 counts, and that single scalar accounts for
46.3% of the within-slide per-gene variance. Measured directly: the mean
per-slide correlation between a gene and its spot's raw library size is +0.611,
and a predictor that gets *only* the per-spot depth right - predicting every gene
as (depth x the slide's mean profile) - scores per-slide PCC 0.640. No model in
this project reaches that. The task had quietly become "predict how much RNA is
in this spot", which is tissue density, which is directly visible in H&E and
needs very few labels. That is why 5% of the labels looked like enough.

What this script changes: the `expressions` array, nothing else. Frozen features
are a function of the image, not the label, so they carry over untouched - which
matters, because re-extracting them is the expensive part and this way the whole
rebuild is minutes of CPU rather than hours of GPU. Patch paths, positions,
patient and sample ids all carry over too.

Normalisation choice, with the residual depth correlation each one leaves:

    log1p(raw)                +0.611   what every existing result used
    CP10K / expr_norm sum     +0.349   expr_norm is NOT library-normalised
                                       despite the README - its per-spot sum
                                       correlates +0.963 with the raw total
    CP10K / c_count           +0.205
    CP10K / HVG-300 sum       +0.217   <- written as the primary target
    proportions, no log       +0.038   <- written as `_prop`, for reference

log1p(CP10K) is the standard scRNA-seq transform and the one ST prediction papers
use, so it is the defensible primary. Its residual +0.217 is not removable by
normalisation: log1p is concave, deeper spots suffer less dropout and so spread
across more genes, and that spread survives the rescaling. The proportion variant
removes essentially all of it but is compositional and heavily skewed, so it is
written alongside as a control rather than as the headline target.

The true library size is the full transcriptome, which this data does not carry -
only 300 HVG counts. Every normaliser here is therefore a proxy.

    python scripts/rebuild_cache_cp10k.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "feature_cache"

COHORTS = ["her2", "kidney", "bc"]
BACKBONES = ["", "_uni"]          # "" = DenseNet-121, "_uni" = UNI ViT-L


def depth_corr(Y: np.ndarray, depth: np.ndarray, sid: np.ndarray) -> float:
    """Mean over slides of the mean over genes of corr(gene, raw library size)."""
    out = []
    for s in np.unique(sid):
        m = sid == s
        if m.sum() < 10:
            continue
        r = depth[m] - depth[m].mean()
        X = Y[m] - Y[m].mean(0)
        den = np.sqrt((X ** 2).sum(0) * (r ** 2).sum())
        out.append(np.nanmean(np.where(den > 1e-9,
                                       (X * r[:, None]).sum(0) / np.maximum(den, 1e-9),
                                       np.nan)))
    return float(np.mean(out)) if out else float("nan")


def main() -> int:
    if not CACHE.exists():
        print(f"no cache dir at {CACHE}")
        return 1

    print(f"{'source':34s}{'spots':>8}{'depth: was':>12}{'cp10k':>9}{'prop':>8}")
    print("-" * 71)
    written = []
    for c in COHORTS:
        for bb in BACKBONES:
            src = CACHE / f"pg_{c}_hvg300{bb}.npz"
            if not src.exists():
                print(f"{src.name:34s}  [missing, skipped]")
                continue
            z = np.load(src, allow_pickle=True)
            E, sid = z["expressions"], z["sample_ids"]

            # The cache holds log1p(raw); invert to counts before rescaling.
            raw = np.expm1(E)
            if np.abs(np.log1p(raw) - E).max() > 1e-3:
                print(f"{src.name:34s}  [NOT log1p(raw), skipped]")
                continue
            tot = np.maximum(raw.sum(1, keepdims=True), 1e-9)
            cp10k = np.log1p(raw / tot * 1e4).astype("float32")
            prop = (raw / tot).astype("float32")

            depth = raw.sum(1)
            print(f"{src.name:34s}{len(E):>8}{depth_corr(E, depth, sid):>12.4f}"
                  f"{depth_corr(cp10k, depth, sid):>9.4f}{depth_corr(prop, depth, sid):>8.4f}")

            for tag, arr in (("cp10k", cp10k), ("prop", prop)):
                dst = CACHE / f"pg_{c}_{tag}{bb}.npz"
                payload = {k: z[k] for k in z.files}
                payload["expressions"] = arr
                np.savez(dst, **payload)
                written.append(dst)

    print(f"\nwrote {len(written)} caches:")
    for w in sorted(written):
        print(f"  {w.name:32s} {w.stat().st_size / 2**20:7.1f} MB")
    print("\nOriginals are untouched. To use one, pass its dataset name, e.g."
          "\n  --dataset pg_her2_cp10k        (DenseNet features)"
          "\n  --dataset pg_her2_cp10k_uni    (UNI features)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
