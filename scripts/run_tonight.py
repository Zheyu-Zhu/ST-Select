"""Overnight driver: generalize the raw-target + UNI end-to-end result to cSCC and He-breast.

Runs UNI (ViT-L) fine-tuned end-to-end with the log1p(RAW) target — the exact
config that gave HER2 top-50 = 0.212 — on the two datasets it has NOT yet been
run on. This is HANDOFF.md §12.1, the single most valuable next experiment:
right now "we reach ~0.2" is HER2-only, one seed.

cSCC (smallest) runs first so a result lands early even if the run is cut short.
Each job is resumable: if its results.json already exists it is skipped, so a
relaunch jumps to what's left. Slide image cache is cleared between jobs.

Prereqs (already built):
  feature_cache/cscc_raw_img.npz       (scripts/prepare_hest_index.py --target raw)
  feature_cache/he_breast_raw_img.npz

Run:
  PATCH_CACHE_GB=5 conda run -n st-select python -u scripts/run_tonight.py
"""

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.run_experiment import run_full_benchmark  # noqa: E402
from src.utils import ExperimentConfig  # noqa: E402
from src.datasets.patch_dataset import clear_slide_cache  # noqa: E402

OUT = ROOT / "results_e2e" / "ceiling_uni_raw"
OUT.mkdir(parents=True, exist_ok=True)
LOG_MD = ROOT / "results_e2e" / "RUN_LOG.md"

# smallest -> largest, so an early cutoff still leaves cSCC done.
DATASETS = ["cscc_raw_img", "he_breast_raw_img"]


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG_MD, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def cfg_for(ds: str) -> ExperimentConfig:
    # Exact HER2 headline config (results_e2e/ceiling_uni_raw/her2_breast_raw_img).
    return ExperimentConfig(
        dataset_name=ds, model_name="uni_regressor", image_mode=True,
        patches_dir="./hest_data/patches", feature_cache_dir="./feature_cache",
        al_methods=["full_supervision"], add_full_supervision=False,
        split_level="patient", n_folds=4, n_seeds=1,
        full_epochs=15, epochs_per_round=15, budget_ratios=[1.0],
        batch_size=16, lr=1e-4, backbone_lr_mult=0.1, loss_fn="mse",
        pcc_top_n=[10, 50, 100, 200, 300], device="cuda", seed=42,
        num_workers=0, output_dir=str(OUT),
    )


def run_job(ds: str) -> None:
    done = OUT / ds / "results.json"
    if done.exists():
        log(f"### SKIP  {ds} — results already exist at {done}")
        return
    clear_slide_cache()
    cfg = cfg_for(ds)
    log(f"### START {ds}  model=uni_regressor raw target, full_epochs=15 batch=16 bb-lr x0.1")
    t0 = time.time()
    try:
        res = run_full_benchmark(cfg)
        dt = (time.time() - t0) / 60
        m = res["mean_pcc_per_gene"].get("full_supervision")
        top = res.get("pcc_topN", {}).get("full_supervision", {})
        det = res.get("detailed", {}).get("full_supervision", [])
        bys = [r["pcc_per_gene_by_slide"][-1] for r in det if r.get("pcc_per_gene_by_slide")]
        per_slide = float(np.nanmean(bys)) if bys else float("nan")
        log(f"### DONE  {ds} in {dt:.1f} min  per_gene={m:.4f} "
            f"per_slide={per_slide:.4f} top10={top.get('10')} top50={top.get('50')}")
    except Exception:
        dt = (time.time() - t0) / 60
        log(f"### FAILED {ds} after {dt:.1f} min\n{traceback.format_exc()}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    log("=" * 70)
    log(f"RUN-TONIGHT: raw + UNI end-to-end on cSCC & He-breast — {datetime.now():%Y-%m-%d %H:%M}")
    log(f"device cuda_available={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    log("=" * 70)
    for ds in DATASETS:
        run_job(ds)
    log("ALL JOBS COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
