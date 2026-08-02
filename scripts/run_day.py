"""Daytime driver: re-report everything under the PI's slide-level convention +
the new clustering/diversity metrics, and run the frozen-vs-trained AL comparison.

Priority queue (cheap/high-value first, longest last), each job resumable
(skips if results.json exists) and failure-isolated (logged, queue continues):

  1. norm_test        frozen ladder (DenseNet/UNI x cp10k/raw), full-supervision
                      -> slide-level top-N + ARI/NMI for the "reaching 0.2" story
  2. frozen_cscc_al   frozen DenseNet features + AL sweep on cSCC (cscc_dn)
  3. her2_al          frozen UNI features + AL sweep on HER2 (3 seeds, strong result)
  4. phase3_trained   trained DenseNet end-to-end + AL sweep on cSCC (cscc_img)

Jobs 2 vs 4 are the clean frozen-vs-trained-backbone AL isolation (same dataset
= cSCC, same backbone family = DenseNet). All outputs land under the gitignored
results_slide/ tree. New metrics (pcc_topN_by_slide, ari, nmi, selection_diversity)
are computed automatically by the evaluation path.

Run:  PATCH_CACHE_GB=5 conda run -n st-select python -u scripts/run_day.py
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

OUT = ROOT / "results_slide"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "RUN_DAY.md"

AL8 = ["random", "entropy", "tod", "coreset", "typiclust", "badge",
       "poisson_disk", "spatial_stratified"]
AL6 = ["random", "entropy", "coreset", "badge", "poisson_disk", "spatial_stratified"]


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def base(**kw) -> dict:
    d = dict(
        feature_cache_dir="./feature_cache", split_level="patient",
        n_folds=4, n_seeds=1, lr=1e-4, loss_fn="mse",
        pcc_top_n=[10, 50, 100, 200, 300], device="cuda", seed=42, num_workers=0,
    )
    d.update(kw)
    return d


# (label, cache/dataset name, is_image_mode, ExperimentConfig kwargs)
JOBS = []

# 1) norm_test frozen ladder — full-supervision only, adds slide-level + clustering.
for ds in ["her2_dn_cp10k", "her2_dn_raw", "her2_uni_cp10k", "her2_uni_raw"]:
    JOBS.append(("norm_test", ds, False, base(
        dataset_name=ds, model_name="feature_predictor",
        al_methods=["full_supervision"], add_full_supervision=False,
        full_epochs=60, epochs_per_round=60, budget_ratios=[1.0],
        batch_size=128, output_dir=str(OUT / "norm_test"),
    )))

# 2) frozen DenseNet AL on cSCC (paired with job 4).
JOBS.append(("frozen_cscc_al", "cscc_dn", False, base(
    dataset_name="cscc_dn", model_name="feature_predictor", al_methods=AL6,
    n_seeds=1, budget_ratios=[0.1, 0.3, 0.5, 1.0],
    epochs_per_round=50, full_epochs=60, initial_budget=256, budget_per_round=512,
    batch_size=128, output_dir=str(OUT / "frozen_cscc_al"),
)))

# 3) frozen UNI AL benchmark on HER2 (3 seeds — the strong, low-variance result).
JOBS.append(("her2_al", "her2_uni_cp10k", False, base(
    dataset_name="her2_uni_cp10k", model_name="feature_predictor", al_methods=AL8,
    n_seeds=3, budget_ratios=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    epochs_per_round=50, full_epochs=60, initial_budget=256, budget_per_round=512,
    batch_size=128, output_dir=str(OUT / "her2_al"),
)))

# 4) trained DenseNet end-to-end AL on cSCC (the missing Phase 3 experiment).
JOBS.append(("phase3_trained", "cscc_img", True, base(
    dataset_name="cscc_img", model_name="st_net", image_mode=True,
    patches_dir="./hest_data/patches", al_methods=AL6, n_seeds=1,
    budget_ratios=[0.1, 0.3, 1.0], epochs_per_round=15, full_epochs=15,
    initial_budget=256, budget_per_round=1024, batch_size=32,
    output_dir=str(OUT / "phase3_trained"),
)))


def summarize(res: dict) -> str:
    lines = []
    mg = res.get("mean_pcc_per_gene", {})
    ms = res.get("mean_pcc_per_gene_by_slide", {})
    ma = res.get("mean_ari", {})
    t50 = res.get("pcc_topN_by_slide", {})
    for m in mg:
        s50 = (t50.get(m) or {}).get("50")
        lines.append(f"      {m:20s} per_gene={mg[m]:.4f} "
                     f"per_slide={ms.get(m) if ms.get(m) is not None else float('nan'):.4f} "
                     f"top50_slide={s50 if s50 is not None else float('nan'):.4f} "
                     f"ari={ma.get(m) if ma.get(m) is not None else float('nan'):.4f}")
    return "\n".join(lines)


def run_job(label, name, image_mode, cfg_kwargs) -> None:
    done = Path(cfg_kwargs["output_dir"]) / name / "results.json"
    if done.exists():
        log(f"### SKIP  [{label}] {name} — exists")
        return
    if image_mode:
        clear_slide_cache()
    log(f"### START [{label}] {name}  methods={cfg_kwargs['al_methods']} "
        f"seeds={cfg_kwargs['n_seeds']} epochs={cfg_kwargs.get('epochs_per_round')}")
    t0 = time.time()
    try:
        res = run_full_benchmark(ExperimentConfig(**cfg_kwargs))
        dt = (time.time() - t0) / 60
        log(f"### DONE  [{label}] {name} in {dt:.1f} min\n{summarize(res)}")
    except Exception:
        dt = (time.time() - t0) / 60
        log(f"### FAILED [{label}] {name} after {dt:.1f} min\n{traceback.format_exc()}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    log("=" * 70)
    log(f"RUN-DAY: slide-level re-report + frozen-vs-trained AL — {datetime.now():%Y-%m-%d %H:%M}")
    log(f"cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    log("=" * 70)
    for label, name, image_mode, cfg_kwargs in JOBS:
        run_job(label, name, image_mode, cfg_kwargs)
    log("ALL DAY JOBS COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
