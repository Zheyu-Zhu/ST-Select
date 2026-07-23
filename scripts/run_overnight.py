"""Autonomous overnight driver for the END-TO-END (image-input) st_net benchmark.

Runs a *prioritized* queue so the most scientifically valuable results land
first and any cutoff (morning) still leaves a coherent set of outputs:

  Phase 1  Ceilings         full_supervision, all 3 datasets   (answers the PI's
                            "can end-to-end reach ~0.2 PCC?")   + saves models
  Phase 2  Random curve     random budget sweep, all 3         (does more data
                            datasets                            help, end-to-end?)
  Phase 3  Strategy compare  random + AL strategies, cscc only  (optional; does
                            (smallest), coarse budget           any strategy beat
                                                                random end-to-end?)

Every job is independent: exceptions are caught, logged with a traceback, and
the queue continues. After each job the human-readable RUN_LOG.md and the
machine-readable overnight_summary.json are flushed to disk, so partial progress
always survives. All models (full_supervision folds) are checkpointed under
results_e2e/models/.

Run:
  python scripts/run_overnight.py
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

OUT = ROOT / "results_e2e"
OUT.mkdir(parents=True, exist_ok=True)
LOG_MD = OUT / "RUN_LOG.md"
SUMMARY_JSON = OUT / "overnight_summary.json"
MODELS_DIR = OUT / "models"

DATASETS = ["cscc_img", "her2_breast_img", "he_breast_img"]  # smallest -> largest
# End-to-end image cache -> matching frozen-feature baseline dir (for comparison).
FROZEN_MAP = {
    "cscc_img": "cscc_uni",
    "her2_breast_img": "her2_breast_uni",
    "he_breast_img": "he_breast_uni",
}

_summary = {"started": datetime.now().isoformat(), "jobs": []}


def log(msg: str) -> None:
    """Append a line to RUN_LOG.md and echo to stdout (so the Monitor sees it)."""
    line = f"{datetime.now():%H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG_MD, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def frozen_baseline(cache_name: str) -> dict:
    """Pull the frozen-feature baseline means for side-by-side comparison."""
    fz = FROZEN_MAP.get(cache_name)
    p = ROOT / "results_final" / "patient" / fz / "results.json" if fz else None
    if not p or not p.exists():
        return {}
    d = json.loads(p.read_text())
    means = d.get("mean_pcc_per_gene", {})
    return {
        "full_supervision": means.get("full_supervision"),
        "random": means.get("random"),
    }


def base_config(**kw) -> ExperimentConfig:
    cfg = dict(
        model_name="st_net", image_mode=True, patches_dir="./hest_data/patches",
        feature_cache_dir="./feature_cache", split_level="patient",
        n_folds=4, n_seeds=1, batch_size=32, lr=1e-4, loss_fn="mse",
        pcc_top_n=[10, 50, 100, 200, 300], device="cuda", seed=42, num_workers=0,
    )
    cfg.update(kw)
    return ExperimentConfig(**cfg)


def summarize(res: dict) -> dict:
    """Compact per-method means across the PCC conventions we report."""
    out = {}
    detailed = res.get("detailed", {})
    for method, means in res.get("mean_pcc_per_gene", {}).items():
        by_slide = [
            r["pcc_per_gene_by_slide"][-1]
            for r in detailed.get(method, [])
            if r.get("pcc_per_gene_by_slide")
        ]
        per_spot = [
            r["pcc_per_spot"][-1]
            for r in detailed.get(method, [])
            if r.get("pcc_per_spot")
        ]
        out[method] = {
            "pcc_per_gene": means,
            "pcc_per_spot": float(np.mean(per_spot)) if per_spot else None,
            "pcc_per_slide": float(np.nanmean(by_slide)) if by_slide else None,
        }
    return out


def run_job(label: str, cache_name: str, cfg: ExperimentConfig) -> None:
    # Resume support: skip a job whose results.json already exists (e.g. cSCC/HER2
    # ceilings finished on a prior launch), so a relaunch jumps to what's left.
    done_path = Path(cfg.output_dir) / cfg.dataset_name / "results.json"
    if done_path.exists():
        log(f"### SKIP  [{label}] {cache_name} — results already exist at {done_path}")
        _summary["jobs"].append({"label": label, "dataset": cache_name, "status": "skipped"})
        SUMMARY_JSON.write_text(json.dumps(_summary, indent=2))
        return
    # Fresh slate for the slide cache so one job's slides don't linger into the next.
    clear_slide_cache()
    log(f"### START [{label}] {cache_name}  methods={cfg.al_methods} "
        f"folds={cfg.n_folds} full_epochs={cfg.full_epochs} "
        f"epochs/round={cfg.epochs_per_round} budgets={cfg.budget_ratios}")
    t0 = time.time()
    job = {"label": label, "dataset": cache_name, "config": cfg.to_dict()}
    try:
        res = run_full_benchmark(cfg)
        dt = time.time() - t0
        comp = summarize(res)
        job.update({"status": "ok", "minutes": round(dt / 60, 1), "summary": comp})
        log(f"### DONE  [{label}] {cache_name} in {dt/60:.1f} min")
        fb = frozen_baseline(cache_name)
        for m, s in comp.items():
            extra = ""
            if m in fb and fb[m] is not None:
                extra = f"  (frozen {m}: {fb[m]:.4f})"
            log(f"      {m:18s} per_gene={s['pcc_per_gene']:.4f} "
                f"per_spot={s['pcc_per_spot'] if s['pcc_per_spot'] is not None else float('nan'):.4f} "
                f"per_slide={s['pcc_per_slide'] if s['pcc_per_slide'] is not None else float('nan'):.4f}"
                f"{extra}")
    except Exception:
        dt = time.time() - t0
        tb = traceback.format_exc()
        job.update({"status": "FAILED", "minutes": round(dt / 60, 1), "error": tb})
        log(f"### FAILED [{label}] {cache_name} after {dt/60:.1f} min\n{tb}")
    finally:
        # Free VRAM between jobs so a leak in one job can't OOM the next.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _summary["jobs"].append(job)
        _summary["updated"] = datetime.now().isoformat()
        SUMMARY_JSON.write_text(json.dumps(_summary, indent=2))


def main():
    log("=" * 70)
    log(f"OVERNIGHT END-TO-END st_net BENCHMARK — {datetime.now():%Y-%m-%d %H:%M}")
    log(f"device cuda_available={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    log("=" * 70)

    # ---- Phase 1: ceilings (full_supervision), all datasets, save models ----
    # 40 epochs x 4 folds x batch 32 ~ 4.4h total (measured ~15.5s/epoch + 15s/fold,
    # scaling with spot count). This is the PI's key "can end-to-end reach ~0.2?".
    for ds in DATASETS:
        cfg = base_config(
            dataset_name=ds, al_methods=["full_supervision"],
            full_epochs=40, epochs_per_round=40,
            budget_ratios=[1.0],
            output_dir=str(OUT / "ceiling"),
            save_models_dir=str(MODELS_DIR),
        )
        run_job("P1-ceiling", ds, cfg)

    # ---- Phase 2: random budget curve (end-to-end data scaling) ----
    # Ceiling already computed in Phase 1, so skip the redundant full_supervision.
    for ds in DATASETS:
        cfg = base_config(
            dataset_name=ds, al_methods=["random"], add_full_supervision=False,
            full_epochs=25, epochs_per_round=25,
            budget_ratios=[0.2, 0.5, 1.0],
            initial_budget=256, budget_per_round=512,
            output_dir=str(OUT / "random_curve"),
        )
        run_job("P2-random", ds, cfg)

    # ---- Phase 3: strategy comparison (optional, cscc only, coarse) ----
    cfg = base_config(
        dataset_name="cscc_img", add_full_supervision=False,
        al_methods=["random", "entropy", "coreset", "badge",
                    "poisson_disk", "spatial_stratified"],
        full_epochs=20, epochs_per_round=20,
        budget_ratios=[0.1, 0.3, 1.0],
        initial_budget=256, budget_per_round=512,
        output_dir=str(OUT / "al_sweep"),
    )
    run_job("P3-strategies", "cscc_img", cfg)

    _summary["finished"] = datetime.now().isoformat()
    SUMMARY_JSON.write_text(json.dumps(_summary, indent=2))
    log("=" * 70)
    log("ALL QUEUED JOBS COMPLETE")
    ok = sum(1 for j in _summary["jobs"] if j.get("status") == "ok")
    log(f"jobs ok: {ok}/{len(_summary['jobs'])}")
    log("=" * 70)


if __name__ == "__main__":
    main()
