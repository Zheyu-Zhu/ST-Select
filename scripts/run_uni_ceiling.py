"""UNI e2e full-supervision ceiling for any pg cohort, with the full metric set.

The original UNI e2e ceilings (HER2, Kidney) were run with pcc_top_n=[10,50,100]
and saved no checkpoints, which is why their top-200 cells are empty and why
recovering that one number costs a full retrain. This script is the corrected
version: all five cutoffs, and weights persisted so any future metric is a
re-evaluation instead of hours of GPU.

    python scripts/run_uni_ceiling.py --dataset pg_her2_hvg300
"""
import argparse, sys, time, traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from src.run_experiment import run_full_benchmark  # noqa: E402
from src.utils import ExperimentConfig  # noqa: E402

OUT = ROOT / "results_pg" / "uni_ceiling"
LOG = OUT / "RUN_UNI_CEILING.md"


def log(m):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {m}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    if (OUT / a.dataset / "results.json").exists():
        log(f"### SKIP {a.dataset} — exists"); return
    cfg = ExperimentConfig(
        dataset_name=a.dataset, model_name="uni_regressor", image_mode=True,
        feature_cache_dir="./feature_cache", output_dir=str(OUT),
        al_methods=["full_supervision"], add_full_supervision=False,
        split_level="patient", n_folds=4, n_seeds=1,
        epochs_per_round=15, full_epochs=15, batch_size=16,
        backbone_lr_mult=0.1, lr=1e-4, loss_fn="mse", budget_ratios=[1.0],
        pcc_top_n=[10, 50, 100, 200, 300],
        save_models_dir=str(OUT / "models"),
        device="cuda", seed=42, num_workers=0,
    )
    log(f"### START ceiling {a.dataset} cutoffs={cfg.pcc_top_n}")
    t0 = time.time()
    try:
        r = run_full_benchmark(cfg)
        m = "full_supervision"
        tn = r.get("pcc_topN_by_slide", {}).get(m, {})
        log(f"### DONE {a.dataset} in {(time.time()-t0)/60:.1f} min  "
            f"per-slide={r['mean_pcc_per_gene_by_slide'][m]:.4f}")
        log("      top-N: " + "  ".join(f"top{k}={tn[k]:.3f}" for k in sorted(tn, key=int)))
    except Exception:
        log(f"### FAILED {a.dataset} after {(time.time()-t0)/60:.1f} min\n{traceback.format_exc()}")
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
