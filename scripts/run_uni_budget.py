"""Budget curve on the BEST model — UNI ViT-L fine-tuned end to end.

E18 established the labelling budget each cohort needs, but on frozen features.
The obvious objection is that the saturation point is a property of the weak
model, not the data: a 304 M-param backbone might keep extracting value from
labels long after a small MLP head has stopped. This runs the same 7-point
controlled curve on uni_regressor to settle it.

Random selection only — E19 showed no strategy beats it on this model, so the
budget question is the one worth the GPU time.

NOTE the checkpoints are warm-started (the loop resets only the head between
budgets, so the backbone carries over). That is the right regime for a budget
*curve* — each point is "what you get having labelled X%" — but it means points
are not independent from-scratch trainings, and the 100% point is not identical
to the from-scratch ceiling from run_uni_ceiling.py.

    python scripts/run_uni_budget.py --dataset pg_her2_hvg300
"""
import argparse, sys, time, traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from src.run_experiment import run_full_benchmark  # noqa: E402
from src.utils import ExperimentConfig  # noqa: E402

OUT = ROOT / "results_pg" / "uni_budget"
LOG = OUT / "RUN_UNI_BUDGET.md"
RATIOS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def log(m):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {m}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pg_her2_hvg300")
    a = ap.parse_args()
    if (OUT / a.dataset / "results.json").exists():
        log(f"### SKIP {a.dataset} — exists"); return
    cfg = ExperimentConfig(
        dataset_name=a.dataset, model_name="uni_regressor", image_mode=True,
        feature_cache_dir="./feature_cache", output_dir=str(OUT),
        al_methods=["random"], add_full_supervision=False,
        split_level="patient", n_folds=4, n_seeds=1,
        epochs_per_round=15, full_epochs=15, batch_size=16,
        backbone_lr_mult=0.1, lr=1e-4, loss_fn="mse",
        budget_ratios=RATIOS, pcc_top_n=[10, 50, 100, 200, 300],
        budget_per_round=512, initial_budget=256,
        device="cuda", seed=42, num_workers=0,
    )
    log(f"### START uni budget {a.dataset} ratios={RATIOS}")
    t0 = time.time()
    try:
        r = run_full_benchmark(cfg)
        runs = r["detailed"]["random"]
        log(f"### DONE {a.dataset} in {(time.time()-t0)/60:.1f} min")
        for i, fr in enumerate(RATIOS):
            v = np.mean([x["pcc_per_gene_by_slide"][i] for x in runs])
            log(f"      {fr:5.0%}  per-slide={v:.4f}")
    except Exception:
        log(f"### FAILED {a.dataset} after {(time.time()-t0)/60:.1f} min\n{traceback.format_exc()}")
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
