"""Final run: strategy breadth (CPU) alongside UNI fine-tuning (GPU), in parallel.

The two remaining pieces use disjoint resources, so they run as separate processes:

  --lane cpu   20-strategy AL sweep on the NEW (learnable) data.
               Closes the last reviewer objection — "you only tried 8 of 29
               strategies" — and does so where the task demonstrably works
               (per-slide 0.23-0.26) rather than on the near-unlearnable old data.
               Cost is the O(N^2) acquisition, not the MLP: GPU stays ~idle.

  --lane gpu   UNI fine-tuned end-to-end, the one empty cell in the
               backbone x training-regime matrix (we have DenseNet frozen/e2e and
               UNI frozen, never UNI e2e on this data). Frozen UNI is already the
               best model at 0.425; a 1-epoch probe scored 0.449 on 2 folds, so
               this is the most plausible route to a higher ceiling.

Excluded from the 20: batchbald and cdal — O(N^2)-with-a-greedy-loop, measured at
20.8 s and 6.7 s per acquisition at N=2000, which caused a 5 h hang before. The
remaining 20 still span all six families.

Run both:
  conda run -n st-select python -u scripts/run_final.py --lane cpu
  conda run -n st-select python -u scripts/run_final.py --lane gpu
"""

import argparse
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

OUT = ROOT / "results_pg" / "final"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "feature_cache"
LOG = OUT / "RUN_FINAL.md"

AL20 = [
    "random", "entropy", "margin", "least_confidence", "mc_dropout",
    "tod", "coreset", "typiclust", "maxherding", "probcover", "badge",
    "similar", "alfa_mix", "cald", "ceal", "coregcn", "lal_rl",
    "poisson_disk", "spatial_stratified", "tissue_aware",
]


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def base(**kw) -> dict:
    d = dict(feature_cache_dir="./feature_cache", split_level="patient",
             n_folds=4, n_seeds=1, lr=1e-4, loss_fn="mse",
             pcc_top_n=[10, 50, 100], device="cuda", seed=42, num_workers=0)
    d.update(kw)
    return d


def summarize(res: dict) -> str:
    mg = res.get("mean_pcc_per_gene", {})
    ms = res.get("mean_pcc_per_gene_by_slide", {})
    sp = res.get("spatial_autocorrelation", {})
    al = {m: v for m, v in ms.items() if m != "full_supervision" and v is not None}
    lines = []
    if len(al) > 1 and "random" in al:
        rnd = al["random"]
        ranked = sorted((m for m in al if m != "random"), key=lambda m: -al[m])
        best = ranked[0]
        lines.append(f"      [AL] random={rnd:.4f}  best={best}={al[best]:.4f}  "
                     f"delta={al[best]-rnd:+.4f}  (n_methods={len(al)})")
        lines.append("      top5: " + ", ".join(f"{m}={al[m]:.4f}" for m in ranked[:5]))
    for m in mg:
        if len(mg) > 3 and m != "full_supervision":
            continue                      # keep the AL log short
        s = ms.get(m)
        row = (f"      {m:20s} per_gene={mg[m]:.4f} "
               f"per_slide={s if s is not None else float('nan'):.4f}")
        if m in sp:
            row += (f" | moran pred={sp[m].get('moran_pred', float('nan')):.3f} "
                    f"fid={sp[m].get('moran_fidelity', float('nan')):.3f}")
        lines.append(row)
    return "\n".join(lines)


def run_job(label: str, cfg: dict) -> None:
    name = cfg["dataset_name"]
    done = Path(cfg["output_dir"]) / name / "results.json"
    if done.exists():
        log(f"### SKIP  [{label}] {name} — exists")
        return
    if not (CACHE / f"{name}.npz").exists():
        log(f"### SKIP  [{label}] {name} — cache missing")
        return
    log(f"### START [{label}] {name} methods={len(cfg['al_methods'])} "
        f"model={cfg['model_name']} image={cfg.get('image_mode', False)}")
    t0 = time.time()
    try:
        res = run_full_benchmark(ExperimentConfig(**cfg))
        log(f"### DONE  [{label}] {name} in {(time.time()-t0)/60:.1f} min\n{summarize(res)}")
    except Exception:
        log(f"### FAILED [{label}] {name} after {(time.time()-t0)/60:.1f} min\n"
            f"{traceback.format_exc()}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def al20(ds):
    return base(dataset_name=f"pg_{ds}_hvg300", model_name="feature_predictor",
                al_methods=AL20, budget_ratios=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
                epochs_per_round=50, full_epochs=60, initial_budget=256,
                budget_per_round=512, batch_size=128,
                output_dir=str(OUT / "al20"))


def uni_e2e(ds):
    return base(dataset_name=f"pg_{ds}_hvg300", model_name="uni_regressor",
                image_mode=True, al_methods=["full_supervision"],
                add_full_supervision=False, full_epochs=15, epochs_per_round=15,
                budget_ratios=[1.0], batch_size=16, backbone_lr_mult=0.1,
                output_dir=str(OUT / "uni_e2e"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=["cpu", "gpu", "serial"], required=True)
    args = ap.parse_args()
    global LOG
    LOG = OUT / f"RUN_FINAL_{args.lane}.md"
    log("=" * 70)
    log(f"RUN-FINAL [lane={args.lane}] — {datetime.now():%Y-%m-%d %H:%M}")
    log("=" * 70)

    if args.lane == "cpu":
        run_job("AL20-her2", al20("her2"))
        run_job("AL20-kidney", al20("kidney"))
    elif args.lane == "gpu":
        run_job("UNIe2e-kidney", uni_e2e("kidney"))
        run_job("UNIe2e-her2", uni_e2e("her2"))
    else:
        # SERIAL: never run these concurrently. Profiling showed the AL sweep
        # trains its MLP on the GPU in ~480 short bursts; running it alongside a
        # job that pins the card at 100% (UNI ViT-L) starved those bursts and cost
        # a ~6x slowdown (3 h of work took >18 h). Ordered by value-per-hour so
        # the most useful result lands first and stopping early is cheap.
        run_job("AL20-her2", al20("her2"))          # ~3 h  closes "only 8 of 29"
        run_job("UNIe2e-her2", uni_e2e("her2"))     # ~4 h  likely new ceiling
        run_job("AL20-kidney", al20("kidney"))      # ~8 h  breadth, 2nd dataset
        run_job("UNIe2e-kidney", uni_e2e("kidney")) # ~7 h  best dataset

    log(f"LANE [{args.lane}] COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
