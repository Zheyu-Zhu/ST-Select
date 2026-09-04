"""The agreed SPIE protocol: 20 acquisition strategies, UNI fine-tuned end to end.

Protocol settled with the collaborator:

  * patient-level split (all slides of a patient stay together) — a slide-level
    split would put 5-7 of HER2's 8 patients on both sides and inflate scores
    1.19-1.83x
  * budgets 5 / 10 / 20 / 30 / 50%, each a fraction of every training slide
  * **100% is NOT re-run** — at full budget every strategy has labelled the whole
    pool, so the point carries zero information about acquisition. The existing
    full-supervision ceilings are reused instead:
        HER2 0.3313 · Kidney 0.3876 · BC 0.1582  (per-slide)
  * 20 strategies (the 9 never-run ones stay out; see BENCHMARK_METHODS.md)

Speed: gradient checkpointing is OFF. Measured on the real HER2 pipeline it is
1.48x faster (0.37 -> 0.25 s/step) and still fits in 8.35 of 10.24 GB. Gradients
are mathematically identical with or without it, so these numbers stay directly
comparable to the existing UNI e2e results — only CUDA nondeterminism differs,
at the ~0.001 level that two identical re-runs already showed.

NOT changed, deliberately: batch size stays 16 and the backbone stays fully
fine-tuned. Doubling the batch halves the gradient steps and changes the
effective learning rate, and LoRA trains 0.83M instead of 304M parameters — both
would break comparability with everything measured so far. Either is worth doing
later, but only after re-measuring the ceiling under the new setting.

    python scripts/run_al20_e2e.py --dataset pg_her2_hvg300
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

OUT = ROOT / "results_pg" / "al20_e2e"
CACHE = ROOT / "feature_cache"
LOG = OUT / "RUN_AL20_E2E.md"

RATIOS = [0.05, 0.1, 0.2, 0.3, 0.5]      # 100% reused from full_supervision

# The 20 that have run before. batchbald and cdal are excluded (O(N^2) with a
# greedy inner loop, 20.8 s and 6.7 s per acquisition at N=2000 — they caused a
# 5 h hang), as are the 7 others never yet exercised.
AL20 = [
    "random", "entropy", "margin", "least_confidence", "mc_dropout",
    "tod", "coreset", "typiclust", "maxherding", "probcover", "badge",
    "similar", "alfa_mix", "cald", "ceal", "coregcn", "lal_rl",
    "poisson_disk", "spatial_stratified", "tissue_aware",
]

CEILING = {"pg_her2_hvg300": 0.3313, "pg_kidney_hvg300": 0.3876,
           "pg_bc_hvg300": 0.1582}


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pg_her2_hvg300")
    ap.add_argument("--methods", nargs="+", default=AL20)
    args = ap.parse_args()

    done = OUT / args.dataset / "results.json"
    if done.exists():
        log(f"### SKIP {args.dataset} — exists")
        return
    if not (CACHE / f"{args.dataset}.npz").exists():
        log(f"### ABORT {args.dataset} — cache missing")
        return

    cfg = ExperimentConfig(
        dataset_name=args.dataset, model_name="uni_regressor", image_mode=True,
        feature_cache_dir="./feature_cache", output_dir=str(OUT),
        al_methods=list(args.methods), budget_ratios=list(RATIOS),
        add_full_supervision=False,          # ceiling already measured, see above
        split_level="patient", n_folds=4, n_seeds=1,
        epochs_per_round=15, full_epochs=15, batch_size=16,
        backbone_lr_mult=0.1, lr=1e-4, loss_fn="mse",
        grad_checkpointing=False,            # 1.48x, mathematically identical
        budget_per_round=512, initial_budget=256,
        pcc_top_n=[10, 50, 100], device="cuda", seed=42, num_workers=0,
    )

    n_train = len(args.methods) * len(RATIOS) * cfg.n_folds
    log("=" * 72)
    log(f"AL20 x UNI e2e — {args.dataset}")
    log(f"{len(args.methods)} strategies x {len(RATIOS)} budgets x {cfg.n_folds} folds "
        f"= {n_train} model trainings")
    log(f"budgets={RATIOS}  (100% reused: ceiling = {CEILING.get(args.dataset,'?')})")
    log(f"grad_checkpointing=False (1.48x, identical gradients)  batch={cfg.batch_size}")
    log("=" * 72)

    t0 = time.time()
    try:
        res = run_full_benchmark(cfg)
        mins = (time.time() - t0) / 60
        ms = {m: v for m, v in (res.get("mean_pcc_per_gene_by_slide") or {}).items()
              if v is not None}
        rnd = ms.get("random")
        log(f"### DONE {args.dataset} in {mins:.1f} min ({mins/60:.1f} h)")
        for m, v in sorted(ms.items(), key=lambda kv: -kv[1]):
            d = f"{v - rnd:+.4f}" if rnd is not None and m != "random" else "      —"
            log(f"      {m:20s} per_slide={v:.4f}  vs_random={d}")
        if rnd is not None:
            best = max((m for m in ms if m != "random"), key=lambda m: ms[m])
            log(f"      -> best non-random: {best} {ms[best]-rnd:+.4f}")
            log(f"      -> ceiling (100%, from full_supervision): "
                f"{CEILING.get(args.dataset, float('nan'))}")
    except Exception:
        log(f"### FAILED after {(time.time()-t0)/60:.1f} min\n{traceback.format_exc()}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
