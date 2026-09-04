"""How much does the stopping epoch move the number? UNI end-to-end, HER2.

Every UNI end-to-end result in this project trains a fixed 15 epochs with no
validation split and reports whatever the model looks like at the end. That is a
defensible protocol, but it has never been checked: nobody has seen the epoch
curve, so nobody knows whether epoch 15 is near the top, past it, or short of it.

This runs the same configuration with a probe that evaluates the TEST split after
every epoch, producing a (budget x fold x epoch) grid of per-slide PCC. From that
grid two protocols can be compared:

    last epoch   - what is reported today. No selection, no leakage.
    best epoch   - the maximum over the 15 epochs.

The second is NOT a reportable result. Picking the epoch on test is test-set
selection: each arm gets fifteen draws and keeps its luckiest, so the arm with
the noisiest epoch curve wins regardless of whether it is better. Since every
strategy effect measured so far is smaller than the training noise, that bias
would dominate the ranking outright. The reason to compute it anyway is to put a
number on the gap - if best-minus-last is large, the fixed-15 protocol is leaving
performance on the table and deserves a validation split; if it is small, the
current protocol is fine and the paper can say so with evidence.

    python scripts/run_epoch_curve.py --dataset pg_her2_hvg300 --methods random
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.run_experiment import run_full_benchmark  # noqa: E402
from src.utils import ExperimentConfig            # noqa: E402


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pg_her2_hvg300")
    ap.add_argument("--methods", nargs="+", default=["random"])
    ap.add_argument("--budgets", nargs="+", type=float,
                    default=[0.05, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--out", default="results_pg/epoch_curve")
    args = ap.parse_args()

    out = ROOT / args.out / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    # Everything here mirrors scripts/run_al20_e2e.py exactly except the two
    # marked lines, so the curve describes the protocol actually in use.
    cfg = ExperimentConfig(
        dataset_name=args.dataset, model_name="uni_regressor", image_mode=True,
        feature_cache_dir="./feature_cache", output_dir=str(out),
        al_methods=list(args.methods), budget_ratios=list(args.budgets),
        add_full_supervision=False,
        split_level="patient", n_folds=args.folds, n_seeds=1,
        epochs_per_round=args.epochs, full_epochs=args.epochs, batch_size=16,
        backbone_lr_mult=0.1, lr=1e-4, loss_fn="mse",
        grad_checkpointing=False,
        budget_per_round=512, initial_budget=256,
        pcc_top_n=[10, 50, 100], device="cuda", seed=42, num_workers=0,
        val_frac=0.0,                    # unchanged: same protocol as the sweep
        record_epoch_curve=True,         # the only difference
    )

    log("=" * 72)
    log(f"EPOCH CURVE  ·  UNI e2e  ·  {args.dataset}")
    log(f"{len(args.methods)} method(s) x {len(args.budgets)} budgets x {args.folds} folds"
        f" x {args.epochs} epochs")
    log("test is evaluated after EVERY epoch - diagnostic only, never for selection")
    log("=" * 72)

    t0 = time.time()
    run_full_benchmark(cfg)
    log(f"done in {(time.time() - t0) / 3600:.2f} h  ->  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
