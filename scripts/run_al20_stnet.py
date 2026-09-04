"""The same 20-strategy acquisition sweep on a 40x smaller end-to-end model.

Why this exists. On UNI ViT-L end-to-end (304 M params) the 5% budget scores
0.3245 against 0.3310 at 50% - 98% of the performance from 5% of the labels.
That is a suspicious result to build a paper on, and one obvious explanation is
the model: a 304 M-parameter backbone fine-tuned on 490 spots is so far into the
over-parameterised regime that the extra labels have nothing left to buy. If the
saturation is a property of the *task*, a much smaller model should show it too.
If it is a property of the *model*, the small one should keep improving.

st_net is DenseNet-121 fine-tuned end to end: 7.69 M parameters, 40x smaller.

Settings differ from the UNI sweep and that is deliberate, so read the numbers as
a separate experiment rather than a drop-in comparison:

    batch 96    - 128 also fits (8.1 GB) but 96 buys back gradient steps at the
                  small budgets, which is where this experiment lives. 192 is not
                  an option: it does not OOM on Windows, it oversubscribes into
                  host memory and runs 9x slower (0.667 -> 6.19 s/step).
    lr 5e-5
    45 epochs, no early stopping. val_frac = 0, so nothing is restored from a
                  checkpoint - the reported model is literally the one at the end
                  of epoch 45.
    budgets 5 / 15 / 30 / 50%, plus full supervision as the ceiling.

The step budget is the thing to watch, since an earlier configuration failed on
exactly this. Fixed epochs equalises how many times each spot is seen, not how
many gradient updates happen, and at small budgets those diverge hard:

    budget   spots   batches/epoch   steps over 45 epochs
      5%       490          6                 270
     15%      1472         16                 720
     30%      2943         31               1395
     50%      4905         52               2340
    100%      9810        103               4635

A 3-epoch probe at the previous settings (batch 128, lr 2e-5) left the 5%
checkpoint at per-slide PCC 0.011 after 12 steps while full supervision reached
0.390 after 231 - i.e. the small budgets were nowhere near converged. These
settings give the 5% checkpoint 270 steps at 2.5x the learning rate, roughly 5x
the total movement of that configuration. record_epoch_curve stays on so the
question is checkable rather than assumed: if the 5% curve is still climbing at
epoch 45, the budget comparison is measuring optimisation, not data.

    python scripts/run_al20_stnet.py --dataset pg_her2_hvg300
"""

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.run_experiment import run_full_benchmark  # noqa: E402
from src.utils import ExperimentConfig            # noqa: E402

CACHE = ROOT / "feature_cache"
OUT = ROOT / "results_pg" / "al20_stnet"
RATIOS = [0.05, 0.15, 0.30, 0.50]

# Same list, same order as scripts/run_al20_e2e.py, so the two sweeps line up
# strategy for strategy. Note tissue_aware is a known duplicate of random - it
# never receives a tissue mask and has no inner strategy, so it falls through to
# a plain random draw. Kept in for now so the two sweeps stay comparable; it
# should be fixed or dropped before anything is written up.
AL20 = ["random", "entropy", "margin", "least_confidence", "mc_dropout", "tod",
        "coreset", "typiclust", "maxherding", "probcover", "badge", "similar",
        "alfa_mix", "cald", "ceal", "coregcn", "lal_rl", "poisson_disk",
        "spatial_stratified", "tissue_aware"]


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pg_her2_hvg300")
    ap.add_argument("--methods", nargs="+", default=AL20)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--max-folds", type=int, default=0,
                    help="run only the first N folds (smoke tests); 0 = all")
    ap.add_argument("--no-epoch-curve", action="store_true")
    args = ap.parse_args()

    if not (CACHE / f"{args.dataset}.npz").exists():
        log(f"### ABORT {args.dataset} - cache missing")
        return
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        dataset_name=args.dataset, model_name="st_net", image_mode=True,
        feature_cache_dir="./feature_cache", output_dir=str(OUT),
        al_methods=list(args.methods), budget_ratios=list(RATIOS),
        add_full_supervision=True,     # no reusable ceiling exists for this model
        split_level="patient", n_folds=args.folds, n_seeds=1,
        epochs_per_round=args.epochs, full_epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
        pretrained=True, frozen_backbone=False,
        loss_fn="mse", val_frac=0.0,
        record_epoch_curve=not args.no_epoch_curve,
        budget_per_round=512, initial_budget=256,
        pcc_top_n=[10, 50, 100], device="cuda", seed=42, num_workers=0,
        **({"max_folds": args.max_folds} if args.max_folds else {}),
    )

    n_train = len(args.methods) * (len(RATIOS) + 1) * cfg.n_folds
    log("=" * 74)
    log(f"AL20 x st_net (DenseNet-121 e2e, 7.69 M params) - {args.dataset}")
    log(f"{len(args.methods)} strategies x {len(RATIOS)} budgets + full sup x "
        f"{cfg.n_folds} folds = {n_train} model trainings")
    log(f"epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}  "
        f"epoch_curve={cfg.record_epoch_curve}")
    log(f"at 5% that is {int(490 / args.batch_size) + 1} batches/epoch = "
        f"~{(int(490 / args.batch_size) + 1) * args.epochs} gradient steps")
    log("=" * 74)

    t0 = time.time()
    try:
        run_full_benchmark(cfg)
    except Exception:
        log("### FAILED\n" + traceback.format_exc())
        raise
    log(f"done in {(time.time() - t0) / 3600:.2f} h  ->  {OUT}")


if __name__ == "__main__":
    main()
