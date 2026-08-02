"""Task #2 - does the "no strategy beats random" null survive on the big model?

Why this exists
---------------
Every AL result we have is on frozen features + a small MLP head
(feature_predictor, 18 runs) or DenseNet (st_net, 2 runs). The best model on this
data by a wide margin is uni_regressor - UNI ViT-L fine-tuned end to end, 304.08 M
trainable params, 0 frozen tensors - and it has NEVER been AL-tested. The standard
objection writes itself: "your strategies had nothing to work with because the
representation was frozen; uncertainty and diversity signals only become
informative once the backbone adapts." This is the one remaining experiment that
could overturn the paper's main negative result, so it has to be run even though
a full 20-strategy sweep is infeasible (~25 h per strategy).

Design
------
HER2, 4 patient folds, 1 seed, budget_ratios = [0.05, 0.30].

The two ratios are load-bearing, not decoration. The AL loop seeds its labeled
pool at the FIRST budget target, so a single-target run ([0.30]) is filled
entirely by that random draw: the acquisition loop never executes and every
strategy reports the same numbers. That failure is invisible in the output - it
looks precisely like "no strategy beats random" while testing nothing. With
[0.05, 0.30] the strategy chooses 25 of the 30 percentage points (83% of the
final pool). training.loop now raises rather than allow the single-target case.

Strategy picks are pre-registered from the 20-strategy sweep by MEAN delta across
BOTH pg datasets, not by rank on HER2 alone - HER2's own top-3 (cald,
poisson_disk, maxherding) all reverse sign on Kidney, so picking them here would
be selecting on the same noise we then test:

  poisson_disk  +0.0017 mean   spatial coverage
  tod           +0.0012 mean   uncertainty / temporal discrepancy
  typiclust     +0.0010 mean   diversity / clustering
  random         reference

Three different families, so a positive result cannot be dismissed as one
family's quirk. Each is still within noise at frozen features; the question is
whether fine-tuning changes that.

Warm start - read the 30% numbers accordingly
---------------------------------------------
Between budget checkpoints the loop calls _reset_weights_partial, which reinits
Linear layers with out_features < 1024. For uni_regressor that is 0 of the 96
backbone layers and all 3 head layers: the ViT-L backbone therefore CARRIES OVER
from the 5% checkpoint into the 30% one, with only the head reinitialized.

That is fine, and arguably required, for the question being asked: every arm gets
the identical warm start (same seed, so the 5% seed pool is the same random draw
for all four), so the strategy-vs-random comparison at 30% is apples-to-apples,
and the representation genuinely moves during the run - which is precisely the
regime the "frozen features gave AL nothing to work with" objection points at.

What it does mean: the 30% scores are NOT directly comparable to the previously
measured 100% UNI e2e ceiling, which was trained once from pretrained weights.
These numbers answer "does strategy beat random", not "does 30% match 100%".

Cost: training happens once per budget target, so each strategy is ~35% of one
full-supervision pass per fold, plus a backbone feature extraction per target for
the three non-random arms. Measured per-job times land in RUN_UNI_AL.md - the
estimates in this file's history have been wrong repeatedly, so trust the log.

Runs strictly serially, and must not share the GPU with anything (UNI pins the
card at 100%; see EXPERIMENTS.md S12 for the ~6x slowdown this caused once).

  conda run -n st-select python -u scripts/run_uni_al.py
"""

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.run_experiment import run_full_benchmark  # noqa: E402
from src.utils import ExperimentConfig  # noqa: E402

OUT = ROOT / "results_pg" / "uni_al"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "feature_cache"
LOG = OUT / "RUN_UNI_AL.md"

RATIOS = [0.05, 0.30]
METHODS = ["random", "poisson_disk", "tod", "typiclust"]


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def cfg(dataset: str, methods, dry: bool = False) -> ExperimentConfig:
    if dry:
        # Cheapest run that still executes every real code path: UNI backbone
        # feature extraction in image mode, head_only prediction inside each
        # strategy, acquisition, then train+eval. Image-mode AL has never been
        # exercised with these strategies (the UNI runs used full_supervision,
        # which bypasses the loop), so this validates the plumbing before
        # committing a full night to it.
        return ExperimentConfig(
            dataset_name=dataset, model_name="uni_regressor", image_mode=True,
            feature_cache_dir="./feature_cache", output_dir=str(OUT / "_dry"),
            al_methods=list(methods), budget_ratios=[0.02, 0.03],
            add_full_supervision=False, split_level="patient",
            # n_folds must stay 4 (PatientKFold requires >= 2, and keeping it at 4
            # means the dry run sees the same split geometry as the real run);
            # max_folds=1 then runs only the first fold.
            n_folds=4, max_folds=1, n_seeds=1, epochs_per_round=1, full_epochs=1,
            batch_size=16, backbone_lr_mult=0.1, lr=1e-4, loss_fn="mse",
            budget_per_round=128, initial_budget=128,
            pcc_top_n=[10], device="cuda", seed=42, num_workers=0,
        )
    return ExperimentConfig(
        dataset_name=dataset, model_name="uni_regressor", image_mode=True,
        feature_cache_dir="./feature_cache", output_dir=str(OUT),
        al_methods=list(methods), budget_ratios=list(RATIOS),
        add_full_supervision=False,      # the 100% ceiling is already measured
        split_level="patient", n_folds=4, n_seeds=1,
        # Same recipe as the measured UNI e2e ceiling, so the AL numbers are
        # comparable to it: 15 epochs, bs 16, backbone at 1/10th the head's LR.
        epochs_per_round=15, full_epochs=15, batch_size=16,
        backbone_lr_mult=0.1, lr=1e-4, loss_fn="mse",
        budget_per_round=512, initial_budget=256,
        pcc_top_n=[10, 50, 100], device="cuda", seed=42, num_workers=0,
    )


def report(res: dict) -> str:
    ms = {m: v for m, v in (res.get("mean_pcc_per_gene_by_slide", {}) or {}).items()
          if v is not None}
    mg = res.get("mean_pcc_per_gene", {}) or {}
    if not ms:
        return "      (no per-slide numbers)"
    lines = []
    rnd = ms.get("random")
    for m, v in sorted(ms.items(), key=lambda kv: -kv[1]):
        d = f"{v - rnd:+.4f}" if rnd is not None else "   n/a"
        lines.append(f"      {m:16s} per_slide={v:.4f}  vs_random={d}  "
                     f"per_gene={mg.get(m, float('nan')):.4f}")
    if rnd is not None and len(ms) > 1:
        best = max((m for m in ms if m != "random"), key=lambda m: ms[m])
        lines.append(f"      -> best non-random: {best} {ms[best] - rnd:+.4f} vs random")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pg_her2_hvg300")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--tag", default="",
                    help="suffix for the output dir, so a re-run does not collide "
                         "with an existing result (e.g. --tag rngfix)")
    ap.add_argument("--dry", action="store_true",
                    help="1 fold, 1 epoch, 2-3%% budgets: validate the image-mode "
                         "AL plumbing end to end before the real run")
    args = ap.parse_args()

    log("=" * 70)
    log(f"UNI e2e AL spot-check{' [DRY]' if args.dry else ''} - "
        f"{datetime.now():%Y-%m-%d %H:%M}")
    log(f"dataset={args.dataset} methods={args.methods} "
        f"ratios={[0.02, 0.03] if args.dry else RATIOS}")
    log("=" * 70)

    out_root = (OUT / "_dry") if args.dry else (OUT / args.tag if args.tag else OUT)
    done = out_root / args.dataset / "results.json"
    if done.exists():
        log(f"### SKIP {args.dataset} - exists ({done})")
        return
    if not (CACHE / f"{args.dataset}.npz").exists():
        log(f"### ABORT {args.dataset} - cache missing")
        return

    t0 = time.time()
    try:
        c = cfg(args.dataset, args.methods, dry=args.dry)
        c.output_dir = str(out_root)
        res = run_full_benchmark(c)
        log(f"### DONE {args.dataset} in {(time.time() - t0) / 60:.1f} min\n{report(res)}")
    except Exception:
        log(f"### FAILED {args.dataset} after {(time.time() - t0) / 60:.1f} min\n"
            f"{traceback.format_exc()}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
