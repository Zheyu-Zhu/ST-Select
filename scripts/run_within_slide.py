"""Within-slide acquisition: measure part of a slide, predict the rest.

This is the question the cost argument in the SPIE abstract actually rests on,
and it is NOT the question the rest of this benchmark answers. Everything else
here uses patient-level splits and asks "can a model trained on some patients
generalise to a NEW patient". This asks "given a slide you are going to profile,
can you measure a fraction of its spots and predict the remainder".

Why it may give a different answer. Every null result in this project was
measured where cross-patient generalisation is the bottleneck — and we concluded
repeatedly that the bottleneck is patient shift, not spot choice. Within-slide
removes patient shift entirely: the model has already seen this slide's
morphology, staining and batch effects, and only has to interpolate. That is the
regime where informative selection is most plausible, and where spatial
strategies (poisson_disk, probcover, spatial_stratified) have a real mechanism.
It is the strongest remaining chance of a positive result.

⚠ The control that makes or breaks the interpretation
-----------------------------------------------------
With an interleaved holdout, every test spot has labelled immediate neighbours,
and ST expression is strongly spatially autocorrelated. A model could therefore
score well by doing nothing but local smoothing — learning nothing from
morphology at all. So this script always runs a NEAREST-NEIGHBOUR CONTROL: predict
each test spot as the mean of its k nearest *labelled* spots on the same slide,
ignoring the image entirely.

Read the results through that control:
  * model >> kNN control  -> morphology genuinely carries information
  * model ~= kNN control  -> the task is spatial interpolation; any "AL wins"
                             claim is a claim about coverage, not informativeness
  * strategy > random but neither beats kNN -> not publishable as an AL result

Both holdout geometries are supported: `--mode random` (interleaved, matches
deployment) and `--mode spatial` (contiguous bands, harder, isolates morphology).

    python scripts/run_within_slide.py --dataset pg_her2_hvg300
    python scripts/run_within_slide.py --dataset pg_her2_hvg300 --mode spatial
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.datasets import WithinSlideSplit  # noqa: E402
from src.evaluation.metrics import pcc_per_gene_grouped  # noqa: E402
from src.run_experiment import run_full_benchmark, _records_from_ids  # noqa: E402
from src.utils import ExperimentConfig  # noqa: E402

OUT = ROOT / "results_pg" / "within_slide"
CACHE = ROOT / "feature_cache"
LOG = OUT / "RUN_WITHIN_SLIDE.md"

RATIOS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
# The 8 core strategies plus the spatial family, which is the one with a real
# mechanism here: on an interpolation problem, coverage should matter.
METHODS = ["random", "entropy", "tod", "coreset", "typiclust", "badge",
           "poisson_disk", "spatial_stratified"]


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def knn_control(dataset: str, mode: str, seed: int, k: int = 6) -> dict:
    """Nearest-neighbour baseline: no image, no model, no training.

    For each test spot, predict the mean expression of its k nearest LABELLED
    spots on the same slide. Uses the same random draw the AL loop starts from,
    at each budget, so it is directly comparable to the model curves.
    """
    d = np.load(CACHE / f"{dataset}.npz", allow_pickle=True)
    expr = d["expressions"].astype("float32")
    pos = d["positions"].astype("float32")
    sid = np.asarray(d["sample_ids"]).astype(str)
    recs = _records_from_ids(d["patient_ids"], d["sample_ids"], len(expr), pos)
    folds = WithinSlideSplit(n_folds=4, seed=seed, mode=mode).split(recs, "sample_id")

    out = {f"{r:.2f}": [] for r in RATIOS}
    for pool, test in folds:
        pool, test = np.asarray(pool), np.asarray(test)
        rng = np.random.default_rng(seed)
        for r in RATIOS:
            n_lab = max(1, int(round(r * len(pool))))
            labelled = rng.choice(pool, size=n_lab, replace=False)
            pred = np.zeros((len(test), expr.shape[1]), dtype="float32")
            for slide in np.unique(sid[test]):
                t_idx = np.where(sid[test] == slide)[0]
                l_idx = labelled[sid[labelled] == slide]
                if len(l_idx) == 0:                 # nothing labelled on this slide
                    pred[t_idx] = expr[labelled].mean(0)
                    continue
                dist = np.linalg.norm(
                    pos[test[t_idx]][:, None, :] - pos[l_idx][None, :, :], axis=2)
                kk = min(k, len(l_idx))
                nn = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
                pred[t_idx] = expr[l_idx][nn].mean(axis=1)
            out[f"{r:.2f}"].append(
                float(pcc_per_gene_grouped(pred, expr[test], sid[test])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pg_her2_hvg300")
    ap.add_argument("--mode", default="random", choices=["random", "spatial"])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    tag = f"{args.dataset}_{args.mode}"
    out_dir = OUT / args.mode
    if (out_dir / args.dataset / "results.json").exists():
        log(f"### SKIP {tag} — exists")
        return

    log("=" * 70)
    log(f"WITHIN-SLIDE ACQUISITION · {args.dataset} · mode={args.mode}")
    log(f"methods={args.methods} seeds={args.seeds} ratios={RATIOS}")
    log("train and test spots share slides — NOT comparable to patient-split numbers")
    log("=" * 70)

    t0 = time.time()

    # Control first: it is cheap, needs no GPU, and if the model cannot beat it
    # the rest of the numbers do not mean what they appear to mean.
    log("--- nearest-neighbour control (no image, no model) ---")
    try:
        ctrl = knn_control(args.dataset, args.mode, seed=42)
        log("      " + "  ".join(
            f"{float(r):.0%}={np.mean(v):.4f}" for r, v in ctrl.items() if v))
    except Exception:
        ctrl = {}
        log(f"      control FAILED\n{traceback.format_exc()}")

    cfg = ExperimentConfig(
        dataset_name=args.dataset, model_name="feature_predictor",
        feature_cache_dir="./feature_cache", output_dir=str(out_dir),
        al_methods=list(args.methods), budget_ratios=list(RATIOS),
        split_level="within_slide", within_slide_mode=args.mode,
        n_folds=4, n_seeds=args.seeds,
        # Early stopping, as established in E18 — a fixed epoch count penalises
        # large budgets harder and tilts the whole curve.
        epochs_per_round=200, val_frac=0.15, patience=10,
        lr=1e-4, loss_fn="mse", pcc_top_n=[10, 50, 100],
        batch_size=128, initial_budget=256, budget_per_round=512,
        add_full_supervision=True,
        device="cuda", seed=42, num_workers=0,
    )

    try:
        res = run_full_benchmark(cfg)
        mins = (time.time() - t0) / 60
        ms = {m: v for m, v in (res.get("mean_pcc_per_gene_by_slide") or {}).items()
              if v is not None}
        rnd = ms.get("random")
        log(f"### DONE {tag} in {mins:.1f} min")
        for m, v in sorted(ms.items(), key=lambda kv: -kv[1]):
            delta = f"{v - rnd:+.4f}" if rnd is not None and m != "random" else "     —"
            log(f"      {m:20s} per_slide={v:.4f}  vs_random={delta}")
        if ctrl and rnd is not None:
            c100 = np.mean(ctrl.get("1.00", [np.nan]))
            log(f"      kNN control @100% = {c100:.4f}   "
                f"model random @100% = {rnd:.4f}   "
                f"model advantage = {rnd - c100:+.4f}")
            if np.isfinite(c100) and rnd - c100 < 0.02:
                log("      ⚠ the model barely beats spatial interpolation — read any "
                    "AL difference as coverage, not informativeness")
        (out_dir / args.dataset).mkdir(parents=True, exist_ok=True)
        json.dump(ctrl, open(out_dir / args.dataset / "knn_control.json", "w"), indent=2)
    except Exception:
        log(f"### FAILED {tag} after {(time.time()-t0)/60:.1f} min\n{traceback.format_exc()}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
