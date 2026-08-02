"""Breadth vs depth — is the binding constraint patients, or spots?

Why this exists
---------------
E18 measured how many labels each cohort needs to match full supervision, and got
three very different answers: HER2 5%, Kidney 50%, BC 75%. The write-up stops
there, because with n = 3 cohorts nothing can be claimed about *why*. The one
structural variable that tracks the ordering is the number of TRAINING PATIENTS
(6 -> 5%, 16 -> 50%, 17 -> 75%) — a cohort with few patients appears to exhaust
its available diversity almost immediately — but patient count is confounded with
cohort size (10.2k / 16.8k / 22.3k training spots), so the correlation proves
nothing on its own.

This run breaks the confound directly. **Hold the spot budget fixed and vary how
many patients supply those spots.** Same number of labels every time, so cost to
the lab is identical; only the breadth of the sample changes.

    k = 2 patients  ->  B spots, deeply sampled from few patients
    k = 4           ->  B spots, spread wider
    k = 8 ...       ->  B spots, spread wider still

If per-slide PCC rises with k at fixed B, patients are the binding constraint and
the practical rule is "label fewer spots per section, across more patients" —
which is actionable advice a lab can act on before spending anything. If the
curve is flat, spots are fungible and only the total matters, which would make
E18's spread a property of cohort size rather than composition.

Design notes
------------
* Budget B is chosen per cohort as a fraction of the smallest patient-subset that
  could supply it, so every k is feasible without resampling bias; B is reported.
* Patient subsets are drawn per (fold, seed) so the comparison is paired.
* Early stopping uses the same group-aware holdout as E18 (a held-out patient),
  except at k = 2 where holding a patient out would leave one — there it falls
  back to a random row split, and the log marks it.
* Evaluation is always the full test fold, untouched, per-slide PCC.

    conda run -n st-select python -u scripts/run_breadth_depth.py
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

from src.datasets import PatientKFold, STFeatureDataset  # noqa: E402
from src.models import FeaturePredictor  # noqa: E402
from src.run_experiment import _evaluate_model  # noqa: E402
from src.training import ALTrainer, group_aware_val_split  # noqa: E402
from src.utils import set_seed, get_device  # noqa: E402

OUT = ROOT / "results_pg" / "breadth_depth"
CACHE = ROOT / "feature_cache"
LOG = OUT / "RUN_BREADTH_DEPTH.md"

DATASETS = ["pg_her2_hvg300", "pg_kidney_hvg300", "pg_bc_hvg300"]
K_LEVELS = [2, 4, 8, 16]          # clipped per cohort to its training patients
SEEDS = [42, 43, 44]
VAL_FRAC = 0.15
EPOCHS = 200                      # cap; early stopping decides (see E18)
PATIENCE = 10


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def choose_budget(pid_train: np.ndarray, ks: list) -> int:
    """Largest spot budget every k can supply, with headroom.

    Taking the smallest k and the smallest patients it could draw guarantees the
    budget is reachable for every k without ever resampling to find a subset that
    fits — resampling would bias which patients appear at small k.
    """
    counts = np.sort([int((pid_train == p).sum()) for p in np.unique(pid_train)])
    kmin = min(ks)
    return int(0.8 * counts[:kmin].sum())


def run_cell(feats, expr, pos, pid, sid, tr, te, k, budget, seed, device, gene_names):
    """One (fold, seed, k) cell: sample k patients, draw `budget` spots, train, eval."""
    set_seed(seed)
    rng = np.random.default_rng(seed * 100 + k)
    pid_tr = pid[tr]
    pats = np.unique(pid_tr)
    if k > len(pats):
        return None

    # Draw k patients, then spread the budget evenly over them (capped by supply).
    chosen = rng.choice(pats, size=k, replace=False)
    per = budget // k
    picked = []
    for p in chosen:
        idx = np.where(pid_tr == p)[0]
        take = min(per, len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False).tolist())
    # Top up from the chosen patients if integer division left us short.
    if len(picked) < budget:
        pool = [i for i in np.where(np.isin(pid_tr, chosen))[0] if i not in set(picked)]
        if pool:
            extra = rng.choice(pool, size=min(budget - len(picked), len(pool)), replace=False)
            picked.extend(extra.tolist())
    picked = sorted(picked)
    if len(picked) < 0.9 * budget:      # cohort could not supply it; skip honestly
        return None

    train_ds = STFeatureDataset(feats[tr], expr[tr], pos[tr] if pos is not None else None)
    test_ds = STFeatureDataset(feats[te], expr[te], pos[te] if pos is not None else None)

    trn, val = group_aware_val_split(picked, VAL_FRAC, seed, pid_tr)
    model = FeaturePredictor(n_genes=expr.shape[1], feature_dim=feats.shape[1])
    trainer = ALTrainer(model=model, lr=1e-4, epochs=EPOCHS, batch_size=128,
                        device=device, loss_fn="mse", patience=PATIENCE)
    hist = (trainer.train(train_ds, trn, val_dataset=train_ds, val_indices=val)
            if val else trainer.train(train_ds, picked))

    m = _evaluate_model(trainer, test_ds, device, [10, 50, 100], sid[te], None)
    return {
        "k": k, "seed": seed, "n_spots": len(picked),
        "n_patients_used": int(len(np.unique(pid_tr[picked]))),
        "n_val_patients": int(len(np.unique(pid_tr[val]))) if val else 0,
        "epochs_run": len(hist.get("train_loss", [])),
        "per_slide": m.get("pcc_per_gene_by_slide"),
        "per_gene": m.get("pcc_per_gene"),
        "mse": m.get("mse"),
    }


def run_dataset(name: str) -> None:
    out = OUT / name
    if (out / "results.json").exists():
        log(f"### SKIP {name} — exists")
        return
    cache = CACHE / f"{name}.npz"
    if not cache.exists():
        log(f"### SKIP {name} — cache missing")
        return

    d = np.load(cache, allow_pickle=True)
    feats = d["features"].astype("float32")
    expr = d["expressions"].astype("float32")
    pos = d["positions"].astype("float32") if "positions" in d else None
    pid = np.asarray(d["patient_ids"]).astype(str)
    sid = np.asarray(d["sample_ids"]).astype(str) if "sample_ids" in d else pid
    gene_names = d["gene_names"] if "gene_names" in d else None

    folds = PatientKFold(n_folds=4, seed=42).split([{"patient_id": p} for p in pid])
    device = get_device("cuda")
    n_train_pat = len(np.unique(pid[np.asarray(folds[0][0])]))
    ks = [k for k in K_LEVELS if k <= n_train_pat]
    if n_train_pat not in ks:
        ks.append(int(n_train_pat))          # always include "all patients"
    budget = choose_budget(pid[np.asarray(folds[0][0])], ks)

    log(f"### START {name} — {n_train_pat} training patients, k={ks}, "
        f"fixed budget B={budget} spots, {len(SEEDS)} seeds x 4 folds")

    rows, t0 = [], time.time()
    for fi, (tr, te) in enumerate(folds):
        tr, te = np.asarray(tr), np.asarray(te)
        for seed in SEEDS:
            for k in ks:
                try:
                    r = run_cell(feats, expr, pos, pid, sid, tr, te, k, budget,
                                 seed, device, gene_names)
                except Exception:
                    log(f"    cell FAILED fold{fi} seed{seed} k{k}\n{traceback.format_exc()}")
                    continue
                if r is None:
                    continue
                r["fold"] = fi
                rows.append(r)
        done = [r for r in rows if r["fold"] == fi]
        log(f"    fold {fi}: {len(done)} cells, "
            + "  ".join(f"k{k}={np.mean([r['per_slide'] for r in done if r['k']==k]):.4f}"
                        for k in ks if any(r['k'] == k for r in done)))

    out.mkdir(parents=True, exist_ok=True)
    summary = {m: {"per_slide": float(np.mean([r["per_slide"] for r in rows if r["k"] == m])),
                   "per_gene": float(np.mean([r["per_gene"] for r in rows if r["k"] == m])),
                   "n_cells": sum(1 for r in rows if r["k"] == m),
                   "mean_spots": float(np.mean([r["n_spots"] for r in rows if r["k"] == m])),
                   "mean_epochs": float(np.mean([r["epochs_run"] for r in rows if r["k"] == m]))}
               for m in sorted({r["k"] for r in rows})}
    json.dump({"dataset": name, "budget_spots": budget, "k_levels": ks,
               "n_train_patients": int(n_train_pat), "seeds": SEEDS,
               "val_frac": VAL_FRAC, "epochs_cap": EPOCHS,
               "summary": summary, "cells": rows},
              open(out / "results.json", "w"), indent=2)

    log(f"### DONE {name} in {(time.time()-t0)/60:.1f} min  (B={budget} spots, fixed)")
    for k, v in summary.items():
        log(f"      k={k:2d} patients  per-slide={v['per_slide']:.4f}  "
            f"per-gene={v['per_gene']:.4f}  spots={v['mean_spots']:.0f}  "
            f"epochs={v['mean_epochs']:.0f}  n={v['n_cells']}")
    ks_sorted = sorted(summary)
    if len(ks_sorted) > 1:
        lo, hi = summary[ks_sorted[0]]["per_slide"], summary[ks_sorted[-1]]["per_slide"]
        log(f"      -> breadth effect at fixed budget: {hi - lo:+.4f} per-slide "
            f"going from {ks_sorted[0]} to {ks_sorted[-1]} patients")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()
    log("=" * 70)
    log("BREADTH vs DEPTH — same spot budget, different numbers of patients")
    log("=" * 70)
    for n in args.datasets:
        try:
            run_dataset(n)
        except Exception:
            log(f"### FAILED {n}\n{traceback.format_exc()}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    log("=== breadth-vs-depth complete ===")


if __name__ == "__main__":
    main()
