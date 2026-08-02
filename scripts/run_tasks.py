"""Multi-task evaluation night: add the pitch's Evaluation Tasks #4 and #5.

Until now the benchmark scored one task (gene-expression reconstruction) with one
family of metrics (PCC/MSE + ARI/NMI). The collaborator's package ships the fields
needed for the two tasks the pitch lists but we could never run:

  Task #4  cell-type composition estimation   <- `cell_type_counts` (67/88 types)
  Task #5  tissue spatial architecture        <- `position` + Moran's I

Phases (cheap first, all resumable):

  T1  gene ceilings + spatial metrics    3 datasets x {DenseNet, UNI}   ~20 min
  T2  cell-type ceilings                 3 datasets x {DenseNet, UNI}   ~20 min
  T3  AL benchmark on the CELL-TYPE task  HER2 -> Kidney -> BC          ~7-8 h

T3 is the scientifically interesting one: our AL null was established on the
gene-expression target. If selection strategies are useless there but useful for a
*different* downstream target, that would be a real scope condition. Same 8
strategies, same patient-level 4-fold protocol.

Run:  conda run -n st-select python -u scripts/run_tasks.py
"""

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

OUT = ROOT / "results_pg" / "tasks"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "RUN_TASKS.md"
CACHE = ROOT / "feature_cache"

AL8 = ["random", "entropy", "tod", "coreset", "typiclust", "badge",
       "poisson_disk", "spatial_stratified"]


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
    lines = []
    mg = res.get("mean_pcc_per_gene", {})
    ms = res.get("mean_pcc_per_gene_by_slide", {})
    sp = res.get("spatial_autocorrelation", {})
    ct = res.get("celltype_composition", {})
    al = {m: v for m, v in ms.items() if m != "full_supervision" and v is not None}
    if len(al) > 1 and "random" in al:
        rnd = al["random"]
        best = max((m for m in al if m != "random"), key=lambda m: al[m])
        lines.append(f"      [AL] random={rnd:.4f} best={best}={al[best]:.4f} "
                     f"delta={al[best]-rnd:+.4f}")
    for m in mg:
        s = ms.get(m)
        row = (f"      {m:20s} per_gene={mg[m]:.4f} "
               f"per_slide={s if s is not None else float('nan'):.4f}")
        if m in sp:
            row += (f" | moran true={sp[m].get('moran_true', float('nan')):.3f} "
                    f"pred={sp[m].get('moran_pred', float('nan')):.3f} "
                    f"fid={sp[m].get('moran_fidelity', float('nan')):.3f}")
        if m in ct:
            row += (f" | cos={ct[m].get('cosine', float('nan')):.3f} "
                    f"jsd={ct[m].get('jsd', float('nan')):.3f} "
                    f"pccType={ct[m].get('pcc_per_type', float('nan')):.3f}")
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
    log(f"### START [{label}] {name} methods={len(cfg['al_methods'])}")
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


def ceiling(name, outdir):
    return base(dataset_name=name, model_name="feature_predictor",
                al_methods=["full_supervision"], add_full_supervision=False,
                full_epochs=60, epochs_per_round=60, budget_ratios=[1.0],
                batch_size=128, output_dir=str(OUT / outdir))


def al(name, outdir):
    return base(dataset_name=name, model_name="feature_predictor", al_methods=AL8,
                budget_ratios=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
                epochs_per_round=50, full_epochs=60, initial_budget=256,
                budget_per_round=512, batch_size=128, output_dir=str(OUT / outdir))


def main() -> None:
    log("=" * 70)
    log(f"RUN-TASKS: multi-task evaluation (pitch tasks #4, #5) — {datetime.now():%Y-%m-%d %H:%M}")
    log("=" * 70)

    # ---- T1: gene-expression ceilings, now WITH spatial-architecture metrics ----
    for ds in ["her2", "bc", "kidney"]:
        run_job(f"T1-gene-{ds}", ceiling(f"pg_{ds}_hvg300", "gene_spatial"))
        run_job(f"T1-gene-{ds}-uni", ceiling(f"pg_{ds}_hvg300_uni", "gene_spatial"))

    # ---- T2: cell-type composition ceilings ----
    for ds in ["her2", "bc", "kidney"]:
        run_job(f"T2-ct-{ds}", ceiling(f"pg_{ds}_celltype", "celltype"))
        run_job(f"T2-ct-{ds}-uni", ceiling(f"pg_{ds}_celltype_uni", "celltype"))

    # ---- T3: does the AL null hold on a DIFFERENT downstream task? ----
    # Smallest dataset first so the key answer lands early.
    for ds in ["her2", "kidney", "bc"]:
        run_job(f"T3-AL-ct-{ds}", al(f"pg_{ds}_celltype", "celltype_al"))

    log("ALL TASK PHASES COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
