"""Phased training driver for the collaborator's `processed_gene/` package.

Context: on this data the task is far more learnable than our earlier HEST
pipeline (HER2 per-gene 0.32 frozen / 0.40 e2e-1epoch, vs 0.066 before), because
the HVG-300 list is variance-ranked and led by immunoglobulin genes (IGKC,
IGHG3, ...) that form H&E-visible plasma-cell infiltrates. That makes it the
right substrate to re-ask the paper's core question: **does active learning beat
random when the task actually has signal?** Our negative result so far was
measured on a near-unlearnable target (PCC ~0.07).

Phases, cheap-and-high-value first, each job resumable (skips if results.json
exists) and failure-isolated:

  P1  caches      DenseNet features + PNG paths for BC / Kidney (HER2 done)
  P2  frozen      full-supervision baseline, all 3 datasets      (the ceiling)
  P3  AL-frozen   8-strategy AL benchmark on HER2                (CORE QUESTION)
  P4  e2e         end-to-end DenseNet, HER2                      (collaborator's ask)
  P5  AL-frozen   AL benchmark on BC + Kidney                    (replication)
  P6  e2e         end-to-end DenseNet, BC + Kidney               (full ceilings)
  P7  target      expression_norm vs hvg300 on HER2              (which target)

Run:  conda run -n st-select python -u scripts/run_pg.py
"""

import subprocess
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

OUT = ROOT / "results_pg"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "RUN_PG.md"
CACHE = ROOT / "feature_cache"

AL8 = ["random", "entropy", "tod", "coreset", "typiclust", "badge",
       "poisson_disk", "spatial_stratified"]


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def base(**kw) -> dict:
    d = dict(
        feature_cache_dir="./feature_cache", split_level="patient",
        n_folds=4, n_seeds=1, lr=1e-4, loss_fn="mse",
        pcc_top_n=[10, 50, 100, 200, 300], device="cuda", seed=42, num_workers=0,
    )
    d.update(kw)
    return d


def ensure_cache(dataset: str, target: str = "hvg300") -> bool:
    """Build the DenseNet feature + PNG-path cache for one dataset if absent."""
    name = f"pg_{dataset.lower()}_{target}"
    if (CACHE / f"{name}.npz").exists():
        return True
    log(f"### BUILD {name}")
    t0 = time.time()
    cmd = [sys.executable, str(ROOT / "scripts" / "prepare_processed_gene.py"),
           "--dataset", dataset, "--target", target,
           "--feature-extractor", "densenet121", "--device", "cuda", "--batch-size", "64"]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=10800)
        ok = r.returncode == 0 and (CACHE / f"{name}.npz").exists()
        log(f"### BUILD {name} {'OK' if ok else 'FAILED'} in {(time.time()-t0)/60:.1f} min"
            + ("" if ok else f"\n{r.stderr[-1200:]}"))
        return ok
    except Exception:
        log(f"### BUILD {name} EXCEPTION\n{traceback.format_exc()}")
        return False


def summarize(res: dict) -> str:
    mg = res.get("mean_pcc_per_gene", {})
    ms = res.get("mean_pcc_per_gene_by_slide", {})
    t = res.get("pcc_topN", {})
    lines = []
    al = {m: v for m, v in ms.items() if m != "full_supervision" and v is not None}
    if len(al) > 1 and "random" in al:
        rnd = al["random"]
        best = max((m for m in al if m != "random"), key=lambda m: al[m])
        lines.append(f"      [AL] random={rnd:.4f}  best={best}={al[best]:.4f}  "
                     f"delta={al[best]-rnd:+.4f}  (per-slide)")
    for m in mg:
        s = ms.get(m)
        t100 = (t.get(m) or {}).get("100")
        lines.append(f"      {m:20s} per_gene={mg[m]:.4f} "
                     f"per_slide={s if s is not None else float('nan'):.4f} "
                     f"top100={t100 if t100 is not None else float('nan'):.4f}")
    return "\n".join(lines)


def run_job(label: str, cfg_kwargs: dict) -> None:
    name = cfg_kwargs["dataset_name"]
    done = Path(cfg_kwargs["output_dir"]) / name / "results.json"
    if done.exists():
        log(f"### SKIP  [{label}] {name} — exists")
        return
    if not (CACHE / f"{name}.npz").exists():
        log(f"### SKIP  [{label}] {name} — cache missing")
        return
    log(f"### START [{label}] {name}  methods={len(cfg_kwargs['al_methods'])} "
        f"image_mode={cfg_kwargs.get('image_mode', False)} epochs={cfg_kwargs.get('full_epochs')}")
    t0 = time.time()
    try:
        res = run_full_benchmark(ExperimentConfig(**cfg_kwargs))
        log(f"### DONE  [{label}] {name} in {(time.time()-t0)/60:.1f} min\n{summarize(res)}")
    except Exception:
        log(f"### FAILED [{label}] {name} after {(time.time()-t0)/60:.1f} min\n{traceback.format_exc()}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def frozen_cfg(ds, **kw):
    return base(dataset_name=ds, model_name="feature_predictor",
                al_methods=["full_supervision"], add_full_supervision=False,
                full_epochs=60, epochs_per_round=60, budget_ratios=[1.0],
                batch_size=128, output_dir=str(OUT / "frozen_dn"), **kw)


def al_cfg(ds, **kw):
    return base(dataset_name=ds, model_name="feature_predictor", al_methods=AL8,
                budget_ratios=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
                epochs_per_round=50, full_epochs=60,
                initial_budget=256, budget_per_round=512, batch_size=128,
                output_dir=str(OUT / "al_frozen"), **kw)


def e2e_cfg(ds, **kw):
    return base(dataset_name=ds, model_name="st_net", image_mode=True,
                al_methods=["full_supervision"], add_full_supervision=False,
                full_epochs=40, epochs_per_round=40, budget_ratios=[1.0],
                batch_size=32, output_dir=str(OUT / "e2e_dn"), **kw)


def main() -> None:
    log("=" * 70)
    log(f"RUN-PG: processed_gene phased training — {datetime.now():%Y-%m-%d %H:%M}")
    log(f"cuda={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    log("=" * 70)

    # ---- P1: caches (HER2/hvg300 already built) ----
    for ds in ["BC", "Kidney"]:
        ensure_cache(ds, "hvg300")

    # ---- P2: frozen full-supervision ceilings, all 3 datasets ----
    for ds in ["her2", "bc", "kidney"]:
        run_job("P2-frozen", frozen_cfg(f"pg_{ds}_hvg300"))

    # ---- P3: THE CORE QUESTION — does AL beat random when signal exists? ----
    run_job("P3-AL-her2", al_cfg("pg_her2_hvg300"))

    # ---- P4: end-to-end DenseNet on HER2 (collaborator's explicit ask) ----
    run_job("P4-e2e-her2", e2e_cfg("pg_her2_hvg300"))

    # ---- P5: AL replication on the other two datasets ----
    for ds in ["bc", "kidney"]:
        run_job(f"P5-AL-{ds}", al_cfg(f"pg_{ds}_hvg300"))

    # ---- P6: end-to-end ceilings on the other two ----
    for ds in ["bc", "kidney"]:
        run_job(f"P6-e2e-{ds}", e2e_cfg(f"pg_{ds}_hvg300"))

    # ---- P7: which target? expression_norm vs hvg300 (HER2) ----
    if ensure_cache("HER2", "norm"):
        run_job("P7-target-norm", frozen_cfg("pg_her2_norm"))

    log("ALL PG PHASES COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
