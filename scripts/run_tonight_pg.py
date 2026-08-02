"""Finish the processed_gene program overnight, in two concurrent lanes.

The GPU work (end-to-end DenseNet over PNG patches) and the CPU work (frozen-
feature AL sweeps, whose cost is the O(N^2) acquisition, not the tiny MLP) use
different resources, so they run as separate processes:

  --lane gpu   e2e DenseNet · BC  -> e2e DenseNet · Kidney     (~9 h, the bottleneck)
  --lane cpu   AL · BC -> AL · Kidney -> target norm -> UNI    (~4.5 h)

e2e epochs stay at 40 to match the already-finished HER2 run, so the three
ceilings are directly comparable. (Note: HER2 e2e at 1 epoch scored 0.400 vs
0.393 at 40 — this task converges almost immediately, so 40 epochs is generous;
kept only for comparability.)

Run both:
  conda run -n st-select python -u scripts/run_tonight_pg.py --lane gpu
  conda run -n st-select python -u scripts/run_tonight_pg.py --lane cpu
"""

import argparse
import subprocess
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

OUT = ROOT / "results_pg"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "feature_cache"

AL8 = ["random", "entropy", "tod", "coreset", "typiclust", "badge",
       "poisson_disk", "spatial_stratified"]
LOG = OUT / "RUN_TONIGHT.md"


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def base(**kw) -> dict:
    d = dict(feature_cache_dir="./feature_cache", split_level="patient",
             n_folds=4, n_seeds=1, lr=1e-4, loss_fn="mse",
             pcc_top_n=[10, 50, 100, 200, 300], device="cuda", seed=42, num_workers=0)
    d.update(kw)
    return d


def summarize(res: dict) -> str:
    mg = res.get("mean_pcc_per_gene", {})
    ms = res.get("mean_pcc_per_gene_by_slide", {})
    t = res.get("pcc_topN", {})
    lines = []
    al = {m: v for m, v in ms.items() if m != "full_supervision" and v is not None}
    if len(al) > 1 and "random" in al:
        rnd = al["random"]
        best = max((m for m in al if m != "random"), key=lambda m: al[m])
        lines.append(f"      [AL] random={rnd:.4f} best={best}={al[best]:.4f} "
                     f"delta={al[best]-rnd:+.4f} (per-slide)")
    for m in mg:
        s = ms.get(m)
        t100 = (t.get(m) or {}).get("100")
        lines.append(f"      {m:20s} per_gene={mg[m]:.4f} "
                     f"per_slide={s if s is not None else float('nan'):.4f} "
                     f"top100={t100 if t100 is not None else float('nan'):.4f}")
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
    log(f"### START [{label}] {name} image_mode={cfg.get('image_mode', False)} "
        f"methods={len(cfg['al_methods'])} epochs={cfg.get('full_epochs')}")
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


def build_cache(dataset: str, target: str, extractor: str) -> bool:
    name = f"pg_{dataset.lower()}_{target}" if extractor == "densenet121" \
        else f"pg_{dataset.lower()}_{target}_uni"
    if (CACHE / f"{name}.npz").exists():
        return True
    log(f"### BUILD {name} ({extractor})")
    t0 = time.time()
    cmd = [sys.executable, str(ROOT / "scripts" / "prepare_processed_gene.py"),
           "--dataset", dataset, "--target", target,
           "--feature-extractor", extractor, "--device", "cuda", "--batch-size", "64"]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=10800)
        ok = r.returncode == 0
        log(f"### BUILD {name} {'OK' if ok else 'FAILED'} in {(time.time()-t0)/60:.1f} min"
            + ("" if ok else f"\n{r.stderr[-1000:]}"))
        return ok
    except Exception:
        log(f"### BUILD {name} EXCEPTION\n{traceback.format_exc()}")
        return False


def e2e(ds):
    return base(dataset_name=f"pg_{ds}_hvg300", model_name="st_net", image_mode=True,
                al_methods=["full_supervision"], add_full_supervision=False,
                full_epochs=40, epochs_per_round=40, budget_ratios=[1.0],
                batch_size=32, output_dir=str(OUT / "e2e_dn"))


def al(ds):
    return base(dataset_name=f"pg_{ds}_hvg300", model_name="feature_predictor",
                al_methods=AL8, budget_ratios=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
                epochs_per_round=50, full_epochs=60, initial_budget=256,
                budget_per_round=512, batch_size=128, output_dir=str(OUT / "al_frozen"))


def frozen(name, outdir):
    return base(dataset_name=name, model_name="feature_predictor",
                al_methods=["full_supervision"], add_full_supervision=False,
                full_epochs=60, epochs_per_round=60, budget_ratios=[1.0],
                batch_size=128, output_dir=str(OUT / outdir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=["gpu", "cpu", "uni"], required=True)
    args = ap.parse_args()
    global LOG
    LOG = OUT / f"RUN_TONIGHT_{args.lane}.md"

    log("=" * 70)
    log(f"RUN-TONIGHT-PG [lane={args.lane}] — {datetime.now():%Y-%m-%d %H:%M}")
    log("=" * 70)

    if args.lane == "uni":
        # Frozen UNI on the remaining datasets, to complete the backbone column.
        for ds in ["BC", "Kidney"]:
            if build_cache(ds, "hvg300", "uni"):
                run_job(f"uni-{ds.lower()}",
                        frozen(f"pg_{ds.lower()}_hvg300_uni", "frozen_uni"))
    elif args.lane == "gpu":
        # End-to-end ceilings: the long pole. BC first (bigger), then Kidney.
        run_job("e2e-bc", e2e("bc"))
        run_job("e2e-kidney", e2e("kidney"))
    else:
        # AL sweeps + the two remaining comparisons.
        run_job("AL-bc", al("bc"))
        run_job("AL-kidney", al("kidney"))
        # Which target: expression_norm (~950 genes, CP-normalized) vs hvg300.
        if build_cache("HER2", "norm", "densenet121"):
            run_job("target-norm", frozen("pg_her2_norm", "frozen_dn"))
        # UNI backbone on the new data, so the new arm is comparable to the
        # original benchmark (which used frozen UNI, not DenseNet).
        if build_cache("HER2", "hvg300", "uni"):
            run_job("uni-her2", frozen("pg_her2_hvg300_uni", "frozen_uni"))

    log(f"LANE [{args.lane}] COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
