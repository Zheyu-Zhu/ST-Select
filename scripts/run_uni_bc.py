"""UNI fine-tuned end-to-end on BC — the last empty cell in the model ladder.

HER2 and Kidney have UNI e2e; BC never ran, so the ladder reads "not run" and BC
has no UNI row in the top-N table. This fills both.

Two fixes carried in, so the gaps this run exists to close cannot recur:

  pcc_top_n = [10, 50, 100, 200, 300]
      The earlier UNI e2e runs used [10, 50, 100], which is why their top-200 and
      top-300 cells are empty. (top-300 turned out to be recoverable for free —
      on a 300-gene panel it is by definition the whole panel, i.e. exactly the
      per-slide PCC; verified identical to 4 dp on 6 independent runs. top-200 is
      genuinely lost without a re-run.)

  save_models_dir = <out>/models
      The earlier runs saved nothing, so re-deriving any additional metric means
      retraining from scratch — 250 min for HER2, 475 for Kidney. With the
      checkpoint on disk, a new cutoff or a new downstream metric is a re-eval,
      not a re-train.

BC is the largest cohort (30 655 spots, 22 320 in a training fold) and the hardest
(~0.13 per-slide under every backbone tried so far — a property of the data, not
the method). Expect this to run longer than Kidney's 475 min; the log records the
measured time rather than an estimate.

    conda run -n st-select python -u scripts/run_uni_bc.py
"""

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

OUT = ROOT / "results_pg" / "final" / "uni_e2e"
LOG = OUT / "RUN_UNI_BC.md"
DATASET = "pg_bc_hvg300"


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    done = OUT / DATASET / "results.json"
    if done.exists():
        log(f"### SKIP {DATASET} — already exists at {done}")
        return

    cfg = ExperimentConfig(
        dataset_name=DATASET, model_name="uni_regressor", image_mode=True,
        feature_cache_dir="./feature_cache", output_dir=str(OUT),
        al_methods=["full_supervision"], add_full_supervision=False,
        split_level="patient", n_folds=4, n_seeds=1,
        # Identical recipe to the HER2/Kidney UNI e2e ceilings so the ladder row
        # is comparable: 15 epochs, bs 16, backbone at 1/10th the head's LR.
        epochs_per_round=15, full_epochs=15, batch_size=16,
        backbone_lr_mult=0.1, lr=1e-4, loss_fn="mse",
        budget_ratios=[1.0],
        pcc_top_n=[10, 50, 100, 200, 300],
        save_models_dir=str(OUT / "models"),
        device="cuda", seed=42, num_workers=0,
    )

    log("=" * 68)
    log(f"UNI e2e · BC — filling the last cell of the model ladder")
    log(f"cutoffs={cfg.pcc_top_n}  save_models_dir={cfg.save_models_dir}")
    log("=" * 68)

    t0 = time.time()
    try:
        res = run_full_benchmark(cfg)
        mins = (time.time() - t0) / 60
        m = "full_supervision"
        ps = res.get("mean_pcc_per_gene_by_slide", {}).get(m)
        pg = res.get("mean_pcc_per_gene", {}).get(m)
        tn = res.get("pcc_topN_by_slide", {}).get(m, {})
        log(f"### DONE {DATASET} in {mins:.1f} min")
        log(f"      per-slide={ps:.4f}  per-gene={pg:.4f}")
        log("      top-N (per-slide): "
            + "  ".join(f"top{k}={tn[k]:.3f}" for k in sorted(tn, key=int)))
        sp = (res.get("spatial_autocorrelation", {}) or {}).get(m, {})
        if sp:
            log(f"      moran pred={sp.get('moran_pred', float('nan')):.3f} "
                f"fidelity={sp.get('moran_fidelity', float('nan')):.3f}")
    except Exception:
        log(f"### FAILED after {(time.time() - t0) / 60:.1f} min\n{traceback.format_exc()}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
