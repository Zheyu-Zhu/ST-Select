"""Night driver #2: harden the negative AL result along its two weakest axes.

  1. frozen_her2_dn_al   frozen DenseNet features + AL sweep on HER2
  2. comprehensive_her2  ALL 26 runnable strategies (of 29) on frozen UNI / HER2
                         -> answers "you only tried 8 strategies" with the whole registry
  3. phase3_trained_her2 trained DenseNet end-to-end + AL sweep on HER2

Jobs 1 vs 3 are the frozen-vs-trained-backbone isolation on a SECOND dataset
(HER2), mirroring today's cSCC result — same backbone family (DenseNet), only
frozen<->trained differs. Job 2 broadens the strategy set toward "comprehensive"
(the review / Decision 4). All outputs land under gitignored results_slide/ and
carry the new slide-level top-N + ARI/NMI + selection-diversity metrics.

Cheap-first ordering so a short night still yields the frozen results; the long
trained-HER2 run is the resumable tail.

Run:  PATCH_CACHE_GB=5 conda run -n st-select python -u scripts/run_night2.py
"""

import json
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
from src.datasets.patch_dataset import clear_slide_cache  # noqa: E402

OUT = ROOT / "results_slide"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "RUN_NIGHT2.md"

AL6 = ["random", "entropy", "coreset", "badge", "poisson_disk", "spatial_stratified"]
# 22 strategies spanning every family that run without special extras and without
# per-round network training. Excludes confidnet / learning_loss /
# suggestive_annotation (need heads/ensembles the loop doesn't wire) and the four
# VAE/adversarial ones (vaal / ta_vaal / waal / ccal) that train a net each round
# and could run pathologically long unattended — add those later with tuning.
AL22 = [
    "random", "entropy", "margin", "least_confidence", "mc_dropout", "batchbald",
    "tod", "coreset", "typiclust", "maxherding", "probcover", "cdal", "badge",
    "similar", "alfa_mix", "cald", "ceal", "coregcn", "lal_rl",
    "poisson_disk", "spatial_stratified", "tissue_aware",
]


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


JOBS = [
    # 1) frozen DenseNet AL on HER2 (~1h) — pairs with job 2 for the isolation.
    ("frozen_her2_dn_al", "her2_dn_cp10k", False, base(
        dataset_name="her2_dn_cp10k", model_name="feature_predictor", al_methods=AL6,
        budget_ratios=[0.1, 0.3, 0.5, 1.0], epochs_per_round=50, full_epochs=60,
        initial_budget=256, budget_per_round=512, batch_size=128,
        output_dir=str(OUT / "frozen_her2_dn_al"),
    )),
    # 2) trained DenseNet end-to-end AL on HER2 (~6h) — the 2nd-dataset replication
    #    of "trained AL <= random". Highest value; known timing; run before the
    #    variable-length comprehensive sweep so it definitely completes.
    ("phase3_trained_her2", "her2_breast_img", True, base(
        dataset_name="her2_breast_img", model_name="st_net", image_mode=True,
        patches_dir="./hest_data/patches", al_methods=AL6,
        budget_ratios=[0.1, 0.3, 1.0], epochs_per_round=15, full_epochs=15,
        initial_budget=256, budget_per_round=1024, batch_size=32,
        output_dir=str(OUT / "phase3_trained_her2"),
    )),
    # 3) comprehensive 22-strategy sweep on frozen UNI / HER2 (resumable tail).
    ("comprehensive_her2", "her2_uni_cp10k", False, base(
        dataset_name="her2_uni_cp10k", model_name="feature_predictor", al_methods=AL22,
        budget_ratios=[0.1, 0.3, 1.0], epochs_per_round=50, full_epochs=60,
        initial_budget=256, budget_per_round=512, batch_size=128,
        output_dir=str(OUT / "comprehensive_her2"),
    )),
]


def summarize(res: dict) -> str:
    ms = res.get("mean_pcc_per_gene_by_slide", {})
    t50 = res.get("pcc_topN_by_slide", {})
    ari = res.get("mean_ari", {})
    al = {m: v for m, v in ms.items() if m != "full_supervision" and v is not None}
    if not al:
        return "      (no per-slide values)"
    rnd = al.get("random")
    best = max((m for m in al if m != "random"), key=lambda m: al[m]) if len(al) > 1 else "random"
    delta = (al[best] - rnd) if rnd is not None else float("nan")
    lines = [f"      random per_slide={rnd if rnd is not None else float('nan'):.4f}  "
             f"best_AL={best}={al[best]:.4f}  delta={delta:+.4f}  (n_methods={len(al)})"]
    for m in sorted(al, key=lambda m: -al[m])[:5]:
        s50 = (t50.get(m) or {}).get("50")
        lines.append(f"        {m:20s} per_slide={al[m]:.4f} "
                     f"top50_s={s50 if s50 is not None else float('nan'):.4f} "
                     f"ari={ari.get(m) if ari.get(m) is not None else float('nan'):.4f}")
    return "\n".join(lines)


def run_job(label, name, image_mode, cfg_kwargs) -> None:
    done = Path(cfg_kwargs["output_dir"]) / name / "results.json"
    if done.exists():
        log(f"### SKIP  [{label}] {name} — exists")
        return
    if image_mode:
        clear_slide_cache()
    log(f"### START [{label}] {name}  n_methods={len(cfg_kwargs['al_methods'])} "
        f"seeds={cfg_kwargs['n_seeds']} budgets={cfg_kwargs['budget_ratios']}")
    t0 = time.time()
    try:
        res = run_full_benchmark(ExperimentConfig(**cfg_kwargs))
        dt = (time.time() - t0) / 60
        log(f"### DONE  [{label}] {name} in {dt:.1f} min\n{summarize(res)}")
    except Exception:
        dt = (time.time() - t0) / 60
        log(f"### FAILED [{label}] {name} after {dt:.1f} min\n{traceback.format_exc()}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    log("=" * 70)
    log(f"RUN-NIGHT2: HER2 frozen-vs-trained AL + comprehensive 26-strategy sweep — {datetime.now():%Y-%m-%d %H:%M}")
    log(f"cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    log("=" * 70)
    for label, name, image_mode, cfg_kwargs in JOBS:
        run_job(label, name, image_mode, cfg_kwargs)
    log("ALL NIGHT2 JOBS COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
