"""Marathon driver: finish everything that can be completed safely + unattended.

Runs AFTER run_night2.py (which covers all the HER2 work). Fills the remaining
gaps toward a complete, comprehensive benchmark, cheap-first and resumable:

  M1  comprehensive 22-strategy sweep on cSCC   (frozen DenseNet, cscc_dn)
  M2  8-method AL benchmark on cSCC, 3 seeds     (frozen DenseNet, new metrics)
  M3  8-method AL benchmark on He-breast, 3 seeds(builds he_breast_dn first)
  M4  seeds on the trained-cSCC Phase 3 result   (seeds 43,44; the long tail)

Backbone note: the negative AL result is backbone-invariant (already shown under
frozen UNI, frozen DenseNet, and trained DenseNet), so using DenseNet frozen
features for the cSCC/He-breast breadth is fine and strengthens "holds regardless
of backbone." HER2 (night2) used UNI; this covers cSCC + He-breast under DenseNet.

NOT included (need a supervised session, not safe unattended): DLPFC / crc-HD
data acquisition (gated download + unknown format), a diffusion baseline (new
model code), and the 4 VAE/adversarial strategies (per-round net training).

Run:  PATCH_CACHE_GB=5 conda run -n st-select python -u scripts/run_marathon.py
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
from src.datasets.patch_dataset import clear_slide_cache  # noqa: E402

OUT = ROOT / "results_slide"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "RUN_MARATHON.md"
CACHE = ROOT / "feature_cache"

AL8 = ["random", "entropy", "tod", "coreset", "typiclust", "badge",
       "poisson_disk", "spatial_stratified"]
# 20-strategy lean set: the runnable-without-extras strategies MINUS batchbald
# and cdal, which are O(N^2)/greedy-loop slow at scale (batchbald = 20.8s, cdal =
# 6.7s at N=2000; they were the 5h hang in night2). The remaining 20 span every
# family and each runs <2.2s at N=2000.
AL_LEAN = [
    "random", "entropy", "margin", "least_confidence", "mc_dropout",
    "tod", "coreset", "typiclust", "maxherding", "probcover", "badge",
    "similar", "alfa_mix", "cald", "ceal", "coregcn", "lal_rl",
    "poisson_disk", "spatial_stratified", "tissue_aware",
]
HE_IDS = [f"SPA{i}" for i in range(51, 119)]  # He-breast slides


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


def ensure_he_breast_dn() -> bool:
    """Build the frozen DenseNet He-breast cache if absent. Returns True on success."""
    if (CACHE / "he_breast_dn.npz").exists():
        return True
    log("### BUILD he_breast_dn (frozen DenseNet features, ~40 min)")
    cmd = [sys.executable, str(ROOT / "scripts" / "prepare_hest_cache.py"),
           "--dataset", "he_breast_dn", "--sample-ids", *HE_IDS,
           "--patient-map", str(ROOT / "hest_data" / "HEST_v1_3_0.csv"),
           "--feature-extractor", "densenet121", "--device", "cuda", "--batch-size", "64"]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=7200)
        if r.returncode == 0 and (CACHE / "he_breast_dn.npz").exists():
            log("### BUILD he_breast_dn OK")
            return True
        log(f"### BUILD he_breast_dn FAILED rc={r.returncode}\n{r.stderr[-1500:]}")
        return False
    except Exception:
        log(f"### BUILD he_breast_dn EXCEPTION\n{traceback.format_exc()}")
        return False


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


def run_job(label, cfg_kwargs, image_mode=False) -> None:
    name = cfg_kwargs["dataset_name"]
    done = Path(cfg_kwargs["output_dir"]) / name / "results.json"
    if done.exists():
        log(f"### SKIP  [{label}] {name} — exists")
        return
    if image_mode:
        clear_slide_cache()
    log(f"### START [{label}] {name}  n_methods={len(cfg_kwargs['al_methods'])} "
        f"seeds={cfg_kwargs['n_seeds']} seed0={cfg_kwargs['seed']} budgets={cfg_kwargs['budget_ratios']}")
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=["all", "cpu", "gpu"], default="all",
                    help="cpu = frozen sweeps M0-M3 (CPU-bound); gpu = DenseNet cache build + "
                         "trained-cSCC M4 (GPU-bound); run both in parallel to use CPU+GPU at once.")
    args = ap.parse_args()
    global LOG
    if args.lane != "all":
        LOG = OUT / f"RUN_MARATHON_{args.lane}.md"   # per-lane log, no interleaving
    log("=" * 70)
    log(f"RUN-MARATHON [lane={args.lane}] — {datetime.now():%Y-%m-%d %H:%M}")
    log(f"cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    log("=" * 70)

    def m0():  # comprehensive 20-strategy sweep on HER2 (frozen UNI)
        run_job("comprehensive_her2", base(
            dataset_name="her2_uni_cp10k", model_name="feature_predictor", al_methods=AL_LEAN,
            budget_ratios=[0.1, 0.3, 1.0], epochs_per_round=50, full_epochs=60,
            initial_budget=256, budget_per_round=512, batch_size=128,
            output_dir=str(OUT / "comprehensive_her2")))

    def m1():  # comprehensive 20-strategy sweep on cSCC (frozen DenseNet)
        run_job("comprehensive_cscc", base(
            dataset_name="cscc_dn", model_name="feature_predictor", al_methods=AL_LEAN,
            budget_ratios=[0.1, 0.3, 1.0], epochs_per_round=50, full_epochs=60,
            initial_budget=256, budget_per_round=512, batch_size=128,
            output_dir=str(OUT / "comprehensive_cscc")))

    def m2():  # 8-method AL on cSCC, 3 seeds
        run_job("cscc_al_3seed", base(
            dataset_name="cscc_dn", model_name="feature_predictor", al_methods=AL8, n_seeds=3,
            budget_ratios=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0], epochs_per_round=50, full_epochs=60,
            initial_budget=256, budget_per_round=512, batch_size=128,
            output_dir=str(OUT / "cscc_al_3seed")))

    def m3():  # 8-method AL on He-breast, 3 seeds (cache built by the GPU lane)
        if (CACHE / "he_breast_dn.npz").exists():
            run_job("he_al_3seed", base(
                dataset_name="he_breast_dn", model_name="feature_predictor", al_methods=AL8, n_seeds=3,
                budget_ratios=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0], epochs_per_round=50, full_epochs=60,
                initial_budget=256, budget_per_round=512, batch_size=128,
                output_dir=str(OUT / "he_al_3seed")))
        else:
            log("### SKIP he_al_3seed — he_breast_dn cache not present (GPU lane builds it)")

    def m4():  # seeds 43,44 on trained-cSCC Phase 3 (GPU-heavy, image mode)
        run_job("phase3_cscc_seeds", base(
            dataset_name="cscc_img", model_name="st_net", image_mode=True,
            patches_dir="./hest_data/patches",
            al_methods=["random", "entropy", "coreset", "badge", "poisson_disk", "spatial_stratified"],
            seed=43, n_seeds=2, budget_ratios=[0.1, 0.3, 1.0], epochs_per_round=15, full_epochs=15,
            initial_budget=256, budget_per_round=1024, batch_size=32,
            output_dir=str(OUT / "phase3_cscc_seeds")), image_mode=True)

    if args.lane == "gpu":
        ensure_he_breast_dn()   # build the DenseNet cache the CPU lane's M3 needs
        m4()
    elif args.lane == "cpu":
        m0(); m1(); m2(); m3()
    else:  # all — original serial order (single GPU)
        m0(); m1(); m2()
        ensure_he_breast_dn(); m3()
        m4()

    log(f"MARATHON LANE [{args.lane}] COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
