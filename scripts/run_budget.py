"""Task #1 - controlled budget re-run.

Why this exists
---------------
The new headline claim is a budget claim: "X% of the labels matches 100%".
Every budget curve we have so far was produced with a FIXED epoch count
(epochs_per_round=50), which makes that claim unfalsifiable in the wrong
direction: a bigger labeled pool also means more gradient steps per round, so
more labels buy both more information AND more overfitting to the training
patients. On pg_her2 the two effects cancelled and then inverted the curve
(random: 5% = 0.2936 > 100% = 0.2597 per-slide PCC). Read literally that says
"labeling less is better", which nobody believes; it says the axis is confounded.

The fix is NOT fixed gradient steps - that just moves the confound (a small pool
seeing the same number of steps overfits harder per sample). It is early
stopping, so each budget trains to *its own* optimum and the curve compares
converged models. The holdout is group-aware (whole patients, see
training.loop.group_aware_val_split): the overfitting we need to stop on is
patient-level, and a random row-level holdout shares the training patients'
batch effects, so its loss keeps falling while test PCC decays.

Arms
----
  --arm legacy      epochs=50,  val_frac=0     reproduces the confound, + a 75% point
  --arm controlled  epochs=200, val_frac=0.15  early-stopped, the honest curve

The gate before spending a full night: run the pilot (HER2, random, 1 seed, both
arms) and check that the controlled curve is monotone-ish. If 5% still beats
100% after early stopping, the inversion is real and the write-up changes rather
than the runs.

  conda run -n st-select python -u scripts/run_budget.py --pilot
  conda run -n st-select python -u scripts/run_budget.py --arm controlled
  conda run -n st-select python -u scripts/run_budget.py --arm legacy

Runs strictly serially and one lane at a time. Frozen-feature AL still trains on
the GPU in hundreds of short bursts (37-42% util), and pairing that with a
100%-util job cost us a measured ~6x slowdown once already (EXPERIMENTS.md S12).
"""

import argparse
import json
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

OUT = ROOT / "results_pg" / "budget"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "feature_cache"
LOG = OUT / "RUN_BUDGET.md"

DATASETS = ["pg_her2_hvg300", "pg_bc_hvg300", "pg_kidney_hvg300"]

# 0.75 is new: without it there is a 2x gap between 0.5 and 1.0, which is where
# the Kidney curve was still climbing, so "50% is enough" had no support above it.
RATIOS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

# random only, deliberately. The claim under test is a BUDGET claim ("X% of the
# labels is enough"), and random is what a lab without an AL implementation
# actually does, so it is the honest reference. Adding strategy arms triples the
# cost to re-answer a question the 20-strategy sweep already answered: on this
# data no strategy is consistently better than random - the sign flips between
# datasets (cald +0.0042 on her2, -0.0032 on kidney; maxherding +0.0037/-0.0038),
# with |delta| <= 0.004 (~1.6% relative) throughout. Pass --methods to add arms.
METHODS = ["random"]

ARMS = {
    "legacy":     dict(epochs_per_round=50,  val_frac=0.0),
    "controlled": dict(epochs_per_round=200, val_frac=0.15, patience=10),
}


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def cfg(dataset: str, arm: str, methods, n_seeds: int, tag: str) -> ExperimentConfig:
    kw = dict(
        dataset_name=dataset, model_name="feature_predictor",
        feature_cache_dir="./feature_cache", output_dir=str(OUT / tag),
        al_methods=list(methods), budget_ratios=list(RATIOS),
        split_level="patient", n_folds=4, n_seeds=n_seeds,
        lr=1e-4, loss_fn="mse", pcc_top_n=[10, 50, 100],
        device="cuda", seed=42, num_workers=0,
        add_full_supervision=False,   # the curve's own 1.0 point is the reference
        # Matched to the 20-strategy sweep (scripts/run_final.py:al20) so the
        # legacy arm reproduces the inverted curve we are trying to explain, and
        # the two arms differ ONLY in epochs + early stopping. The config default
        # batch_size of 64 would halve the batch and thus double the gradient
        # steps per epoch, amplifying the exact confound under study.
        batch_size=128, initial_budget=256, budget_per_round=512,
    )
    kw.update(ARMS[arm])
    return ExperimentConfig(**kw)


def curve(res: dict, method: str, key: str = "by_ratio_by_slide"):
    """Mean per-slide PCC at each budget ratio, as [(ratio, pcc, n_runs), ...].

    Shape is {method: {"0.05": [one value per (fold, seed)]}}, so each point is a
    mean over 4 folds x n_seeds.
    """
    per_ratio = (res.get(key, {}) or {}).get(method, {})
    out = []
    for r in RATIOS:
        vals = [v for v in per_ratio.get(f"{r:.2f}", []) if v is not None]
        if vals:
            out.append((r, sum(vals) / len(vals), len(vals)))
    return out


def report(res: dict, methods) -> str:
    lines = []
    for m in methods:
        c = curve(res, m)
        if not c:
            continue
        full = c[-1][1]
        pad = " " * len(m)
        lines.append(f"      {m}: " + "  ".join(f"{r:.0%}={v:.4f}" for r, v, _ in c)
                     + f"   (n={c[0][2]} runs/point)")
        if full:
            lines.append(f"      {pad}  vs-100%: "
                         + "  ".join(f"{r:.0%}={v / full:.0%}" for r, v, _ in c))
            enough = [r for r, v, _ in c if v >= 0.95 * full]
            if enough:
                lines.append(f"      {pad}  >=95% of full at: {min(enough):.0%}")
        peak = max(c, key=lambda t: t[1])
        if peak[0] < 1.0:
            lines.append(f"      {pad}  NOTE peak at {peak[0]:.0%} ({peak[1]:.4f}), "
                         f"not at 100% ({full:.4f}) - curve still non-monotone")
    return "\n".join(lines) or "      (no per-slide numbers found)"


def per_run_curves(res: dict, method: str):
    """[(ratio, [pcc for each (fold,seed)])] - the unaggregated per-slide series.

    Kept unaggregated because the folds are wildly heterogeneous on this data
    (HER2 fold 2 scores ~0.06 against ~0.44 for folds 0-1, a capacity limit not a
    hard patient). Averaging first and *then* reading off "the budget where the
    mean reaches 95% of full" would let one fold's shape stand in for all of them,
    so saturation is computed per fold and the spread reported.
    """
    per_ratio = (res.get("by_ratio_by_slide", {}) or {}).get(method, {})
    return [(r, [v for v in per_ratio.get(f"{r:.2f}", []) if v is not None])
            for r in RATIOS]


def saturation(res: dict, method: str, thresh: float = 0.95):
    """Per-run smallest budget reaching `thresh` x that run's own 100% score."""
    series = per_run_curves(res, method)
    if not series or not series[-1][1]:
        return []
    n = min((len(v) for _, v in series if v), default=0)
    out = []
    for i in range(n):
        full = series[-1][1][i]
        if full is None or full <= 0:
            continue
        hit = [r for r, v in series if i < len(v) and v[i] >= thresh * full]
        out.append(min(hit) if hit else 1.0)
    return out


def epochs_by_ratio(res: dict, method: str):
    """[(ratio, mean epochs actually trained)] - read straight from `detailed`.

    A flat line means the arm trained a fixed number of epochs regardless of pool
    size (the confound). A line that rises with budget means early stopping let
    bigger pools train longer, which is what "each budget trains to its own
    optimum" should look like.
    """
    runs = (res.get("detailed", {}) or {}).get(method, [])
    acc = {}
    for run in runs:
        for fr, ep in zip(run.get("frac", []), run.get("epochs_run", [])):
            r = min(RATIOS, key=lambda x: abs(x - fr))
            acc.setdefault(r, []).append(ep)
    return [(r, sum(acc[r]) / len(acc[r])) for r in RATIOS if acc.get(r)]


def spearman(xs, ys) -> float:
    """Rank correlation of budget vs score: ~1.0 if the curve is monotone up."""
    def rank(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        rk = [0.0] * len(a)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def compare(dataset: str, method: str = "random") -> None:
    """Print the legacy vs controlled curves side by side."""
    log(f"--- BEFORE/AFTER  {dataset}  method={method} ---")
    arms = {}
    for tag, label in (("pilot_legacy", "legacy (50 ep, no val)"),
                       ("pilot_controlled", "controlled (early stop)"),
                       ("legacy", "legacy full"), ("controlled", "controlled full")):
        p = OUT / tag / dataset / "results.json"
        if p.exists():
            arms[label] = json.loads(p.read_text(encoding="utf-8"))
    if not arms:
        log("      (no arms found yet)")
        return
    for label, res in arms.items():
        c = curve(res, method)
        if not c:
            continue
        ratios = [r for r, _, _ in c]
        means = [v for _, v, _ in c]
        full = means[-1]
        rho = spearman(ratios, means)
        peak_r, peak_v = max(zip(ratios, means), key=lambda t: t[1])
        sat = saturation(res, method)
        log(f"      {label}")
        log(f"        curve   : " + "  ".join(f"{r:.0%}={v:.4f}" for r, v in zip(ratios, means)))
        log(f"        vs 100% : " + "  ".join(f"{r:.0%}={v / full:.0%}" for r, v in zip(ratios, means)))
        log(f"        monotone: spearman(budget, pcc) = {rho:+.3f}   "
            f"peak at {peak_r:.0%}" + ("  <-- STILL INVERTED" if peak_r < 1.0 else ""))
        if sat:
            log(f"        per-run budget reaching 95% of its own 100%: "
                + ", ".join(f"{s:.0%}" for s in sat)
                + f"   (median {sorted(sat)[len(sat) // 2]:.0%})")
        ep = epochs_by_ratio(res, method)
        if ep:
            log(f"        epochs run: "
                + "  ".join(f"{r:.0%}={e:.0f}" for r, e in ep)
                + "   (flat = fixed-epoch arm; rising with budget = the confound)")


def run_job(tag: str, dataset: str, arm: str, methods, n_seeds: int) -> dict:
    done = OUT / tag / dataset / "results.json"
    if done.exists():
        log(f"### SKIP  [{tag}] {dataset} - exists")
        return json.loads(done.read_text(encoding="utf-8"))
    if not (CACHE / f"{dataset}.npz").exists():
        log(f"### SKIP  [{tag}] {dataset} - cache missing")
        return {}
    log(f"### START [{tag}] {dataset} arm={arm} methods={list(methods)} "
        f"seeds={n_seeds} epochs={ARMS[arm]['epochs_per_round']} "
        f"val_frac={ARMS[arm]['val_frac']}")
    t0 = time.time()
    try:
        res = run_full_benchmark(cfg(dataset, arm, methods, n_seeds, tag))
        log(f"### DONE  [{tag}] {dataset} in {(time.time() - t0) / 60:.1f} min\n"
            f"{report(res, methods)}")
        return res
    except Exception:
        log(f"### FAILED [{tag}] {dataset} after {(time.time() - t0) / 60:.1f} min\n"
            f"{traceback.format_exc()}")
        return {}
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="HER2 + random + 1 seed, both arms: is the curve monotone "
                         "once training is controlled? Gate for the full run.")
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--compare", action="store_true",
                    help="Re-print the before/after tables from results on disk")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    if args.compare:
        for ds in args.datasets:
            compare(ds, args.methods[0])
        return

    if args.pilot:
        log("=== PILOT: does early stopping restore a monotone budget curve? ===")
        for arm in ("legacy", "controlled"):
            run_job(f"pilot_{arm}", "pg_her2_hvg300", arm, ["random"], 1)
        compare("pg_her2_hvg300", "random")
        log("=== PILOT done - if the controlled curve still peaks below 100%, the "
            "inversion is real and the write-up changes, not the runs ===")
        return

    if not args.arm:
        ap.error("pass --pilot or --arm {legacy,controlled}")

    log(f"=== FULL [{args.arm}] datasets={args.datasets} methods={args.methods} "
        f"seeds={args.seeds} ===")
    for ds in args.datasets:
        run_job(args.arm, ds, args.arm, args.methods, args.seeds)
    log(f"=== FULL [{args.arm}] complete ===")


if __name__ == "__main__":
    main()
