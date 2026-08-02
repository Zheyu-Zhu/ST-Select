"""Sequential job queue for the Tier-1 and Tier-2 runs.

Runs every job one after another, never in parallel. That is not a stylistic
choice: frozen-feature jobs train on the GPU in hundreds of short bursts, and
pairing one with a job that pins the card (UNI ViT-L) cost a measured ~6x
slowdown once — 3 h of work took over 18 h (EXPERIMENTS.md S12).

Properties that matter for a ~2-day unattended run:

  * **Resumable.** Every job declares the file it produces. Re-running the queue
    skips whatever already exists, so a crash or a reboot costs one job, not the
    queue. Safe to launch repeatedly.
  * **Failure-isolated.** Each job is a separate process. A crash is logged with
    its traceback and the queue moves to the next job rather than dying.
  * **Ordered by value.** Cheap jobs that close a real hole run first, so if the
    queue is stopped early the most useful results are already in.
  * **Honest about time.** Estimates below are scaled from *measured* runs in
    this project, not guessed. They have still been wrong before — treat the
    total as a rough planning figure and read RUN_QUEUE.md for actuals.

    python scripts/run_queue.py            # run everything not yet done
    python scripts/run_queue.py --list     # show the plan and exit
    python scripts/run_queue.py --only t1  # just the cheap tier
"""

import argparse
import atexit
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG = ROOT / "results_pg" / "RUN_QUEUE.md"
RES = ROOT / "results_pg" / "resources.csv"

# (tier, name, produces, est_min, argv, why)
JOBS = [
    ("t1", "celltype-budget",
     "results_pg/budget/controlled/pg_her2_celltype/results.json", 35,
     [PY, "-u", "scripts/run_budget.py", "--arm", "controlled", "--seeds", "3",
      "--datasets", "pg_her2_celltype", "pg_bc_celltype", "pg_kidney_celltype"],
     "Does the cell-type task saturate at the same budget as expression? "
     "Caches exist; scaled from the 6.6/10.9/13.6 min gene-panel runs."),

    ("t1", "breadth-vs-depth",
     "results_pg/breadth_depth/pg_bc_hvg300/results.json", 120,
     [PY, "-u", "scripts/run_breadth_depth.py"],
     "Fixed spot budget, varying patient count — the mechanism behind the "
     "5%/50%/75% spread. New experiment; estimate is the least certain here."),

    ("t2", "uni-al-her2-rngfix",
     "results_pg/uni_al/rngfix/pg_her2_hvg300/results.json", 336,
     [PY, "-u", "scripts/run_uni_al.py", "--dataset", "pg_her2_hvg300",
      "--tag", "rngfix"],
     "Re-run of E19 after the per-checkpoint reseed fix; removes the "
     "initialisation caveat. 336 min measured for the identical job."),

    ("t2", "uni-al-kidney",
     "results_pg/uni_al/pg_kidney_hvg300/results.json", 640,
     [PY, "-u", "scripts/run_uni_al.py", "--dataset", "pg_kidney_hvg300"],
     "Replicates the AL null on a 2nd cohort with the 304M model — currently "
     "HER2-only. 336 min x the 1.90 Kidney/HER2 cost ratio."),

    ("t2", "uni-budget-her2",
     "results_pg/uni_budget/pg_her2_hvg300/results.json", 750,
     [PY, "-u", "scripts/run_uni_budget.py", "--dataset", "pg_her2_hvg300"],
     "The budget curve on the best model, 7 points. Budget fractions sum to "
     "2.9 full-dataset passes x the 250 min measured ceiling."),

    ("t2", "uni-ceiling-her2",
     "results_pg/uni_ceiling/pg_her2_hvg300/results.json", 250,
     [PY, "-u", "scripts/run_uni_ceiling.py", "--dataset", "pg_her2_hvg300"],
     "Recovers the missing top-200 cell and saves weights so no future metric "
     "needs a retrain. 250 min measured."),

    ("t2", "uni-ceiling-kidney",
     "results_pg/uni_ceiling/pg_kidney_hvg300/results.json", 475,
     [PY, "-u", "scripts/run_uni_ceiling.py", "--dataset", "pg_kidney_hvg300"],
     "Same for Kidney. 475 min measured."),
]


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def pending(jobs):
    return [j for j in jobs if not (ROOT / j[2]).exists()]


def show_plan(jobs) -> None:
    todo = pending(jobs)
    done = [j for j in jobs if j not in todo]
    total = sum(j[3] for j in todo)
    print(f"\n{'':2s} {'tier':5s} {'job':22s} {'est':>7s}  status")
    print("-" * 72)
    for t, name, prod, est, _, _ in jobs:
        st = "done (skip)" if (ROOT / prod).exists() else "pending"
        print(f"   {t:5s} {name:22s} {est:5d}m  {st}")
    print("-" * 72)
    print(f"   {len(done)} already done, {len(todo)} pending")
    print(f"   estimated remaining: {total} min = {total/60:.1f} h "
          f"(~{total/60/24:.1f} days)")
    print(f"   projected finish: {(datetime.now()+timedelta(minutes=total)):%a %d %b %H:%M}")
    print("   estimates are scaled from measured runs and have been wrong before.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", choices=["t1", "t2"])
    args = ap.parse_args()

    jobs = [j for j in JOBS if not args.only or j[0] == args.only]
    if args.list:
        show_plan(jobs)
        return

    todo = pending(jobs)
    est = sum(j[3] for j in todo)
    log("=" * 72)
    log(f"QUEUE START — {len(todo)} jobs pending, estimated {est} min = {est/60:.1f} h")
    log("=" * 72)

    # Record GPU/CPU/RAM for the whole session. Sampling by hand only ever shows
    # the instant you looked; when a 10 h job ends you cannot reconstruct whether
    # it was saturated throughout, throttled, or stalled. --note tags each row
    # with the job that was running, so the CSV is attributable per job.
    res_proc = None

    def start_recorder(note):
        nonlocal res_proc
        stop_recorder()
        try:
            res_proc = subprocess.Popen(
                [PY, "-u", "scripts/log_resources.py", "--out", str(RES),
                 "--interval", "30", "--note", note],
                cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            log(f"        (resource recorder failed to start: {exc!r})")

    def stop_recorder():
        nonlocal res_proc
        if res_proc and res_proc.poll() is None:
            res_proc.terminate()
            try:
                res_proc.wait(timeout=10)
            except Exception:
                res_proc.kill()
        res_proc = None

    atexit.register(stop_recorder)

    t_all = time.time()
    results = []
    for i, (tier, name, prod, est_min, argv, why) in enumerate(jobs, 1):
        if (ROOT / prod).exists():
            log(f"[{i}/{len(jobs)}] SKIP {name} — already produced {prod}")
            continue
        log(f"[{i}/{len(jobs)}] START {name} ({tier}, est {est_min} min)")
        log(f"        {why}")
        start_recorder(name)
        t0 = time.time()
        try:
            rc = subprocess.run(argv, cwd=str(ROOT)).returncode
        except Exception as exc:                       # launcher-level failure
            rc, exc_txt = -1, repr(exc)
            log(f"        launcher error: {exc_txt}")
        mins = (time.time() - t0) / 60
        stop_recorder()
        ok = (rc == 0) and (ROOT / prod).exists()
        results.append((name, ok, mins, est_min))
        log(f"[{i}/{len(jobs)}] {'DONE ' if ok else 'FAILED'} {name} in {mins:.1f} min "
            f"(est {est_min}) rc={rc}"
            + ("" if ok else " — continuing to next job"))

    total = (time.time() - t_all) / 60
    log("=" * 72)
    stop_recorder()
    log(f"QUEUE COMPLETE — {total:.1f} min ({total/60:.1f} h)")
    log(f"    resource log: {RES}  "
        f"(python scripts/log_resources.py --summarise {RES})")
    for name, ok, mins, est_min in results:
        log(f"    {'ok  ' if ok else 'FAIL'} {name:24s} {mins:7.1f} min "
            f"(est {est_min:4d}, {mins/est_min:.1f}x)")
    bad = [n for n, ok, _, _ in results if not ok]
    if bad:
        log(f"    {len(bad)} job(s) failed: {', '.join(bad)} — re-running the "
            f"queue retries only those.")
    log("=" * 72)


if __name__ == "__main__":
    main()
