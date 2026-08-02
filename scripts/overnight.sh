#!/bin/sh
# Unattended chain for the night of 2026-07-29/30.
#
#   1. wait for the running budget job (legacy arm, Kidney) to exit
#   2. DRY-validate the image-mode UNI AL path (1 fold, 1 epoch, 2-3% budgets)
#   3. only if the dry run actually succeeded, launch the real #2 run
#
# Why the gate: image-mode AL has never been exercised with these strategies (the
# earlier UNI runs used full_supervision, which bypasses the AL loop entirely), so
# the real run could crash hours in on a head_only/backbone-shape bug. The dry run
# costs minutes and covers the same code path.
#
# Strictly serial throughout - never overlap UNI (100% GPU) with the frozen-feature
# sweep, which trains in hundreds of short GPU bursts. Measured cost of ignoring
# that once: ~6x slowdown, 3 h of work took >18 h (EXPERIMENTS.md S12).

set -u
cd /c/Users/zheyu/workspace/ST-Select
PY=/c/Users/zheyu/anaconda3/envs/st-select/python.exe
LOG=results_pg/overnight.log

say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }

say "=== overnight chain start ==="

# --- 1. wait for the in-flight budget job -----------------------------------
# NOT pgrep: it does not exist in this Git Bash, so `while pgrep ...` exits
# immediately (127) and silently skips the wait - which looks like "the job
# finished" and would start UNI on top of a running sweep. PowerShell reads the
# actual command lines instead.
running() {
  powershell -NoProfile -Command \
    "if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | \
       Where-Object { \$_.CommandLine -like '*run_budget.py*' }) { exit 0 } else { exit 1 }" \
    >/dev/null 2>&1
}
say "waiting for any running budget job to finish..."
while running; do sleep 30; done
say "no budget job running"

# Record the completed 2x2 so the morning has it even if later steps fail.
$PY -u scripts/run_budget.py --compare \
    --datasets pg_her2_hvg300 pg_bc_hvg300 pg_kidney_hvg300 >> "$LOG" 2>&1
say "budget before/after tables written to $LOG"

# --- 2. dry-validate the image-mode AL path ---------------------------------
say "--- DRY: validating image-mode UNI AL plumbing ---"
$PY -u scripts/run_uni_al.py --dry > results_pg/uni_al_dry.log 2>&1
DRY_RC=$?
if [ $DRY_RC -ne 0 ] || ! grep -q "### DONE" results_pg/uni_al_dry.log; then
  say "DRY RUN FAILED (rc=$DRY_RC) - NOT launching the long run. Tail:"
  tail -40 results_pg/uni_al_dry.log | tee -a "$LOG"
  say "=== overnight chain aborted at the dry gate ==="
  exit 1
fi
say "DRY passed - all four strategies completed a full acquire/train/eval cycle"
grep -E "per_slide|best non-random" results_pg/uni_al_dry.log | tee -a "$LOG"

# --- 3. the real run --------------------------------------------------------
say "--- REAL: UNI e2e AL spot-check, HER2, 4 folds, random+poisson_disk+tod+typiclust ---"
$PY -u scripts/run_uni_al.py > results_pg/uni_al.log 2>&1
REAL_RC=$?
say "uni_al finished rc=$REAL_RC"
tail -20 results_pg/uni_al.log | tee -a "$LOG"

say "=== overnight chain complete ==="
