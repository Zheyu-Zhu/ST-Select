#!/bin/sh
# Wait for the in-flight BC ceiling, then work the Tier-1/Tier-2 queue to the end.
#
# Nothing here runs in parallel. Frozen-feature jobs train on the GPU in hundreds
# of short bursts; pairing one with a job that pins the card (UNI ViT-L) cost a
# measured ~6x slowdown once — 3 h of work took >18 h (EXPERIMENTS.md S12).
#
# The queue itself is resumable and failure-isolated, so if this script dies the
# recovery is simply to run it again: completed jobs are skipped.

set -u
cd /c/Users/zheyu/workspace/ST-Select
PY=/c/Users/zheyu/anaconda3/envs/st-select/python.exe
LOG=results_pg/RUN_ALL.log

say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }

# pgrep does not exist in this Git Bash — `while pgrep ...` exits 127 immediately
# and silently skips the wait, which would start UNI on top of the running job.
running() {
  powershell -NoProfile -Command \
    "if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | \
       Where-Object { \$_.CommandLine -like '*$1*' }) { exit 0 } else { exit 1 }" \
    >/dev/null 2>&1
}

say "=== run_all start ==="
say "waiting for the BC ceiling (run_uni_bc.py) to finish..."
while running "run_uni_bc.py"; do sleep 60; done
say "BC ceiling no longer running"
tail -6 results_pg/uni_bc.log 2>/dev/null | tee -a "$LOG"

say "--- starting the Tier-1/Tier-2 queue ---"
$PY -u scripts/run_queue.py 2>&1 | tee -a "$LOG"
say "=== run_all complete ==="
