#!/bin/sh
# Within-slide acquisition on HER2, both holdout geometries, strictly serial.
#
#   random   interleaved holdout — matches deployment (skip spots on a dense grid),
#            but every test spot has labelled immediate neighbours, so spatial
#            autocorrelation alone scores well. Measured kNN control: 0.4795 @100%.
#   spatial  contiguous bands — the nearest labelled spot is much further away, so
#            interpolation is weak and morphology has to do the work. This is the
#            setting that actually tests "can H&E predict expression".
#
# Running both is the point: the difference between them separates "won by
# covering space" from "won by picking informative spots".

set -u
cd /c/Users/zheyu/workspace/ST-Select
PY=/c/Users/zheyu/anaconda3/envs/st-select/python.exe
LOG=results_pg/RUN_WS_BOTH.log

say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }

say "=== within-slide chain start ==="

# Record GPU/CPU/RAM for the whole chain (see log_resources.py --summarise).
$PY -u scripts/log_resources.py --out results_pg/resources.csv \
    --interval 30 --note within_slide >/dev/null 2>&1 &
REC=$!
trap 'kill $REC 2>/dev/null' EXIT INT TERM

say "--- 1/2 mode=random (interleaved holdout) ---"
$PY -u scripts/run_within_slide.py --dataset pg_her2_hvg300 --mode random --seeds 3 \
    >> "$LOG" 2>&1
say "--- 1/2 finished rc=$?"

say "--- 2/2 mode=spatial (contiguous blocks) ---"
$PY -u scripts/run_within_slide.py --dataset pg_her2_hvg300 --mode spatial --seeds 3 \
    >> "$LOG" 2>&1
say "--- 2/2 finished rc=$?"

kill $REC 2>/dev/null
say "=== within-slide chain complete ==="
