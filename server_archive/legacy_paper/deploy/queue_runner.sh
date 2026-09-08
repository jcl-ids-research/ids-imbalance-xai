#!/bin/bash
# Server-side job queue. Runs under nohup so it survives client disconnects.
#
#   ./queue_runner.sh <max_concurrent> <threads> <nice>
#
# Reads jobs from queue.txt, one shell command per line, '#' comments ignored.
# Keeps at most max_concurrent of OUR jobs alive; other users' processes are
# never touched. Writes progress to queue_runner.log.

cd /opt/ids_revision/deploy || exit 1

MAXC=${1:-2}
THREADS=${2:-8}
NICEV=${3:-19}
QUEUE=queue.txt
LOG=queue_runner.log

export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS

log() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }

running_count() {
  pgrep -f 'python3 scripts/(baseline_fair|run_instrumented|correct_cross_dataset)\.py' \
    | wc -l
}

log "queue_runner start  max_concurrent=$MAXC threads=$THREADS nice=$NICEV"

mapfile -t JOBS < <(grep -vE '^\s*(#|$)' "$QUEUE")
log "loaded ${#JOBS[@]} jobs"

i=0
while [ $i -lt ${#JOBS[@]} ] || [ "$(running_count)" -gt 0 ]; do
  while [ $i -lt ${#JOBS[@]} ] && [ "$(running_count)" -lt "$MAXC" ]; do
    JOB="${JOBS[$i]}"
    NAME=$(echo "$JOB" | grep -oE '\-\-dataset [a-z0-9]+ --seed [0-9]+' | tr ' ' '_')
    TAG=$(echo "$JOB" | grep -oE 'scripts/[a-z_]+\.py' | sed 's|scripts/||;s|\.py||')
    OUTLOG="logs/${TAG}${NAME}.log"
    mkdir -p logs
    log "launch [$((i+1))/${#JOBS[@]}] $NAME ($TAG)"
    setsid nohup nice -n "$NICEV" bash -c "$JOB" </dev/null >"$OUTLOG" 2>&1 &
    i=$((i+1))
    sleep 10
  done
  sleep 60
  log "running=$(running_count) launched=$i/${#JOBS[@]} load=$(cut -d' ' -f1 /proc/loadavg)"
done

log "ALL DONE ($i jobs)"
