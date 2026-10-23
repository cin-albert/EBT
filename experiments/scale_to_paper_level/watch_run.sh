#!/usr/bin/env bash
# Run a launch script as a child and, if its log stops growing for STALL_MIN minutes, send SIGUSR1 to every process in
# its tree: train_model.py registers faulthandler on SIGUSR1, so each rank and DataLoader worker prints its stacks
# into the log (py-spy cannot attach: kernel.yama.ptrace_scope=1). Process list and nvidia-smi go to <log>.stall.
# Usage: GPU_IDS=0,1,2,3 bash watch_run.sh <log> <run script> [run args]
set -uo pipefail
LOG=$1; shift
STALL_MIN=${STALL_MIN:-10}
bash "$@" > "$LOG" 2>&1 &
CHILD=$!
descendants() { local c; for c in $(ps -o pid= --ppid "$1"); do echo "$c"; descendants "$c"; done; }  # no pstree here
dumped=0
while kill -0 "$CHILD" 2>/dev/null; do
  sleep 60
  idle=$(( ($(date +%s) - $(stat -c %Y "$LOG")) / 60 ))
  if [ "$idle" -ge "$STALL_MIN" ] && [ "$dumped" = 0 ]; then
    { date; echo "log idle ${idle} min"; nvidia-smi; } > "$LOG.stall"
    pids=$(descendants "$CHILD"); ps -o pid,ppid,stat,wchan:30,cmd -p "$CHILD,${pids//$'\n'/,}" | cut -c1-200 >> "$LOG.stall"
    for pid in $CHILD $pids; do
      grep -q train_model.py "/proc/$pid/cmdline" 2>/dev/null && kill -USR1 "$pid"  # others would die on USR1
    done
    dumped=1
  fi
  [ "$idle" -lt "$STALL_MIN" ] && dumped=0
done
wait "$CHILD"; rc=$?
echo "EXIT=$rc" >> "$LOG"
exit $rc
