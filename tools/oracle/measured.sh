#!/bin/sh
# Usage: tools/oracle/measured.sh <logfile> <command...>
# Runs <command> under the model lock (tools/modellock.sh) and appends to <logfile>:
# date, swap usage, vm_stat pageins/pageouts/free before and after, /usr/bin/time -l, exit code.
# Page-cache state matters a lot for llama.cpp speed on this 233 GB model: the "before" free-pages
# and the pagein delta tell you whether a run was cold (many pageins) or warm.
set -u
LOG="$1"; shift
# Always the MAIN checkout's lock (worktrees would otherwise lock their own logs/.model.lock).
LOCK=${MACSEEK_LOCK:-/Users/arnav/Desktop/Projects/MacSeek/tools/modellock.sh}
mkdir -p "$(dirname "$LOG")"
snap() {
  echo "--- $1: $(date '+%Y-%m-%d %H:%M:%S')"
  sysctl vm.swapusage
  vm_stat | grep -E 'Pages free|Pages active|Pageins|Pageouts|Swapins|Swapouts|File-backed'
  memory_pressure 2>/dev/null | tail -1
}
{
  echo "=== CMD: $*"
  snap BEFORE
} >> "$LOG" 2>&1
"$LOCK" /usr/bin/time -l "$@" >> "$LOG" 2>&1
rc=$?
{
  echo "rc=$rc"
  snap AFTER
} >> "$LOG" 2>&1
exit $rc
