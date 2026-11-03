#!/bin/sh
# Usage: tools/mrun.sh <log> <macseek args...>
# Runs this checkout's build/macseek (or $MACSEEK_BIN) under the shared model lock of the MAIN checkout, with the Coder
# model directory of the main checkout (worktrees have no models/ link), appending output + swap before/after to <log>.
set -u
LOG=$1; shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MAIN=${MACSEEK_MAIN:-/Users/arnav/Desktop/MacSeek}
BIN=${MACSEEK_BIN:-$ROOT/build/macseek}
export MACSEEK_MODEL_DIR=${MACSEEK_MODEL_DIR:-$MAIN/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS}
export MACSEEK_DRAFT=${MACSEEK_DRAFT:-$MAIN/models/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf}
mkdir -p "$(dirname "$LOG")"
{
  echo "=== $(date '+%F %T') $BIN $*"
  env | grep '^MACSEEK_' | grep -v 'MODEL_DIR\|MACSEEK_DRAFT' | sort | tr '\n' ' '; echo
  sysctl vm.swapusage
} >> "$LOG"
/usr/bin/lockf -k "$MAIN/logs/.model.lock" "$BIN" "$@" >> "$LOG" 2>&1
rc=$?
{ echo "rc=$rc"; sysctl vm.swapusage; } >> "$LOG"
exit $rc
