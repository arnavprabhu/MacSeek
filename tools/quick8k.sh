#!/bin/sh
# Fast iteration speed check: prefill 8192 tokens of the long prompt from an empty context, then generate 4096 tokens
# (greedy). Full 262K runs are for the end. Usage: tools/quick8k.sh <label> [env assignments...]
# -> logs/quick8k/<label>.err + one summary line.
D=$(cd "$(dirname "$0")/.." && pwd); cd "$D"
L=${1:?label}; shift
mkdir -p logs/quick8k
env "$@" MACSEEK_PROF=1 tools/modellock.sh ${MACSEEK_BIN:-build/macseek} profile --ids tools/prompts/long_needle_d50_260k.ids \
  --n 8192 --decode ${QUICK_DECODE:-4096} 2> logs/quick8k/$L.err
PF=$(grep -E "^prefill:" logs/quick8k/$L.err | sed -E 's/^prefill: ([0-9]+) tok in [0-9]+ chunks, ([0-9.]+) tok\/s.*/\2/')
DEC=$(grep -E "^decode @" logs/quick8k/$L.err | sed -E 's/^decode @[0-9]+: ([0-9.]+) tok\/s.*total ([0-9.]+) gpu ([0-9.]+).*io_stall ([0-9.]+).*hit ([0-9.]+).*/\1 tok\/s (\2 ms\/tok, gpu \3, io stall \4, hit \5)/')
echo "$L $*: prefill 8K ${PF:-FAILED} tok/s | decode 4K ${DEC:-FAILED}"
