#!/bin/sh
# Quality gate on the 8 short prompts: full logits vs the pinned llama.cpp oracle (logs/oracle/v0).
# Usage: tools/gate8.sh <label> [--decode K] [env assignments...]   -> logs/gate/<label>/{*.logits,run.err,quality.txt}
# --decode K: positions >= K are fed one by one through decode() (checks the decode kernels).
D=$(cd "$(dirname "$0")/.." && pwd); cd "$D"
L=${1:?label}; shift
DEC=""
if [ "$1" = "--decode" ]; then DEC="--decode $2"; shift 2; fi
O=logs/gate/$L; mkdir -p "$O"
P="en_prose code_python code_cpp math chat chat_think cjk json"
IDS=""; OUT=""
for p in $P; do IDS="$IDS,tools/prompts/$p.ids"; OUT="$OUT,$O/$p.logits"; done
env "$@" tools/modellock.sh ${MACSEEK_BIN:-build/macseek} logits $DEC --ids "${IDS#,}" --out "${OUT#,}" 2> "$O/run.err" || { echo "gate8: run failed"; tail -5 "$O/run.err"; exit 1; }
python3 tools/quality.py --dir logs/oracle/v0 "$O" > "$O/quality.txt" 2>&1
tail -1 "$O/quality.txt"
