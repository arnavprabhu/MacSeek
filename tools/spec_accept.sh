#!/bin/sh
# Measure dspark draft acceptance with stock llama.cpp (llama-speculative-simple, greedy) on one prompt.
# Usage: tools/spec_accept.sh <prompt-name> <n_max> <n_predict> [outdir]
# Runs under the shared model lock (main checkout's logs/.model.lock); aborts if swap grows past 2.8 GB.
set -u
P=$1; K=$2; N=$3; OUT=${4:-logs/spec}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MAIN=/Users/arnav/Desktop/MacSeek
MODEL=${MACSEEK_TARGET:-$MAIN/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf}
DRAFT=${MACSEEK_DRAFT:-"/Volumes/SanDisk500G/Models/Deepseek v4.1 Flash 1017 GSQ-RCO/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"}
mkdir -p "$ROOT/$OUT"
LOG="$ROOT/$OUT/accept_${P}_n${K}.out"
# prompt text without the literal BOS (llama.cpp adds it)
TXT="$ROOT/$OUT/.prompt_$P.txt"
python3 -c "import sys; t=open(sys.argv[1]).read(); b='<｜begin▁of▁sentence｜>'; t=t[len(b):] if t.startswith(b) else t; open(sys.argv[2],'w').write(t)" "$MAIN/tools/prompts/$P.txt" "$TXT"
/usr/bin/lockf -k "$MAIN/logs/.model.lock" sh -c '
  sysctl vm.swapusage
  "$@" &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    used=$(sysctl -n vm.swapusage | sed -E "s/.*used = ([0-9.]+)M.*/\1/")
    if [ "$(echo "$used > 2800" | bc)" = 1 ]; then echo "ABORT: swap $used MB"; kill $pid; fi
    sleep 2
  done
  wait $pid; echo "rc=$?"
  sysctl vm.swapusage
' sh llama-speculative-simple -m "$MODEL" -md "$DRAFT" --spec-type draft-dspark --spec-draft-n-max "$K" --spec-draft-p-min 0 \
   -ngl 99 -ncmoe 99 -ngld 99 -f "$TXT" -n "$N" --temp 0 -c 2048 -t 12 -fa on > "$LOG" 2>&1
grep -E 'n_draft|n_predict|n_drafted|n_accept|accept  |decoded|rc=|ABORT|swap' "$LOG"
