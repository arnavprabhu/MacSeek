#!/bin/sh
# Stock llama.cpp speed baseline (the number MacSeek must beat). Same model config as the oracle dumps:
#   -ngl 99 (all layers on Metal) + -ncmoe 99 (all MoE expert tensors stay on CPU, mmap'd) — full offload fails
#   (Metal "Insufficient Memory"), see docs/modules/oracle.md.
# usage: tools/oracle/bench_baseline.sh <label> [extra llama-bench args...]
#   e.g. tools/oracle/bench_baseline.sh d0 -p 512,2048 -n 128 -r 2
# Output: logs/baseline/<label>.log (full log incl. swap/vm_stat before+after, /usr/bin/time -l) and
#         logs/baseline/<label>.jsonl (llama-bench results, one JSON object per test).
set -eu
LABEL="$1"; shift
HERE=$(cd "$(dirname "$0")" && pwd)
MAIN=${MACSEEK_MAIN:-$(cd "$HERE/../.." && pwd)}
OUT="$MAIN/logs/baseline"
M="${MACSEEK_MODEL:-$(python3 "$MAIN/tools/model_path.py")}"
mkdir -p "$OUT"
"$HERE/measured.sh" "$OUT/$LABEL.log" /opt/homebrew/bin/llama-bench -m "$M" -ngl 99 -ncmoe 99 -o jsonl --progress "$@" || true
grep '^{' "$OUT/$LABEL.log" > "$OUT/$LABEL.jsonl" || true
python3 - "$OUT/$LABEL.jsonl" <<'EOF'
import json, sys
for l in open(sys.argv[1]):
    d = json.loads(l)
    print(f"{d.get('n_prompt',0):>6} pp {d.get('n_gen',0):>4} tg  depth {d.get('n_depth',0):>7}  t={d.get('n_threads')}  "
          f"{d.get('avg_ts',0):8.2f} ± {d.get('stddev_ts',0):.2f} tok/s  samples={d.get('samples_ts')}")
EOF
