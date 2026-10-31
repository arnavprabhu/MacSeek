#!/bin/sh
# Quick check (~1-2 min): 200K snapshot -> decode 128 + 2048-token prefill chunk at depth; quality proxy on 3 prompts vs the
# pinned baseline. Usage: tools/quick.sh <label> [extra env assignments...]   e.g. tools/quick.sh exp1 MACSEEK_FOO=1
# Writes logs/quick/<label>.json (+ .log). Holds the model lock.
D=$(cd "$(dirname "$0")/.." && pwd); cd "$D"
L=${1:-run}; shift
export MACSEEK_ARENA_GB=${MACSEEK_ARENA_GB:-26}   # fixed arena for comparable A/B runs (auto sizing varies run to run)
mkdir -p logs/quick/$L
STATE=${MACSEEK_STATE:-logs/state/s200k.state}
CORPUS=${MACSEEK_CORPUS:-tools/prompts/long_needle_d50.ids}
BASE=${MACSEEK_QBASE:-logs/quality/base}
PROMPTS="en_prose code_python chat_think"
IDS=""; OUT=""
for p in $PROMPTS; do IDS="$IDS,tools/prompts/$p.ids"; OUT="$OUT,logs/quick/$L/$p.logits"; done
IDS=${IDS#,}; OUT=${OUT#,}
{
  echo "== $L $(date) $*"; sysctl -n vm.swapusage
  env "$@" tools/modellock.sh ${MACSEEK_BIN:-build/macseek} bench --state "$STATE" --ids "$CORPUS" --decode ${QDEC:-128} --prefill ${QPF:-4096} > logs/quick/$L/bench.json 2> logs/quick/$L/bench.err
  cat logs/quick/$L/bench.json
  [ -n "$QSKIPQ" ] || env "$@" tools/modellock.sh ${MACSEEK_BIN:-build/macseek} logits --ids "$IDS" --out "$OUT" 2> logs/quick/$L/logits.err | tail -3
  [ -n "$QSKIPQ" ] || for p in $PROMPTS; do python3 tools/quality.py "$BASE/$p.logits" "logs/quick/$L/$p.logits" | tail -1; done
  sysctl -n vm.swapusage
} 2>&1 | tee logs/quick/$L.log
