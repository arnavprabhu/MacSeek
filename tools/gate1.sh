#!/bin/sh
# One-prompt quality probe vs the llama.cpp oracle: tools/gate1.sh <prompt> [env assignments...] -> one summary line
D=$(cd "$(dirname "$0")/.." && pwd); cd "$D"
P=${1:?prompt}; shift
O=$(mktemp -d "${TMPDIR:-/tmp}/gate1.XXXXXX")
env "$@" tools/modellock.sh ${MACSEEK_BIN:-build/macseek} logits --ids tools/prompts/$P.ids --out $O/$P.logits 2> $O/run.err || { echo "run failed"; tail -3 $O/run.err; exit 1; }
echo "$* : $(python3 tools/quality.py --dir logs/oracle/v0 $O 2>&1 | grep "^$P " | cut -c1-110)"
rm -rf "$O"
