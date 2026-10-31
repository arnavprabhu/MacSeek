#!/bin/sh
# Build the test-only oracle tools against Homebrew llama.cpp/ggml. Output: build/oracle/ (in the checkout root).
# Binaries are linked to a temp name and renamed into place, so a running model job keeps its (old) inode.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$ROOT/build/oracle"
mkdir -p "$OUT"
LLAMA_LIB=$(ls -d /opt/homebrew/Cellar/llama.cpp/*/lib 2>/dev/null | tail -1)
CXX=${CXX:-clang++}
$CXX -std=c++20 -O2 -Wall -Wno-deprecated-declarations \
  -I/opt/homebrew/include \
  "$ROOT/tools/oracle/oracle_logits.cpp" -o "$OUT/.oracle_logits.tmp" \
  -L/opt/homebrew/lib -L"$LLAMA_LIB" -Wl,-rpath,/opt/homebrew/lib -Wl,-rpath,"$LLAMA_LIB" \
  -lllama -lggml -lggml-base
mv -f "$OUT/.oracle_logits.tmp" "$OUT/oracle_logits"
echo "built $OUT/oracle_logits"
$CXX -std=c++20 -O2 -Wall -I/opt/homebrew/include "$ROOT/tools/oracle/dequant_check.cpp" -o "$OUT/.dequant_check.tmp" \
  -L/opt/homebrew/lib -Wl,-rpath,/opt/homebrew/lib -lggml -lggml-base
mv -f "$OUT/.dequant_check.tmp" "$OUT/dequant_check"
echo "built $OUT/dequant_check"
