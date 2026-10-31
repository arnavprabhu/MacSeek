#!/bin/sh
# Build the test-only oracle tools against Homebrew llama.cpp/ggml. Output: build/oracle/ (in the checkout root).
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$ROOT/build/oracle"
mkdir -p "$OUT"
LLAMA_LIB=$(ls -d /opt/homebrew/Cellar/llama.cpp/*/lib 2>/dev/null | tail -1)
CXX=${CXX:-clang++}
$CXX -std=c++20 -O2 -Wall -Wno-deprecated-declarations \
  -I/opt/homebrew/include \
  "$ROOT/tools/oracle/oracle_logits.cpp" -o "$OUT/oracle_logits" \
  -L/opt/homebrew/lib -L"$LLAMA_LIB" -Wl,-rpath,/opt/homebrew/lib -Wl,-rpath,"$LLAMA_LIB" \
  -lllama -lggml -lggml-base
echo "built $OUT/oracle_logits"
$CXX -std=c++20 -O2 -Wall -I/opt/homebrew/include "$ROOT/tools/oracle/dequant_check.cpp" -o "$OUT/dequant_check" \
  -L/opt/homebrew/lib -Wl,-rpath,/opt/homebrew/lib -lggml -lggml-base
echo "built $OUT/dequant_check"
