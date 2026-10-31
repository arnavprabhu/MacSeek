#!/bin/sh
# Start the MacSeek chat server: OpenAI-compatible API + web UI on http://127.0.0.1:${PORT:-8080}
#   PORT=8080 HOST=127.0.0.1 tools/serve.sh [extra macseek serve flags]
# Holds the model lock (one model process at a time). Ctrl+C stops it.
set -e
D=$(cd "$(dirname "$0")/.." && pwd)
B=${MACSEEK_BUILD:-$D/build}
if [ ! -f "$B/build.ninja" ]; then cmake -S "$D" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null; fi
ninja -C "$B" macseek >/dev/null
M=$(python3 "$D/tools/model_path.py" 2>/dev/null || true)
if [ -n "$M" ] && [ ! -f "$M" ]; then
  echo "macseek: model not found at" >&2
  echo "  $M" >&2
  echo "Download it (see README: Get the model), or point to where it is:" >&2
  echo "  export HF_HUB_CACHE=/path/to/huggingface/hub        # a Hugging Face cache that has it" >&2
  echo "  export MACSEEK_MODEL_DIR=/path/to/UD-IQ1_S          # or the folder with the 5 .gguf files" >&2
  exit 1
fi
echo "macseek: loading the model (about 10 s), then open http://${HOST:-127.0.0.1}:${PORT:-8080}"
exec "$D/tools/modellock.sh" "$B/macseek" serve --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" "$@"
