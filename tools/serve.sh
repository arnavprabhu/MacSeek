#!/bin/sh
# Start the MacSeek chat server: OpenAI-compatible API + web UI on http://127.0.0.1:${PORT:-8080}
#   PORT=8080 HOST=127.0.0.1 tools/serve.sh [extra macseek serve flags]
# Holds the model lock (one model process at a time). Ctrl+C stops it.
set -e
D=$(cd "$(dirname "$0")/.." && pwd)
B=${MACSEEK_BUILD:-$D/build}
if [ ! -f "$B/build.ninja" ]; then cmake -S "$D" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null; fi
ninja -C "$B" macseek >/dev/null
echo "macseek: loading the model (about 10 s), then open http://${HOST:-127.0.0.1}:${PORT:-8080}"
exec "$D/tools/modellock.sh" "$B/macseek" serve --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" "$@"
