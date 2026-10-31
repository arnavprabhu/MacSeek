#!/bin/sh
# Usage: tools/modellock.sh <command...>   — runs the command holding the single model lock (one model process at a time).
D=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$D/logs"
exec /usr/bin/lockf -k "$D/logs/.model.lock" "$@"
