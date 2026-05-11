#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]})" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT"
BENCHMARK_NAME="benchmark_20k"

python -m ml.benchmark             --config "$REPO_ROOT/configs/benchmark_20k.yaml"             --benchmark-name "$BENCHMARK_NAME"
