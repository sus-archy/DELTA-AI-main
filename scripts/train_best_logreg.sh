#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]})" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT"

python -m ml.train             --config "$REPO_ROOT/configs/benchmark_20k_round2.yaml"             --model logreg_tfidf             --class-balance effective             --feature-variant value_char             --tune-c             --c-grid 0.5 1.0 2.0 4.0 8.0             --run-name "logreg_tfidf_valuechar_effective_tuned_repro"
