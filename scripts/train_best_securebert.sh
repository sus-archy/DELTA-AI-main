#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]})" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT"

python -m ml.finetune             --config "$REPO_ROOT/configs/benchmark_20k_round2.yaml"             --encoder securebert2-base             --peft lora             --epochs 3             --class-balance effective             --loss balanced_softmax             --calibration temperature             --run-name "securebert2_lora_balanced_softmax_effective_epoch3_temp_repro"
