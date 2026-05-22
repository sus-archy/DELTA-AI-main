$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = $repoRoot

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $pythonCmd) {
    Write-Error "Python not found. Install Python 3.11+ and ensure it is on your PATH."
}

& $pythonCmd `
  -m ml.finetune `
  --config (Join-Path $repoRoot "configs\benchmark_20k_round2.yaml") `
  --encoder securebert2-base `
  --peft lora `
  --epochs 3 `
  --class-balance effective `
  --loss balanced_softmax `
  --calibration temperature `
  --run-name "securebert2_lora_balanced_softmax_effective_epoch3_temp_repro"