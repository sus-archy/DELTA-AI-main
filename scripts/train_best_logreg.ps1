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
  -m ml.train `
  --config (Join-Path $repoRoot "configs\benchmark_20k_round2.yaml") `
  --model logreg_tfidf `
  --class-balance effective `
  --feature-variant value_char `
  --tune-c `
  --c-grid 0.5 1.0 2.0 4.0 8.0 `
  --run-name "logreg_tfidf_valuechar_effective_tuned_repro"