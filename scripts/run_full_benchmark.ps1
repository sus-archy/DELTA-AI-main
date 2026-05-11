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
  -m ml.benchmark `
  --config (Join-Path $repoRoot "configs\benchmark.yaml") `
  --benchmark-name "final_db_label_benchmark" `
  @args
