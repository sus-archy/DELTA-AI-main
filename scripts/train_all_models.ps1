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

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  DELTA-AI: Full Model Competition" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Run: python scripts/train_all.py --help" -ForegroundColor Gray
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host

& $pythonCmd (Join-Path $repoRoot "scripts\train_all.py") @args