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
$benchmarkName = "benchmark_10k"
$benchmarkRoot = Join-Path $repoRoot "artifacts\benchmarks\$benchmarkName"
$stdoutLog = Join-Path $benchmarkRoot "stdout.log"
$stderrLog = Join-Path $benchmarkRoot "stderr.log"
$launcherPath = Join-Path $benchmarkRoot "launcher.json"

New-Item -ItemType Directory -Force -Path $benchmarkRoot | Out-Null

$process = Start-Process `
  -FilePath $pythonCmd `
  -ArgumentList @(
    "-m", "ml.benchmark",
    "--config", (Join-Path $repoRoot "configs\benchmark_10k.yaml"),
    "--benchmark-name", $benchmarkName
  ) `
  -WorkingDirectory $repoRoot `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog `
  -PassThru

$launcher = [ordered]@{
  benchmark = $benchmarkName
  pid = $process.Id
  startedAt = (Get-Date).ToString("o")
  stdoutLog = $stdoutLog
  stderrLog = $stderrLog
  benchmarkRoot = $benchmarkRoot
}

$launcher | ConvertTo-Json | Set-Content -Path $launcherPath -Encoding UTF8
$launcher | ConvertTo-Json
