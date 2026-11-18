# One-command demo smoke test (PowerShell)
# Usage: .\scripts\demo.ps1 [-QuickEval] [-SkipTrain]
param(
  [switch]$QuickEval,
  [switch]$SkipTrain,
  [double]$Budget = 10000000
)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$args = @("scripts/demo.py", "--budget", "$Budget")
if ($QuickEval) { $args += "--quick-eval" }
if ($SkipTrain) { $args += "--skip-train" }
& $py @args
exit $LASTEXITCODE
