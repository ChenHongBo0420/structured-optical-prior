param([string]$Python = "python")

$ErrorActionPreference = "Stop"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    & $Python run_paper.py --exp smoke --epochs 2 --seeds 0
    if ($LASTEXITCODE -ne 0) { throw "Smoke run failed (exit $LASTEXITCODE)." }
} finally {
    Pop-Location
}
