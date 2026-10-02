param([string]$Python = "python")

$ErrorActionPreference = "Stop"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    & $Python run_paper.py --exp all_core --epochs 300 --seeds '0,1,2'
    if ($LASTEXITCODE -ne 0) { throw "Core run failed (exit $LASTEXITCODE)." }
} finally {
    Pop-Location
}
