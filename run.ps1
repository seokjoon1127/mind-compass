# Decision Debugger — local run (PowerShell)
# Usage:  .\run.ps1
$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Setting up virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt
}
Write-Host "Decision Debugger -> http://localhost:8000" -ForegroundColor Green
& $py -m uvicorn backend.main:app --reload --port 8000
