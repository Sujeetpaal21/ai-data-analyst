$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Write-Host "Run run_app.ps1 once first (it creates the environment)." -ForegroundColor Red; exit 1 }
& $venvPy -m pip install -q pytest
& $venvPy -m pytest -v
