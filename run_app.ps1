# One-click launcher (Windows PowerShell). Creates .venv, installs dependencies, starts the app.
# No need to activate the virtual environment manually.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$py = $null
foreach ($c in @("py", "python", "python3")) { if (Get-Command $c -ErrorAction SilentlyContinue) { $py = $c; break } }
if (-not $py) { Write-Host "Python 3.10+ not found. Install from https://www.python.org/downloads/ (tick 'Add python.exe to PATH')." -ForegroundColor Red; exit 1 }

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "Creating virtual environment ..."
    if ($py -eq "py") { & py -3 -m venv .venv } else { & $py -m venv .venv }
}
$stamp = Join-Path $PSScriptRoot ".venv\.deps_installed"
if (-not (Test-Path $stamp) -or ((Get-Item "requirements.txt").LastWriteTime -gt (Get-Item $stamp).LastWriteTime)) {
    Write-Host "Installing dependencies (first run takes a few minutes) ..."
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r requirements.txt
    New-Item -ItemType File -Force -Path $stamp | Out-Null
}
if (-not (Test-Path ".env") -and (Test-Path ".env.example")) { Copy-Item ".env.example" ".env"; Write-Host "Created .env - add ANTHROPIC_API_KEY there for AI mode (optional)." -ForegroundColor Yellow }
Write-Host "Starting app at http://localhost:8501 ..." -ForegroundColor Green
& $venvPy -m streamlit run app.py
