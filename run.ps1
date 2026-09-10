$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectDir '.venv\Scripts\python.exe'
$appFile = Join-Path $projectDir 'wondersnap.py'

if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Host 'The local Python environment is missing.' -ForegroundColor Yellow
    Write-Host 'Run these commands once:'
    Write-Host '  python -m venv .venv'
    Write-Host '  .\.venv\Scripts\python.exe -m pip install -r requirements.txt'
    exit 2
}

# The camera password now ships with the app, so nothing is asked for here.
# Set WONDERSNAP_RTSP_PASSWORD before running to override it, for example:
#   $env:WONDERSNAP_RTSP_PASSWORD = 'other-password'; .\run.ps1

& $pythonExe $appFile @args
exit $LASTEXITCODE
