param(
    [string]$Python = "python",
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu130"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\")).Path
$venv = Join-Path $repo "tmp\swinir-venv"
$venvPython = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    & $Python -m venv $venv
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $repo "requirements-swinir.txt")
& $venvPython -m pip install `
    "torch==2.13.0+cu130" "torchvision==0.28.0+cu130" `
    --index-url $CudaIndex --extra-index-url "https://pypi.org/simple"
& $venvPython (Join-Path $repo "scripts\get_swinir.py")

Write-Host "SwinIR experimental environment is ready: $venvPython"
