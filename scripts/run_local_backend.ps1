$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\..\backend"

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:MODEL_PATH = "..\models\pneumonia_resnet18_224.pt"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
