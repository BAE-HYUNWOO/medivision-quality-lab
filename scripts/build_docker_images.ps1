$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\.."

docker build -f docker/Dockerfile.backend -t medivision-backend:latest .
docker build -f docker/Dockerfile.model -t medivision-model:latest .
docker build -f docker/Dockerfile.frontend -t medivision-frontend:latest .\frontend

Write-Host "Built medivision Docker images."
