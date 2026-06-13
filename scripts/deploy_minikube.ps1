$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\.."

minikube start --driver=docker

minikube image load medivision-backend:latest
minikube image load medivision-model:latest
minikube image load medivision-frontend:latest

kubectl apply -f k8s/
kubectl get pods -n medivision

Write-Host "Open frontend with: minikube service frontend-service -n medivision"
