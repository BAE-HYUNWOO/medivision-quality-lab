# MediVision Quality Lab

**MediVision Quality Lab** is a research-oriented medical image AI prototype.
It trains a PyTorch image classification model on MedMNIST, serves predictions through FastAPI, visualizes model attention with Grad-CAM, and deploys the frontend, backend, and model service on Kubernetes.

> ⚠️ This project is for research and education only. It is **not** a clinical diagnosis tool and must not be used for medical decision-making.

## MVP target

The first MVP uses **PneumoniaMNIST** from MedMNIST:

- Input: chest X-ray style image from MedMNIST or user-uploaded image
- Model: ResNet18 fine-tuned with PyTorch
- Output: predicted class, confidence, and Grad-CAM heatmap
- Deployment: React + FastAPI + PyTorch model service + Kubernetes

## Architecture

```text
React frontend
  ↓
FastAPI backend / API gateway
  ↓
PyTorch model API
  ↓
Prediction + Grad-CAM explanation
```

Kubernetes version:

```text
frontend pod       React/Vite UI
backend pod        FastAPI gateway
model-api pod      PyTorch model inference + Grad-CAM
batch-eval job     batch evaluation on MedMNIST test split
```

## Repository structure

```text
medivision-quality-lab/
├─ ml/               PyTorch training, evaluation, prediction, Grad-CAM scripts
├─ backend/          FastAPI app
├─ frontend/         React/Vite app
├─ docker/           Dockerfiles
├─ k8s/              Kubernetes manifests
├─ scripts/          Windows PowerShell helper scripts
├─ notebooks/        Starter notebooks
├─ data/             Placeholder; MedMNIST is downloaded by code
└─ models/           Placeholder for trained model checkpoint
```

## 1. Train the model

Install dependencies:

```bash
pip install -r requirements.txt
```

Train ResNet18 on PneumoniaMNIST at 224x224:

```bash
python -m ml.train --data-flag pneumoniamnist --size 224 --model-name resnet18 --epochs 8 --batch-size 32 --output-path models/pneumonia_resnet18_224.pt
```

Evaluate:

```bash
python -m ml.evaluate --data-flag pneumoniamnist --size 224 --model-name resnet18 --model-path models/pneumonia_resnet18_224.pt
```

Predict one image:

```bash
python -m ml.predict --image-path path/to/image.png --model-path models/pneumonia_resnet18_224.pt --model-name resnet18
```

## 2. Run the FastAPI backend locally

```bash
cd backend
pip install -r requirements.txt
$env:MODEL_PATH="../models/pneumonia_resnet18_224.pt"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

Main endpoints:

```text
GET  /api/health
POST /api/predict
POST /api/predict-with-gradcam
```

## 3. Run the React frontend locally

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## 4. Docker build

From repository root:

```bash
docker build -f docker/Dockerfile.backend -t medivision-backend:latest .
docker build -f docker/Dockerfile.frontend -t medivision-frontend:latest ./frontend
docker build -f docker/Dockerfile.model -t medivision-model:latest .
```

## 5. Kubernetes on minikube

Start minikube:

```bash
minikube start --driver=docker
```

Load local images into minikube:

```bash
minikube image load medivision-backend:latest
minikube image load medivision-frontend:latest
minikube image load medivision-model:latest
```

Deploy:

```bash
kubectl apply -f k8s/
kubectl get pods -n medivision
```

Open frontend:

```bash
minikube service frontend-service -n medivision
```

## 6. Model checkpoint note

This starter package intentionally does **not** include datasets or trained model files.
After training, place the checkpoint here:

```text
models/pneumonia_resnet18_224.pt
```

For Docker/Kubernetes deployment, either:

1. Build the model checkpoint into the model image, or
2. Mount it as a volume/PVC, or
3. Download it from object storage during container startup.

The current manifests are intended as a local development starting point.

## Research framing

This project should be framed as a **medical AI interpretability and deployment prototype**, not as a diagnostic product.

Possible research questions:

1. How should medical image AI predictions and uncertainty be shown to non-expert users?
2. Does Grad-CAM help users understand model outputs, or does it increase over-trust?
3. Can a Kubernetes-based research workflow make medical image AI experiments more reproducible and easier to demonstrate?

## Disclaimer

This software is for research demonstration only. It is not validated for clinical use. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.
