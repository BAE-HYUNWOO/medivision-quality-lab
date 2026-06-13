# Colab training quick start

Run these cells in Google Colab.

```python
!pip install torch torchvision medmnist grad-cam scikit-learn opencv-python tqdm
```

Upload this project to Google Drive or clone it from GitHub, then run:

```python
!python -m ml.train \
  --data-flag pneumoniamnist \
  --size 224 \
  --model-name resnet18 \
  --epochs 8 \
  --batch-size 32 \
  --output-path models/pneumonia_resnet18_224.pt
```

Download the model checkpoint and place it in:

```text
models/pneumonia_resnet18_224.pt
```
