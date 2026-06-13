# Models folder

This folder is intentionally empty.

After training, save your checkpoint here:

```text
models/pneumonia_resnet18_224.pt
```

The checkpoint should contain:

```python
{
    "model_name": "resnet18",
    "data_flag": "pneumoniamnist",
    "size": 224,
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "num_classes": num_classes,
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225]
}
```

Do not commit large trained model files unless you explicitly want to release them.
