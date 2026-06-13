# Data folder

This folder is intentionally empty.

MedMNIST data is downloaded automatically by the Python API when you run:

```python
from medmnist import PneumoniaMNIST
PneumoniaMNIST(split="train", download=True, size=224)
```

Do not commit downloaded datasets to GitHub.
