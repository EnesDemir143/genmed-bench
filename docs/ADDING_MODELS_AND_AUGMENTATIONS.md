# Adding Models and Augmentations

> Guide to add new models and augmentation methods to GenMed-Bench.

---

## 📌 Adding Models

### Current Architecture

Models are loaded from `timm` library. Configurations are defined in `configs/models.yaml`.

### Step 1: Add to models.yaml

**File:** `configs/models.yaml`

```yaml
new_model_name:
  pretrained_cfg:
    model_name: "timm_model_name"  # Name in timm
    pretrained: true
    in_channels: 3
    drop_path_rate: 0.1
  
  linear_probe:
    epochs: 50
    batch_size: 64
    optimizer:
      name: "sgd"
      lr: 0.01
      momentum: 0.9
      weight_decay: 0.0
    scheduler:
      name: "cosine"
      warmup_epochs: 5
  
  full_finetune:
    epochs: 100
    batch_size: 32
    optimizer:
      name: "adamw"
      lr: 1e-4
      weight_decay: 0.05
    scheduler:
      name: "cosine"
      warmup_epochs: 10
```

### Step 2: Test

```bash
# Check if model exists in timm
uv run python -c "import timm; print(timm.list_models('*new_model*'))"

# Training test
uv run python train.py --model new_model_name --dataset nih --epochs 1
```

---

## 🔧 Adding Custom Backbone (non-timm)

For models not in timm:

### Step 1: Add to backbone.py

**File:** `src/models/backbone.py`

```python
def create_backbone(model_name: str, pretrained: bool = True, **kwargs):
    """Create backbone model."""
    
    # Custom model check
    if model_name == "my_custom_model":
        return _load_custom_model(pretrained, **kwargs)
    
    # Default: load from timm
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        **kwargs
    )

def _load_custom_model(pretrained: bool, **kwargs):
    """Load custom model."""
    from my_custom_module import CustomModel
    
    model = CustomModel(**kwargs)
    if pretrained:
        state_dict = torch.load("path/to/weights.pth")
        model.load_state_dict(state_dict)
    
    return model
```

---

## 🎨 Adding Augmentations

### Current Architecture

Augmentations are in `src/data/augmentation/` and used in `trainer_sup.py`.

### Step 1: Write Augmentation Class

**File:** `src/data/augmentation/my_augmentation.py`

```python
"""MyAugmentation - New augmentation method."""

import torch
import torch.nn as nn
from typing import Tuple


class MyAugmentation(nn.Module):
    """
    New augmentation class.
    
    Args:
        alpha: Beta distribution parameter for mix ratio
        prob: Probability of applying augmentation
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
    
    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply augmentation.
        
        Args:
            images: [B, C, H, W] tensor
            labels: [B] or [B, num_classes] tensor
            
        Returns:
            mixed_images: Mixed images
            labels_a: Original labels
            labels_b: Mixed labels
            lam: Mix ratio (for loss calculation)
        """
        if torch.rand(1).item() > self.prob:
            return images, labels, labels, 1.0
        
        batch_size = images.size(0)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        index = torch.randperm(batch_size, device=images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_images, labels_a, labels_b, lam
```

### Step 2: Add to __init__.py

**File:** `src/data/augmentation/__init__.py`

```python
from .my_augmentation import MyAugmentation  # ← ADD

__all__ = [
    ...
    'MyAugmentation',  # ← ADD
]
```

### Step 3: Add to trainer_sup.py

**File:** `src/train/trainer_sup.py`

```python
def _setup_augmentation(self, augmentation_name: Optional[str]):
    ...
    elif augmentation_name == "my_augmentation":  # ← ADD
        from src.data.augmentation import MyAugmentation
        self.augmentation = MyAugmentation(
            alpha=self.config.get('mixup_alpha', 1.0),
            prob=self.config.get('mixup_prob', 0.5),
        )
```

### Step 4: Add to CLI

**File:** `train.py`

```python
choices=['none', 'xdomainmix', 'pipmix', 'my_augmentation']  # ← ADD
```

### Step 5: Test

```bash
uv run python train.py \
    --model resnet50 \
    --dataset nih \
    --augmentation my_augmentation \
    --epochs 1
```

---

## 📋 Summary

### Adding a Model

| Step | File | Action |
|------|------|--------|
| 1 | `configs/models.yaml` | Define model config |
| 2 | (optional) `src/models/backbone.py` | Add custom loader |
| 3 | Test | `train.py --model new_model` |

### Adding an Augmentation

| Step | File | Action |
|------|------|--------|
| 1 | `src/data/augmentation/new_aug.py` | Write class |
| 2 | `src/data/augmentation/__init__.py` | Add export |
| 3 | `src/train/trainer_sup.py` | Add to setup function |
| 4 | `train.py` | Add to CLI choices |
| 5 | Test | `train.py --augmentation new_aug` |

---

## Existing Examples

| Augmentation | File | Description |
|--------------|------|-------------|
| XDomainMix | `xdomainmix.py` | Cross-domain mixup |
| PipMix | `pipmix.py` | Progressive image processing mix |

| Model | timm name | Type |
|-------|-----------|------|
| `resnet50` | `resnet50` | CNN |
| `vit_small_patch16` | `vit_small_patch16_224` | ViT |
| `swin_tiny_patch4` | `swin_tiny_patch4_window7_224` | Swin |
