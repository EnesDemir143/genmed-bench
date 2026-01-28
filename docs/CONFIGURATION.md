# Configuration Guide

> Complete reference for all configuration options in GenMed-Bench.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/config.yaml` | Data paths, preprocessing settings |
| `configs/models.yaml` | Model architectures, hyperparameters |

---

## config.yaml

### Data Paths

```yaml
data:
  raw_root: "data/raw/chest-datasets"      # Raw dataset location
  processed_root: "data/processed"          # LMDB databases
  
  datasets:
    nih:
      raw: "${data.raw_root}/nih"
      processed: "${data.processed_root}/nih/nih.lmdb"
    covidx:
      raw: "${data.raw_root}/covidx"
      processed: "${data.processed_root}/covidx/covidx.lmdb"
    vinbigdata:
      raw: "${data.raw_root}/vinbigdata"
      processed: "${data.processed_root}/vinbigdata/vinbigdata.lmdb"

preprocessing:
  image_size: 256        # Default image size for LMDB storage
  normalize: true        # Apply normalization
```

---

## models.yaml

### Global Defaults

```yaml
defaults:
  input_size: 224        # Model input size
  in_chans: 3            # Input channels (RGB)
  num_classes: 2         # Default number of classes
  pretrained: true       # Use pretrained weights
  mixed_precision: true  # Enable AMP
  
  dataloader:
    batch_size: 64
    num_workers: 4
    pin_memory: true
    drop_last: true
```

### Training Mode Defaults

#### Linear Probe
```yaml
linear_probe:
  freeze_backbone: true
  trainable_layers: ["head", "fc", "classifier"]
  
  optimizer:
    name: SGD
    lr: 0.1
    momentum: 0.9
    weight_decay: 0.0
    nesterov: true
    
  scheduler:
    name: cosine
    T_max: 100
    eta_min: 0.0
    
  training:
    epochs: 100
    warmup_epochs: 0
    batch_size: 256
    gradient_clip: null
```

#### Full Fine-tune
```yaml
full_finetune:
  freeze_backbone: false
  
  optimizer:
    name: AdamW
    lr: 1.0e-4
    betas: [0.9, 0.999]
    weight_decay: 0.05
    
  scheduler:
    name: cosine_with_warmup
    T_max: 100
    eta_min: 1.0e-6
    warmup_lr_init: 1.0e-7
    
  training:
    epochs: 100
    warmup_epochs: 5
    batch_size: 32
    gradient_clip: 1.0
    layer_decay: 0.65       # Layer-wise LR decay for transformers
    drop_path_rate: 0.1     # Stochastic depth
```

### Model-Specific Configuration

Each model can override global settings:

```yaml
models:
  vit_small_patch16:
    timm_name: vit_small_patch16_224
    type: vit
    params_m: 22.0
    
    linear_probe:
      lr: 0.1
      batch_size: 256
      epochs: 100
      
    full_finetune:
      optimizer:
        name: AdamW
        betas: [0.9, 0.999]
      lr: 5.0e-5
      batch_size: 64
      epochs: 100
      warmup_epochs: 5
      weight_decay: 0.05
      layer_decay: 0.65
      drop_path_rate: 0.1
```

---

## CLI Parameters

Command-line arguments override config file values:

| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | Model name from models.yaml | `vit_small_patch16` |
| `--dataset` | Dataset name | `nih`, `covidx`, `vinbigdata` |
| `--mode` | Training mode | `linear_probe`, `full_finetune` |
| `--epochs` | Override epochs | `50` |
| `--batch_size` | Override batch size | `32` |
| `--lr` | Override learning rate | `1e-4` |
| `--weight_decay` | Override weight decay | `0.01` |
| `--warmup_epochs` | Warmup epochs | `5` |
| `--gradient_clip` | Gradient clipping | `1.0` |
| `--drop_path_rate` | Drop path rate | `0.1` |
| `--early_stopping` | Early stopping patience | `10` |
| `--val_ratio` | Validation split ratio | `0.2` |
| `--multi_label` | Multi-label mode | flag |
| `--mixed_precision` | Enable AMP | flag |

---

## Optimizer Options

| Name | Use Case | Key Parameters |
|------|----------|----------------|
| `SGD` | Linear probe, CNNs | `lr`, `momentum`, `nesterov` |
| `AdamW` | Transformers, fine-tuning | `lr`, `betas`, `weight_decay` |
| `RMSprop` | EfficientNet | `lr`, `momentum`, `eps` |

---

## Scheduler Options

| Name | Description |
|------|-------------|
| `cosine` | Cosine annealing without warmup |
| `cosine_with_warmup` | Cosine annealing with linear warmup |
| `step` | Step decay |
| `plateau` | Reduce on plateau |

---

## Model Types

| Type | Examples | Recommended Optimizer |
|------|----------|----------------------|
| `cnn` | ResNet, MobileNet, EfficientNet | SGD (linear), SGD/RMSprop (finetune) |
| `vit` | ViT, DeiT, BEiT, EVA | SGD (linear), AdamW (finetune) |
| `swin` | Swin, SwinV2 | SGD (linear), AdamW (finetune) |

---

## Priority Order

1. CLI arguments (highest)
2. Model-specific config in models.yaml
3. Training mode defaults
4. Global defaults (lowest)
