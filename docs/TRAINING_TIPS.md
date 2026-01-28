# Training Tips

> Best practices and tips for training medical image classification models.

---

## General Guidelines

### Linear Probe vs Full Fine-tune

| Mode | When to Use | Typical LR |
|------|-------------|------------|
| **Linear Probe** | Quick evaluation, limited data, baseline | 0.1 (SGD) |
| **Full Fine-tune** | Best performance, enough data | 1e-4 to 5e-5 (AdamW) |

```bash
# Linear probe (fast, good baseline)
uv run python train.py --model vit_small_patch16 --mode linear_probe --epochs 50

# Full fine-tune (best results)
uv run python train.py --model vit_small_patch16 --mode full_finetune --epochs 100
```

---

## Learning Rate

### By Model Type

| Model Type | Linear Probe | Fine-tune |
|------------|--------------|-----------|
| CNN (ResNet) | 0.1 | 1e-3 |
| ViT / DeiT | 0.1 | 5e-5 |
| Swin | 0.1 | 3e-5 to 5e-5 |
| ConvNeXt | 0.1 | 5e-5 |

### Layer-wise LR Decay

For transformers, use layer-wise decay:

```yaml
full_finetune:
  layer_decay: 0.65  # Earlier layers get smaller LR
```

---

## Batch Size

| GPU Memory | Recommended Batch Size |
|------------|----------------------|
| 8GB | 16-32 |
| 16GB | 32-64 |
| 24GB+ | 64-128 |

> **Tip**: For linear probe, use larger batch sizes (256+).

```bash
# Smaller batch for fine-tuning
uv run python train.py --model swin_base --mode full_finetune --batch_size 16
```

---

## Regularization

### Drop Path (Stochastic Depth)

```yaml
drop_path_rate: 0.1  # Small models
drop_path_rate: 0.2  # Medium models
drop_path_rate: 0.3  # Large models
```

### Weight Decay

| Training Mode | Recommended |
|--------------|-------------|
| Linear probe | 0.0 |
| Fine-tune (CNN) | 1e-4 |
| Fine-tune (Transformer) | 0.05 |

### Label Smoothing

```bash
uv run python train.py --label_smoothing 0.1
```

---

## Data Augmentation

### When to Use XDomainMix/PipMix

| Scenario | Recommendation |
|----------|----------------|
| Single-domain training | Standard augmentation |
| Multi-domain training | XDomainMix |
| Domain generalization | XDomainMix + PipMix |

```bash
# With augmentation
uv run python train.py --augmentation xdomainmix --mixup_alpha 0.2 --mixup_prob 0.5
```

---

## Dataset-Specific Tips

### NIH ChestX-ray14 (Multi-label)

```bash
uv run python train.py \
    --dataset nih \
    --multi_label \
    --epochs 100 \
    --batch_size 64
```

- 14 classes, highly imbalanced
- Use `--multi_label` flag
- Consider class weights or focal loss

### COVIDx (Multi-class)

```bash
uv run python train.py \
    --dataset covidx \
    --epochs 50 \
    --batch_size 32
```

- 3 classes: COVID, Normal, Pneumonia
- Smaller dataset, watch for overfitting
- Consider early stopping

### VinBigData (Binary)

```bash
uv run python train.py \
    --dataset vinbigdata \
    --epochs 30 \
    --batch_size 64
```

- Binary classification
- Fast training
- Good for quick experiments

---

## Early Stopping

```bash
# Stop if no improvement for 10 epochs
uv run python train.py --early_stopping 10
```

---

## Mixed Precision

```bash
# Enable (default for supported GPUs)
uv run python train.py --mixed_precision

# Disable (for debugging)
uv run python train.py --no_mixed_precision
```

---

## Memory Optimization

### If Running Out of Memory

1. Reduce batch size: `--batch_size 16`
2. Use gradient checkpointing (model-dependent)
3. Use smaller model variant
4. Disable mixed precision debugging

### Gradient Accumulation (Manual)

Effective batch size = batch_size × accumulation_steps

---

## Training Schedule

### Typical Schedule (100 epochs)

| Phase | Epochs | Description |
|-------|--------|-------------|
| Warmup | 0-5 | LR ramps up |
| Training | 5-90 | Cosine decay |
| Final | 90-100 | LR near minimum |

```bash
uv run python train.py --warmup_epochs 5 --epochs 100
```

---

## Monitoring Training

### Watch for Overfitting

- Train loss decreasing, val loss increasing
- Train accuracy >> val accuracy
- **Solution**: Early stopping, more regularization

### Watch for Underfitting

- Both losses plateau high
- Low accuracy
- **Solution**: Train longer, larger model, lower regularization

### Good Signs

- Both losses decreasing
- Gap between train/val stable
- Metrics improving on validation

---

## Quick Experiments

```bash
# Fast experiment (1 epoch, small batch)
uv run python train.py --model resnet50 --dataset nih --epochs 1 --batch_size 4

# Resume interrupted training
uv run python train.py --retrain <run_folder_name>
```

---

## Model Selection

### By Compute Budget

| Budget | Recommended Models |
|--------|-------------------|
| Low | `mobilenetv3_small`, `efficientnet_b0` |
| Medium | `resnet50`, `vit_small_patch16` |
| High | `swin_base`, `vit_base_patch16` |

### By Task

| Task | Recommended |
|------|-------------|
| Quick baseline | `resnet50` + linear probe |
| Best accuracy | `swin_base` + fine-tune |
| Fast inference | `mobilenetv3_small` |
| Transformers | `vit_small_patch16` |
