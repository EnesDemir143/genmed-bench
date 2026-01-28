# Reproducibility Guide

> How to ensure reproducible results in GenMed-Bench experiments.

---

## Quick Start

```bash
# Set seed via CLI (recommended)
uv run python train.py --model resnet50 --dataset nih --seed 42
```

---

## Seed System

GenMed-Bench uses comprehensive seeding through `src/utils/seed.py`:

```python
from src.utils.seed import set_seed, get_generator, worker_init_fn

# Set all seeds
set_seed(42)

# Get reproducible generator for DataLoader
g = get_generator(42)

# Use in DataLoader
DataLoader(dataset, generator=g, worker_init_fn=worker_init_fn)
```

### What Gets Seeded

| Component | Seeded |
|-----------|--------|
| Python random | ✅ |
| NumPy | ✅ |
| PyTorch CPU | ✅ |
| PyTorch CUDA | ✅ |
| PyTorch MPS | ✅ |
| PYTHONHASHSEED | ✅ |
| DataLoader workers | ✅ |

---

## Deterministic Mode

For fully deterministic results:

```python
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

> **Note**: Deterministic mode may reduce performance.

---

## DataLoader Reproducibility

```python
from src.utils.seed import get_generator, worker_init_fn

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    generator=get_generator(seed),   # Reproducible shuffling
    worker_init_fn=worker_init_fn,   # Seed each worker
    num_workers=4,
)
```

---

## Experiment Tracking

Each run saves configuration for reproducibility:

```
runs/<experiment>/
├── config.yaml      # All hyperparameters
├── train.log        # Training log with seed info
└── checkpoints/     # Saved models
```

### Resume Training

```bash
# Resume from exact state
uv run python train.py --retrain <run_folder_name>
```

This restores:
- Model weights
- Optimizer state
- Scheduler state
- Training epoch
- Configuration

---

## Checklist for Reproducibility

- [x] Set `--seed` argument
- [x] Use same Python/PyTorch/CUDA version
- [x] Use same hardware (GPU model)
- [x] Use same `--num_workers`
- [x] Save and use `config.yaml`
- [x] Use same data split (`--val_ratio`)

---

## Known Non-Deterministic Operations

Some operations are inherently non-deterministic:

| Operation | Workaround |
|-----------|------------|
| CUDA atomics | Use deterministic mode |
| Multi-threaded data loading | Use `num_workers=0` |
| Batch norm running stats | Freeze during evaluation |
| Dropout | Seeds are set per-forward |

---

## Environment Logging

Training logs include environment info:

```
[2026-01-28 19:30:00] Seed: 42
[2026-01-28 19:30:00] PyTorch: 2.1.0
[2026-01-28 19:30:00] Device: cuda:0 (NVIDIA A100)
```

---

## Recommendations

1. **Always set seed**: `--seed 42`
2. **Save config**: Automatically done in `runs/`
3. **Use retrain**: For resuming experiments
4. **Document hardware**: Note GPU model
5. **Version control**: Track code with git
