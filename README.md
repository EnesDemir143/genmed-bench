# GenMed-Bench 🏥

A benchmarking framework for **Medical Image Classification** with focus on domain generalization. Provides efficient LMDB-based data pipelines, multiple model architectures (ViT, ResNet, Swin), and advanced augmentation methods (XDomainMix, PipMix).

## ✨ Features

- **Multi-Dataset Support**: NIH ChestX-ray14, COVIDx, VinBigData (easily extensible)
- **Efficient Data Pipeline**: LMDB-based storage for fast I/O
- **Multiple Models**: ViT, Swin Transformer, ResNet, MobileNet, EfficientNet via `timm`
- **Training Modes**: Linear Probe & Full Fine-tuning
- **Augmentations**: XDomainMix, PipMix for domain generalization
- **Comprehensive Logging**: Metrics CSV, loss plots, confusion matrix, ROC curves
- **Checkpoint Management**: Best/last checkpoints with resume support

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/EnesDemir143/genmed-bench.git
cd genmed-bench

# Install with uv (recommended)
uv sync
```

### 2. Prepare Data

```bash
# Link raw data
ln -s /path/to/chest-datasets data/raw/

# Convert to LMDB
uv run python -m src.data.converters.nih_converter
uv run python -m src.data.converters.covidx_converter
uv run python -m src.data.converters.vinbigdata_converter
```

### 3. Train Model

```bash
# Basic training
uv run python train.py \
    --model resnet50 \
    --dataset nih \
    --mode linear_probe \
    --epochs 50

# Full example with all options
uv run python train.py \
    --model vit_small_patch16 \
    --mode linear_probe \
    --augmentation xdomainmix \
    --dataset nih \
    --batch_size 64 \
    --epochs 100 \
    --val_ratio 0.2 \
    --multi_label
```

### 4. Results

Training creates a run folder with all artifacts:

```
runs/<model>_<mode>_<augmentation>_<dataset>_<timestamp>/
├── checkpoints/
│   ├── best.pth          # Best validation AUC
│   └── last.pth          # Last epoch
├── plots/
│   ├── loss_curve.png
│   ├── roc_curve.png
│   └── confusion_matrix.png
├── metrics.csv           # Per-epoch metrics
├── config.yaml           # Saved configuration
└── train.log             # Training logs
```

---

## 📋 CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Model name (e.g., `resnet50`, `vit_small_patch16`, `swin_tiny`) |
| `--dataset` | required | Dataset (`nih`, `covidx`, `vinbigdata`) |
| `--mode` | `linear_probe` | Training mode (`linear_probe`, `full_finetune`) |
| `--augmentation` | `none` | Augmentation (`none`, `xdomainmix`, `pipmix`) |
| `--epochs` | from config | Number of epochs |
| `--batch_size` | from config | Batch size |
| `--lr` | from config | Learning rate |
| `--val_ratio` | `0.2` | Validation split ratio |
| `--multi_label` | `False` | Multi-label classification |
| `--early_stopping` | `0` | Early stopping patience (0=disabled) |
| `--retrain` | `None` | Resume from run folder name |

---

## 🧠 Supported Models

| Category | Models |
|----------|--------|
| **Vision Transformers** | `vit_small_patch16`, `vit_base_patch16`, `deit_small_patch16` |
| **Swin Transformer** | `swin_tiny_patch4`, `swin_small_patch4` |
| **CNNs** | `resnet50`, `resnet101`, `efficientnet_b0`, `mobilenetv3_small` |
| **ConvNeXt** | `convnext_tiny`, `convnext_small` |

All models are loaded from `timm` with ImageNet pretrained weights.

---

## 📊 Datasets

| Dataset | Images | Classes | Type |
|---------|--------|---------|------|
| NIH ChestX-ray14 | ~112K | 14 | Multi-label |
| COVIDx | ~30K | 3 | Multi-class |
| VinBigData | ~18K | 2 | Binary |

### Adding New Datasets

See [docs/ADDING_NEW_DATASET.md](docs/ADDING_NEW_DATASET.md) for step-by-step guide.

---

## 📖 Documentation

| Guide | Description |
|-------|-------------|
| [Adding New Datasets](docs/ADDING_NEW_DATASET.md) | How to add a new medical imaging dataset |
| [Adding Models & Augmentations](docs/ADDING_MODELS_AND_AUGMENTATIONS.md) | How to add new models and augmentation methods |
| [Configuration](docs/CONFIGURATION.md) | Complete reference for config.yaml and models.yaml |
| [Training Tips](docs/TRAINING_TIPS.md) | Best practices and hyperparameter recommendations |
| [Reproducibility](docs/REPRODUCIBILITY.md) | How to ensure reproducible experiments |

---

## 📁 Project Structure

```
genmed-bench/
├── configs/
│   ├── config.yaml       # Data paths, preprocessing
│   └── models.yaml       # Model-specific hyperparameters
├── data/
│   ├── raw/              # Original datasets (symlinks)
│   ├── processed/        # LMDB databases
│   └── splits/           # Train/val splits (parquet)
├── src/
│   ├── data/
│   │   ├── augmentation/ # XDomainMix, PipMix
│   │   ├── converters/   # Raw → LMDB converters
│   │   └── dataset/      # Dataset classes
│   ├── models/
│   │   ├── backbone.py   # Backbone loader
│   │   └── classifier.py # Classification head
│   ├── train/
│   │   ├── trainer_base.py
│   │   ├── trainer_sup.py
│   │   └── experiment_logger.py
│   └── utils/
│       ├── metrics.py    # AUC, F1, Confusion Matrix
│       └── seed.py       # Reproducibility
├── scripts/              # Data preparation scripts
├── runs/                 # Training outputs
└── train.py              # Main entry point
```

---

## 🔄 Resume Training

```bash
# Resume from run folder
uv run python train.py --retrain vit_small_patch16_linear_probe_baseline_nih_20260128_123456
```

This automatically loads:
- `config.yaml` from the run folder
- `checkpoints/last.pth`
- Continues logging to existing files

---

## 📦 Dependencies

- **PyTorch** 2.0+ with MPS/CUDA support
- **timm** - Pre-trained models
- **LMDB** - Fast data storage
- **scikit-learn** - Metrics

Install all with `uv sync`.

---

## 📝 License

MIT License

## 🙏 Acknowledgments

- [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) for ChestX-ray14
- [COVID-Net](https://github.com/lindawangg/COVID-Net) for COVIDx
- [VinBigData](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) for VinDr-CXR
