# GenMed-Bench 🏥

A benchmarking framework for **Domain Generalization in Medical Image Analysis**, specifically designed for chest X-ray datasets. This project provides efficient data pipelines and training infrastructure for evaluating domain generalization methods across multiple medical imaging sources.

## ✨ Features

- **Multi-Dataset Support**: NIH ChestX-ray14, COVIDx, VinBigData
- **Efficient Data Pipeline**: LMDB-based storage with parallel processing for fast I/O
- **DVC Integration**: Version-controlled data management with Google Drive remote
- **Training Infrastructure**: Supervised and self-supervised learning trainers
- **Interpretability**: GradCAM visualization support

## 📁 Project Structure

```
genmed-bench/
├── configs/            # YAML configuration files
├── data/
│   ├── raw/            # Symlinks to source datasets
│   ├── processed/      # LMDB databases
│   ├── splits/         # Train/val/test splits
│   └── stats/          # Normalization stats (.npy)
├── notebooks/          # Jupyter notebooks for exploration
├── scripts/            # Data preparation & evaluation scripts
├── src/
│   ├── data/           # Dataset classes & data loading
│   ├── models/         # Model architectures
│   ├── train/          # Training loops (supervised & SSL)
│   └── utils/          # Logging, metrics, GradCAM
└── logs/               # Training logs
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/EnesDemir143/genmed-bench.git
cd genmed-bench

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Data Preparation

1. **Link your raw data**:
   ```bash
   ln -s /path/to/your/chest-datasets data/raw/chest-datasets
   ```

2. **Convert to LMDB format**:
   ```bash
   # Single dataset
   python scripts/prepare_data.py --dataset nih --image-size 256 --quality 95

   # All datasets
   python scripts/prepare_data.py --dataset all
   ```

3. **Prepare metadata**:
   ```bash
   python scripts/prepare_metadata.py --dataset all
   ```

4. **Compute normalization stats** (mean/std for each dataset):
   ```bash
   python -m src.data.compute_stats --lmdb_path data/processed/nih/nih.lmdb
   # Saves to data/stats/nih.npy
   ```

### Using DVC

```bash
# Pull processed data from remote
dvc pull

# Run data pipeline
dvc repro
```

## 📊 Supported Datasets

| Dataset | Images | Domain | Task |
|---------|--------|--------|------|
| NIH ChestX-ray14 | ~112K | Hospital A | Multi-label classification |
| COVIDx | ~30K | Mixed sources | COVID-19 detection |
| VinBigData | ~18K | Vietnamese hospitals | Abnormality detection |

## 🛠️ Configuration

All settings are managed through `configs/config.yaml`:

```yaml
data:
  raw_root: "data/raw/chest-datasets"
  processed_root: "data/processed"
  datasets:
    nih:
      raw: "${data.raw_root}/nih"
      processed: "${data.processed_root}/nih/nih.lmdb"

preprocessing:
  image_size: 256
  normalize: true
```

## 📦 Dependencies

Core libraries:
- **PyTorch** + **torchvision** - Deep learning framework
- **timm** - Pre-trained models
- **albumentations** - Image augmentations
- **LMDB** - Fast key-value storage
- **DVC** - Data version control

## 📝 License

MIT License

## 🙏 Acknowledgments

- [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) for ChestX-ray14
- [COVID-Net](https://github.com/lindawangg/COVID-Net) for COVIDx
- [VinBigData](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) for VinDr-CXR
