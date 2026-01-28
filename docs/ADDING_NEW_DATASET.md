# Adding a New Dataset

> Step-by-step guide to add a new medical imaging dataset to GenMed-Bench.

---

## Steps Overview

| Step | Description | Required |
|------|-------------|----------|
| 1 | Download dataset | ✅ |
| 2 | Explore data (CSV/metadata analysis) | ✅ |
| 3 | Write LMDB converter | ✅ |
| 4 | Write dataset class | ✅ |
| 5 | Add to `__init__.py` | ✅ |
| 6 | Add to `train.py` | ✅ |
| 7 | Test | ✅ |

---

## Step 1: Download Dataset

```bash
# Example: From Kaggle
kaggle datasets download -d <username>/<dataset-name>

# Extract
unzip <dataset-name>.zip -d data/raw/<dataset_name>/
```

**Directory structure:**
```
data/raw/<dataset_name>/
├── images/          # Images
├── train.csv        # or labels.csv, metadata.csv, etc.
└── ...
```

---

## Step 2: Explore Data

### 2.1 Analyze CSV/Metadata

```python
import pandas as pd

df = pd.read_csv('data/raw/<dataset_name>/labels.csv')
print(df.head())
print(df.columns.tolist())
print(df['label_column'].value_counts())
```

### 2.2 Information to Find

| Info | Example | Notes |
|------|---------|-------|
| **Image ID column** | `image_id`, `filename`, `Image Index` | Used as LMDB key |
| **Label column** | `label`, `Finding`, `class`, `target` | Model output |
| **Multi-label?** | `"Pneumonia\|Edema"` or separate columns | Determines BCE vs CE loss |
| **Image format** | PNG, JPEG, DICOM | Processed in converter |

### 2.3 Label Mapping

```python
# Binary
label_map = {'normal': 0, 'abnormal': 1}

# Multi-class
label_map = {'COVID': 0, 'Normal': 1, 'Pneumonia': 2}

# Multi-label (14 diseases)
diseases = ['Atelectasis', 'Cardiomegaly', ...]
# Each row becomes [0,1,0,0,...] vector
```

---

## Step 3: Write LMDB Converter

**File:** `src/data/converters/<dataset_name>_converter.py`

```python
"""
<DatasetName> -> LMDB Converter

Usage:
    python -m src.data.converters.<dataset_name>_converter
"""

import lmdb
import pandas as pd
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm


def convert_to_lmdb(
    image_dir: str,
    metadata_path: str,
    output_path: str,
    image_size: int = 224,
):
    """Convert dataset to LMDB format."""
    df = pd.read_csv(metadata_path)
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    lmdb_path = output_path / '<dataset_name>.lmdb'
    
    env = lmdb.open(
        str(lmdb_path),
        map_size=50 * 1024 * 1024 * 1024,  # 50GB
        readonly=False,
    )
    
    with env.begin(write=True) as txn:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Image ID (used as LMDB key)
            image_id = row['<IMAGE_ID_COLUMN>']  # ← COLUMN NAME HERE
            
            # Load image
            img_path = image_dir / f"{image_id}.png"
            if not img_path.exists():
                continue
            
            img = Image.open(img_path).convert('RGB')
            img = img.resize((image_size, image_size))
            
            # Encode as JPEG
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            img_bytes = buffer.getvalue()
            
            # Write to LMDB
            txn.put(image_id.encode(), img_bytes)
    
    env.close()
    
    # Save metadata
    df.to_parquet(output_path / '<dataset_name>_all.parquet', index=False)
    print(f"✅ LMDB: {lmdb_path}")


if __name__ == '__main__':
    convert_to_lmdb(
        image_dir='data/raw/<dataset_name>/images',
        metadata_path='data/raw/<dataset_name>/labels.csv',
        output_path='data/processed/<dataset_name>',
    )
```

**Run:**
```bash
uv run python -m src.data.converters.<dataset_name>_converter
```

---

## Step 4: Write Dataset Class

**File:** `src/data/dataset/<dataset_name>_dataset.py`

```python
"""<DatasetName> Dataset class."""

from typing import Optional, Callable
from .base import LMDBDataset


class <DatasetName>Dataset(LMDBDataset):
    """LMDB dataset for <DatasetName>."""
    
    def __init__(
        self,
        lmdb_path: str,
        metadata_path: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__(lmdb_path, metadata_path, transform)
    
    def _get_label(self, row) -> int:
        """Extract label from row."""
        # ========================================
        # WRITE LABEL EXTRACTION CODE HERE
        # ========================================
        
        # Example 1: Binary
        return int(row['label'])
        
        # Example 2: Multi-class (string → int)
        # label_map = {'COVID': 0, 'Normal': 1, 'Pneumonia': 2}
        # return label_map[row['class']]
        
        # Example 3: Multi-label (pipe-separated)
        # diseases = ['Atelectasis', 'Cardiomegaly', ...]
        # findings = str(row['findings']).split('|')
        # return [1 if d in findings else 0 for d in diseases]
    
    def _get_key(self, row) -> str:
        """Return LMDB key (image ID)."""
        return str(row['image_id'])  # or 'filename', 'Image Index'
```

---

## Step 5: Add to `__init__.py`

**File:** `src/data/dataset/__init__.py`

```python
from .<dataset_name>_dataset import <DatasetName>Dataset  # ← ADD

__all__ = [
    ...
    '<DatasetName>Dataset',  # ← ADD
]
```

---

## Step 6: Add to `train.py`

### 6.1 Import

```python
from src.data.dataset import <DatasetName>Dataset
```

### 6.2 Add case in `get_dataset()`

```python
elif dataset_name == '<dataset_name>':
    lmdb_path = processed_path / '<dataset_name>' / '<dataset_name>.lmdb'
    return <DatasetName>Dataset(
        lmdb_path=str(lmdb_path),
        metadata_path=str(metadata_path),
        transform=transform,
    )
```

### 6.3 Set `num_classes`

```python
elif args.dataset == '<dataset_name>':
    num_classes = <N>
    model_config['multi_label'] = <True/False>
```

### 6.4 Update CLI choices

```python
choices=['nih', 'covidx', 'vinbigdata', '<dataset_name>']
```

---

## Step 7: Test

```bash
# Dataset test
uv run python -c "
from src.data.dataset import <DatasetName>Dataset
ds = <DatasetName>Dataset('data/processed/<dataset_name>/<dataset_name>.lmdb', 'data/splits/<dataset_name>/val_0.2/train.parquet')
print(len(ds), ds[0])
"

# Training test
uv run python train.py --model resnet50 --dataset <dataset_name> --epochs 1
```

---

## Final Directory Structure

```
data/
├── raw/<dataset_name>/           # Downloaded raw data
├── processed/<dataset_name>/     # LMDB and metadata
│   ├── <dataset_name>.lmdb/
│   └── <dataset_name>_all.parquet
└── splits/<dataset_name>/        # Train/val splits
    └── val_0.2/
        ├── train.parquet
        └── val.parquet

src/data/
├── converters/<dataset_name>_converter.py
└── dataset/<dataset_name>_dataset.py
```

---

## Existing Examples

| Dataset | File | Label Type |
|---------|------|------------|
| NIH ChestX-ray | `nih_dataset.py` | Multi-label (14) |
| COVIDx | `covidx_dataset.py` | Multi-class (3) |
| VinBigData | `vinbigdata_dataset.py` | Binary |
