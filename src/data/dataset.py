import io
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import lmdb
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class LMDBDataset(Dataset):
    """
    Base dataset class for LMDB-backed image data.
    
    Uses lazy initialization for LMDB environment to support multiprocessing
    with DataLoader num_workers > 0.
    
    Args:
        lmdb_path: Path to the LMDB database directory.
        transform: Optional transform to apply to images.
    """
    
    def __init__(
        self,
        lmdb_path: str,
        transform: Optional[Callable] = None,
    ):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self._env = None  # Lazy init for multiprocessing support
        
        # Get dataset length (open temporarily, then close)
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self._length = int(txn.get(b'length').decode('ascii'))
        env.close()
    
    def _init_lmdb(self):
        """Lazy init LMDB - called in each worker process."""
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
    
    @property
    def env(self):
        """Get LMDB environment, initializing if needed."""
        self._init_lmdb()
        return self._env
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get image by index."""
        self._init_lmdb()  # Ensure LMDB is open in this worker
        
        with self._env.begin(write=False) as txn:
            key = f"{idx}".encode('ascii')
            img_bytes = txn.get(key)
        
        if img_bytes is None:
            raise KeyError(f"Image with index {idx} not found in LMDB")
        
        # Decode image
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img
    
    def close(self):
        """Close LMDB environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class NIHChestXrayDataset(Dataset):
    """
    NIH Chest X-ray14 dataset with multi-label classification.
    
    Uses lazy LMDB initialization for multiprocessing support.
    
    Args:
        lmdb_path: Path to the LMDB database directory.
        metadata_path: Path to the parquet metadata file (can be split-specific).
        transform: Optional transform to apply to images.
        index_mapping: Optional dict mapping metadata indices to LMDB keys.
    
    Labels (14 classes):
        Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
        Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
        Pleural_Thickening, Hernia
    """
    
    CLASSES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    def __init__(
        self,
        lmdb_path: str,
        metadata_path: str,
        transform: Optional[Callable] = None,
        index_mapping_path: Optional[str] = None,
    ):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.num_classes = len(self.CLASSES)
        self._env = None  # Lazy init for multiprocessing
        
        # Load metadata (can be split-specific file)
        self.metadata = pd.read_parquet(metadata_path)
        
        # Load or create index mapping (image_index -> LMDB key)
        if index_mapping_path and Path(index_mapping_path).exists():
            self.index_map = pd.read_parquet(index_mapping_path)
            self.index_map = dict(zip(
                self.index_map['image_index'], 
                self.index_map['lmdb_key']
            ))
        else:
            if 'lmdb_idx' in self.metadata.columns:
                self.index_map = dict(zip(
                    self.metadata['image_index'],
                    self.metadata['lmdb_idx']
                ))
            else:
                self.index_map = None
        
        self.metadata = self.metadata.reset_index(drop=True)
    
    def _init_lmdb(self):
        """Lazy init LMDB - called in each worker process."""
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def _get_lmdb_key(self, idx: int) -> str:
        """Get LMDB key for a metadata row index."""
        if self.index_map is not None:
            image_index = self.metadata.iloc[idx]['image_index']
            return str(self.index_map[image_index])
        else:
            if 'lmdb_idx' in self.metadata.columns:
                return str(self.metadata.iloc[idx]['lmdb_idx'])
            return str(idx)
    
    def _get_labels(self, findings: str) -> torch.Tensor:
        """Convert finding labels string to multi-hot tensor."""
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        
        if findings == 'No Finding':
            return labels
        
        for finding in findings.split('|'):
            finding = finding.strip()
            if finding in self.CLASSES:
                labels[self.CLASSES.index(finding)] = 1.0
        
        return labels
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and labels by index."""
        self._init_lmdb()  # Ensure LMDB is open in this worker
        
        row = self.metadata.iloc[idx]
        lmdb_key = self._get_lmdb_key(idx)
        
        with self._env.begin(write=False) as txn:
            key = lmdb_key.encode('ascii')
            img_bytes = txn.get(key)
        
        if img_bytes is None:
            raise KeyError(f"Image with index {idx} not found in LMDB")
        
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        labels = self._get_labels(row['finding_labels'])
        
        return img, labels
    
    def close(self):
        """Close LMDB environment."""
        if self._env is not None:
            self._env.close()
            self._env = None



class COVIDxDataset(Dataset):
    """COVIDx dataset with lazy LMDB init for multiprocessing."""
    
    CLASSES = ['negative', 'positive', 'COVID-19']
    
    def __init__(self, lmdb_path: str, metadata_path: str, transform: Optional[Callable] = None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.num_classes = len(self.CLASSES)
        self._env = None
        self.metadata = pd.read_parquet(metadata_path).reset_index(drop=True)
    
    def _init_lmdb(self):
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def _get_lmdb_key(self, idx: int) -> str:
        if 'lmdb_idx' in self.metadata.columns:
            return str(self.metadata.iloc[idx]['lmdb_idx'])
        return str(idx)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        self._init_lmdb()
        row = self.metadata.iloc[idx]
        lmdb_key = self._get_lmdb_key(idx)
        
        with self._env.begin(write=False) as txn:
            img_bytes = txn.get(lmdb_key.encode('ascii'))
        
        if img_bytes is None:
            raise KeyError(f"Image {idx} not found")
        
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, self.CLASSES.index(row['label'])
    
    def close(self):
        if self._env:
            self._env.close()
            self._env = None


class VinBigDataDataset(Dataset):
    """VinBigData dataset with lazy LMDB init for multiprocessing."""
    
    def __init__(self, lmdb_path: str, metadata_path: str, transform: Optional[Callable] = None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self._env = None
        self.metadata = pd.read_parquet(metadata_path).reset_index(drop=True)
    
    def _init_lmdb(self):
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def _get_lmdb_key(self, idx: int) -> str:
        if 'lmdb_idx' in self.metadata.columns:
            return str(self.metadata.iloc[idx]['lmdb_idx'])
        return str(idx)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        self._init_lmdb()
        lmdb_key = self._get_lmdb_key(idx)
        
        with self._env.begin(write=False) as txn:
            img_bytes = txn.get(lmdb_key.encode('ascii'))
        
        if img_bytes is None:
            raise KeyError(f"Image {idx} not found")
        
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img
    
    def close(self):
        if self._env:
            self._env.close()
            self._env = None


class SSLDataset(Dataset):
    """
    Self-Supervised Learning dataset wrapper.
    
    Applies two different augmentations to the same image for contrastive learning.
    
    Args:
        base_dataset: Base LMDB dataset (LMDBDataset, NIHChestXrayDataset, etc.).
        transform1: First augmentation pipeline.
        transform2: Second augmentation pipeline.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        transform1: Callable,
        transform2: Callable,
    ):
        self.base_dataset = base_dataset
        self.transform1 = transform1
        self.transform2 = transform2
        
        # Temporarily disable base transform
        self._original_transform = getattr(base_dataset, 'transform', None)
        base_dataset.transform = None
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get two augmented views of the same image."""
        item = self.base_dataset[idx]
        
        # Handle datasets that return (img, label) tuples
        if isinstance(item, tuple):
            img = item[0]
        else:
            img = item
        
        # Apply augmentations
        view1 = self.transform1(img)
        view2 = self.transform2(img)
        
        return view1, view2
    
    def close(self):
        """Close underlying dataset."""
        if hasattr(self.base_dataset, 'close'):
            self.base_dataset.close()


def get_dataset(
    dataset_name: str,
    lmdb_path: str,
    metadata_path: str,
    transform: Optional[Callable] = None,
) -> Dataset:
    """
    Factory function to create dataset by name.
    
    Args:
        dataset_name: One of 'nih', 'covidx', 'vinbigdata'.
        lmdb_path: Path to LMDB database.
        metadata_path: Path to parquet metadata (use split-specific file like 'data/splits/nih_train.parquet').
        transform: Optional image transforms.
    
    Returns:
        Dataset instance.
    
    Example:
        # Load train split
        train_dataset = get_dataset(
            dataset_name='nih',
            lmdb_path='data/processed/nih/nih.lmdb',
            metadata_path='data/splits/nih/train.parquet',
            transform=train_transforms
        )
        
        # Load validation split
        val_dataset = get_dataset(
            dataset_name='nih',
            lmdb_path='data/processed/nih/nih.lmdb',
            metadata_path='data/splits/nih/val.parquet',
            transform=val_transforms
        )
    """
    datasets = {
        'nih': NIHChestXrayDataset,
        'covidx': COVIDxDataset,
        'vinbigdata': VinBigDataDataset,
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(datasets.keys())}")
    
    dataset_cls = datasets[dataset_name]
    return dataset_cls(lmdb_path, metadata_path, transform=transform)
