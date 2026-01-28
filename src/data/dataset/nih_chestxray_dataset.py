import io
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple
import lmdb
from PIL import Image

class NIHChestXrayDataset(Dataset):
    """
    NIH Chest X-ray14 dataset with multi-label classification.
    """
    CLASSES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    def __init__(self, lmdb_path: str, metadata_path: str, transform: Optional[Callable] = None, index_mapping_path: Optional[str] = None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.num_classes = len(self.CLASSES)
        self._env = None
        self.metadata = pd.read_parquet(metadata_path)
        if index_mapping_path and Path(index_mapping_path).exists():
            self.index_map = pd.read_parquet(index_mapping_path)
            self.index_map = dict(zip(self.index_map['image_index'], self.index_map['lmdb_key']))
        else:
            if 'lmdb_idx' in self.metadata.columns:
                self.index_map = dict(zip(self.metadata['image_index'], self.metadata['lmdb_idx']))
            else:
                self.index_map = None
        self.metadata = self.metadata.reset_index(drop=True)
    def _init_lmdb(self):
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    def __len__(self) -> int:
        return len(self.metadata)
    def _get_lmdb_key(self, idx: int) -> str:
        if self.index_map is not None:
            image_index = self.metadata.iloc[idx]['image_index']
            return str(self.index_map[image_index])
        else:
            if 'lmdb_idx' in self.metadata.columns:
                return str(self.metadata.iloc[idx]['lmdb_idx'])
            return str(idx)
    def _get_labels(self, findings: str) -> torch.Tensor:
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        if findings == 'No Finding':
            return labels
        for finding in findings.split('|'):
            finding = finding.strip()
            if finding in self.CLASSES:
                labels[self.CLASSES.index(finding)] = 1.0
        return labels
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._init_lmdb()
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
        if self._env is not None:
            self._env.close()
            self._env = None
