import io
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional
import lmdb
from PIL import Image

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
