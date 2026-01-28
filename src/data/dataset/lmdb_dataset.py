import io
import lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional

class LMDBDataset(Dataset):
    """
    Base dataset class for LMDB-backed image data.
    """
    def __init__(self, lmdb_path: str, transform: Optional[Callable] = None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self._env = None
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self._length = int(txn.get(b'length').decode('ascii'))
        env.close()
    def _init_lmdb(self):
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    @property
    def env(self):
        self._init_lmdb()
        return self._env
    def __len__(self) -> int:
        return self._length
    def __getitem__(self, idx: int) -> torch.Tensor:
        self._init_lmdb()
        with self._env.begin(write=False) as txn:
            key = f"{idx}".encode('ascii')
            img_bytes = txn.get(key)
        if img_bytes is None:
            raise KeyError(f"Image with index {idx} not found in LMDB")
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
    def __del__(self):
        self.close()
