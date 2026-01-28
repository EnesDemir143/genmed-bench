from torch.utils.data import Dataset
from typing import Callable

class SSLDataset(Dataset):
    """
    Self-Supervised Learning dataset wrapper.
    Applies two different augmentations to the same image for contrastive learning.
    Args:
        base_dataset: Base LMDB dataset (LMDBDataset, NIHChestXrayDataset, etc.).
        transform1: First augmentation pipeline.
        transform2: Second augmentation pipeline.
    """
    def __init__(self, base_dataset: Dataset, transform1: Callable, transform2: Callable):
        self.base_dataset = base_dataset
        self.transform1 = transform1
        self.transform2 = transform2
        self._original_transform = getattr(base_dataset, 'transform', None)
        base_dataset.transform = None
    def __len__(self) -> int:
        return len(self.base_dataset)
    def __getitem__(self, idx: int):
        item = self.base_dataset[idx]
        if isinstance(item, tuple):
            img = item[0]
        else:
            img = item
        view1 = self.transform1(img)
        view2 = self.transform2(img)
        return view1, view2
    def close(self):
        if hasattr(self.base_dataset, 'close'):
            self.base_dataset.close()
