import argparse
import io
from pathlib import Path
from typing import Tuple

import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_mean_std(
    lmdb_path: str,
    num_samples: int = None,
    image_size: int = 224,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute channel-wise mean and std for an LMDB dataset.
    
    Args:
        lmdb_path: Path to LMDB database.
        num_samples: Number of samples to use (None = all).
        image_size: Resize images to this size.
    
    Returns:
        Tuple of (mean, std) as numpy arrays of shape (3,).
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    with env.begin(write=False) as txn:
        length = int(txn.get(b'length').decode('ascii'))
    
    if num_samples is not None:
        length = min(length, num_samples)
    
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    
    print(f"Computing statistics for {length} images...")
    
    with env.begin(write=False) as txn:
        for idx in tqdm(range(length), desc="Processing"):
            key = str(idx).encode('ascii')
            img_bytes = txn.get(key)
            
            if img_bytes is None:
                continue
            
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img = img.resize((image_size, image_size), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float64) / 255.0
            
            channel_sum += img_array.sum(axis=(0, 1))
            channel_sum_sq += (img_array ** 2).sum(axis=(0, 1))
            total_pixels += image_size * image_size
    
    env.close()
    
    mean = channel_sum / total_pixels
    variance = (channel_sum_sq / total_pixels) - (mean ** 2)
    std = np.sqrt(variance)
    
    return mean, std


def main():
    parser = argparse.ArgumentParser(description='Compute mean/std for LMDB dataset')
    parser.add_argument('--lmdb_path', type=str, required=True, help='Path to LMDB database')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples (None=all)')
    parser.add_argument('--image_size', type=int, default=224, help='Image size for computation')
    parser.add_argument('--output', type=str, default=None, help='Output .npy path (default: data/stats/{dataset_name}.npy)')
    args = parser.parse_args()
    
    mean, std = compute_mean_std(args.lmdb_path, args.num_samples, args.image_size)
    
    print("\n" + "="*50)
    print("Results:")
    print("="*50)
    print(f"Mean (RGB): [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"Std (RGB):  [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    print("\nFor PyTorch transforms.Normalize():")
    print(f"transforms.Normalize(")
    print(f"    mean=[{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}],")
    print(f"    std=[{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print(f")")
    
    # Save to .npy
    if args.output:
        output_path = Path(args.output)
    else:
        dataset_name = Path(args.lmdb_path).parent.name
        output_path = Path("data/stats") / f"{dataset_name}.npy"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = np.array([mean, std])  # Shape: (2, 3) - [mean_rgb, std_rgb]
    np.save(output_path, stats)
    
    print(f"\nSaved to: {output_path}")
    
    return mean, std


if __name__ == '__main__':
    main()
