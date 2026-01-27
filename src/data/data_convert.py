"""
LMDB Data Converter Module.

This module provides utilities for converting image datasets to LMDB format
with parallel processing support for efficient data loading during training.
"""

import io
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import lmdb
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def process_one_image(
    file_path: str,
    img_size: Tuple[int, int],
    quality: int = 95
) -> Optional[bytes]:
    """
    Process a single image file: load, convert to RGB, resize, and encode as JPEG bytes.

    Args:
        file_path: Path to the image file.
        img_size: Target size as (width, height) tuple.
        quality: JPEG compression quality (1-100). Default is 95.

    Returns:
        JPEG encoded image bytes if successful, None otherwise.
    """
    try:
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            img = img.resize(img_size, resample=Image.LANCZOS)

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=quality)
            return img_byte_arr.getvalue()
    except Exception as e:
        logger.warning(f"Failed to process image {file_path}: {e}")
        return None


def get_image_files(data_path: str) -> list:
    """
    Get all supported image files from a directory.

    Args:
        data_path: Path to the directory containing images.

    Returns:
        Sorted list of image filenames with supported extensions.
    """
    image_files = [
        f for f in os.listdir(data_path)
        if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    image_files.sort()
    return image_files


def create_lmdb_fast(
    data_path: str,
    lmdb_path: str,
    img_size: Tuple[int, int] = (224, 224),
    quality: int = 95,
    n_jobs: int = -1
) -> None:
    """
    Convert a directory of images to LMDB format with parallel processing.

    Args:
        data_path: Path to the directory containing source images.
        lmdb_path: Path where the LMDB database will be created.
        img_size: Target image size as (width, height). Default is (224, 224).
        quality: JPEG compression quality (1-100). Default is 95.
        n_jobs: Number of parallel jobs. -1 uses all available CPU cores.

    Raises:
        ValueError: If no valid images are found in data_path.
    """
    image_files = get_image_files(data_path)

    if not image_files:
        raise ValueError(f"No valid images found in {data_path}")

    logger.info(f"Found {len(image_files)} images in {data_path}")
    logger.info(f"Processing images with {n_jobs} workers (size={img_size}, quality={quality})...")

    processed_imgs = Parallel(n_jobs=n_jobs)(
        delayed(process_one_image)(
            os.path.join(data_path, f),
            img_size,
            quality
        )
        for f in tqdm(image_files, desc="Processing images")
    )

    valid_count = sum(1 for img in processed_imgs if img is not None)
    failed_count = len(processed_imgs) - valid_count

    if failed_count > 0:
        logger.warning(f"Failed to process {failed_count} images")

    logger.info(f"Writing {valid_count} images to LMDB at {lmdb_path}...")

    env = lmdb.open(lmdb_path, map_size=int(1e12))

    with env.begin(write=True) as txn:
        write_idx = 0
        for img_bytes in tqdm(processed_imgs, desc="Writing to LMDB"):
            if img_bytes is not None:
                key = f"{write_idx}".encode('ascii')
                txn.put(key, img_bytes)
                write_idx += 1

        txn.put('length'.encode('ascii'), str(write_idx).encode('ascii'))

    env.close()
    logger.info(f"LMDB creation complete! Total images: {write_idx}")