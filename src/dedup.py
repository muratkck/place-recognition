"""
Near-duplicate detection using pHash (perceptual hash).

Compares gallery vs query images within the same class to detect
potential data leakage from near-duplicate or repeated-frame images.
"""

import csv
import cv2
import numpy as np
from pathlib import Path
from src.logger import Logger

logger = Logger(__name__)


def compute_phash(image_path: str, hash_size: int = 8, dct_size: int = 32) -> int:
    """
    Computes a perceptual hash (pHash) for an image.

    Steps:
        1. Read in grayscale and resize to dct_size x dct_size.
        2. Apply 2D DCT (Discrete Cosine Transform).
        3. Keep only the top-left hash_size x hash_size block (low frequencies).
        4. Compute the median of those values.
        5. Set each bit to 1 if above the median, 0 otherwise.

    Returns:
        An integer representing the hash (hash_size * hash_size bits).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return None

    resized = cv2.resize(img, (dct_size, dct_size)).astype(np.float32)

    # Apply 2D DCT
    dct_result = cv2.dct(resized)

    # Keep top-left low-frequency block
    dct_low = dct_result[:hash_size, :hash_size]

    # Threshold by median
    median_val = np.median(dct_low)
    bits = (dct_low > median_val).flatten()

    # Convert to integer
    hash_val = 0
    for bit in bits:
        hash_val = (hash_val << 1) | int(bit)

    return hash_val


def hamming_distance(hash1: int, hash2: int) -> int:
    """Returns the number of differing bits between two hashes."""
    return bin(hash1 ^ hash2).count("1")


def check_leakage(csv_path: str, root_dir: str, threshold: int = 10) -> list[dict]:
    """
    Scans for near-duplicate images across gallery/query split.

    For each class, computes pHash for every gallery and query image,
    then compares all cross-split pairs. Pairs with Hamming distance
    <= threshold are flagged as potential leakage.

    Returns:
        A list of dicts, each with keys:
        'gallery_image', 'query_image', 'class_name', 'distance'.
    """
    root = Path(root_dir)
    # Group images by class and split
    classes = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row["class_name"].strip()
            split = row["split"].strip()
            relpath = row.get("relpath", "").strip()

            if split not in ("gallery", "query") or not relpath:
                continue

            if cls not in classes:
                classes[cls] = {"gallery": [], "query": []}
            classes[cls][split].append(relpath)

    flagged = []

    for cls, splits in classes.items():
        gallery_paths = splits["gallery"]
        query_paths = splits["query"]

        if not gallery_paths or not query_paths:
            continue

        logger.info(
            f"Checking class '{cls}': {len(gallery_paths)} gallery, {len(query_paths)} query"
        )

        # Hash values
        gallery_hashes = []
        for path in gallery_paths:
            hash_value = compute_phash(str(root / path))
            if hash_value is not None:
                gallery_hashes.append((path, hash_value))

        query_hashes = []
        for path in query_paths:
            hash_value = compute_phash(str(root / path))
            if hash_value is not None:
                query_hashes.append((path, hash_value))

        # Cross-compare
        for gallery_path, gallery_hash in gallery_hashes:
            for query_path, query_hash in query_hashes:
                dist = hamming_distance(gallery_hash, query_hash)
                if dist <= threshold:
                    logger.warning(f"Near-duplicate in '{cls}' (dist={dist})")
                    logger.warning(f"  gallery: {gallery_path}")
                    logger.warning(f"  query:   {query_path}")
                    flagged.append(
                        {
                            "gallery_image": gallery_path,
                            "query_image": query_path,
                            "class_name": cls,
                            "distance": dist,
                        }
                    )

    if flagged:
        logger.warning(f"Total: {len(flagged)} near-duplicate pair(s) found.")
    else:
        logger.info("No near-duplicates detected. Split is clean.")

    return flagged
