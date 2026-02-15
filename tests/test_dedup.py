"""
Unit tests for near-duplicate detection (src.dedup).

Uses small synthetic images created with NumPy so no real files are needed.
"""

import tempfile
import os
import cv2
import numpy as np
from src.dedup import compute_phash, hamming_distance


def _save_solid_image(path: str, color: int, size: int = 64):
    """Helper: creates a solid grayscale image."""
    img = np.full((size, size), color, dtype=np.uint8)
    cv2.imwrite(path, img)


def test_identical_images_zero_distance():
    """Two identical images should produce the same hash (distance = 0)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "img.jpg")
        _save_solid_image(path, color=128)

        h1 = compute_phash(path)
        h2 = compute_phash(path)

        assert h1 is not None
        assert hamming_distance(h1, h2) == 0


def test_different_images_large_distance():
    """Very different images should have a large Hamming distance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        white_path = os.path.join(tmpdir, "white.jpg")
        noise_path = os.path.join(tmpdir, "noise.jpg")

        _save_solid_image(white_path, color=255)

        rng = np.random.RandomState(42)
        noise = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        cv2.imwrite(noise_path, noise)

        h_white = compute_phash(white_path)
        h_noise = compute_phash(noise_path)

        assert h_white is not None and h_noise is not None
        assert hamming_distance(h_white, h_noise) > 10


def test_hamming_distance_known_values():
    """Direct test of hamming_distance with known bit patterns."""
    assert hamming_distance(0b0000, 0b0000) == 0
    assert hamming_distance(0b1111, 0b0000) == 4
    assert hamming_distance(0b1010, 0b0101) == 4
    assert hamming_distance(0b1111, 0b1110) == 1
