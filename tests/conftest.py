"""Shared pytest fixtures for the place-recognition test suite."""

import csv
import numpy as np
import pytest
import torch
import cv2


@pytest.fixture
def gallery_data():
    """
    Builds a synthetic gallery with 5 items.
    Vectors are L2-normalized so dot-product == cosine similarity.
    """
    np.random.seed(42)
    raw = np.random.randn(5, 128).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vectors = raw / norms  # L2-normalized

    return {
        "embeddings": torch.from_numpy(vectors),
        "classes": ["park", "bridge", "tower", "park", "bridge"],
        "ids": ["img_001", "img_002", "img_003", "img_004", "img_005"],
        "paths": [
            "landmarks/park/001.jpg",
            "landmarks/bridge/002.jpg",
            "landmarks/tower/003.jpg",
            "landmarks/park/004.jpg",
            "landmarks/bridge/005.jpg",
        ],
    }


@pytest.fixture
def tmp_manifest(tmp_path):
    """
    Creates a temporary manifest.csv and matching tiny JPEG files.
    Returns (csv_path, root_dir).
    """
    # Create image directories
    for folder in ("landmarks/parkA", "landmarks/parkB"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)

    filenames = [
        "landmarks/parkA/g1.jpg",
        "landmarks/parkA/g2.jpg",
        "landmarks/parkB/q1.jpg",
    ]
    for fname in filenames:
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / fname), img)

    csv_path = tmp_path / "manifest.csv"
    rows = [
        {
            "image_id": "g1",
            "class_name": "parkA",
            "relpath": "landmarks/parkA/g1.jpg",
            "split": "gallery",
        },
        {
            "image_id": "g2",
            "class_name": "parkA",
            "relpath": "landmarks/parkA/g2.jpg",
            "split": "gallery",
        },
        {
            "image_id": "q1",
            "class_name": "parkB",
            "relpath": "landmarks/parkB/q1.jpg",
            "split": "query",
        },
        # Non-existent file (should be skipped)
        {
            "image_id": "missing",
            "class_name": "parkA",
            "relpath": "landmarks/parkA/nope.jpg",
            "split": "gallery",
        },
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "class_name", "relpath", "split"])
        writer.writeheader()
        writer.writerows(rows)

    return str(csv_path), str(tmp_path)
