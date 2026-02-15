"""
Unit tests for PlaceDataset (src.dataset).
"""

import torch
from src.dataset import PlaceDataset


def test_load_manifest_filters_split(tmp_manifest):
    """Only rows matching the requested split should be loaded."""
    csv_path, root_dir = tmp_manifest

    gallery_ds = PlaceDataset(csv_path, root_dir, split="gallery")
    query_ds = PlaceDataset(csv_path, root_dir, split="query")

    # gallery has 2 valid rows (g1, g2) — the "missing" row is skipped
    assert len(gallery_ds) == 2
    # query has 1 row (q1)
    assert len(query_ds) == 1


def test_getitem_returns_expected_keys(tmp_manifest):
    """__getitem__ should return a dict with image, image_id, class_name, relpath."""
    csv_path, root_dir = tmp_manifest
    ds = PlaceDataset(csv_path, root_dir, split="gallery")

    sample = ds[0]
    assert sample is not None
    assert set(sample.keys()) == {"image", "image_id", "class_name", "relpath"}
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 224, 224)
