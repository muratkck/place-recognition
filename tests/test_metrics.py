"""
Unit tests for evaluation metrics (src.evaluate.calculate_ap).

Covers single-positive, multi-positive, and no-relevant query cases
as required by the project spec.
"""

from src.evaluate import calculate_ap


def test_ap_perfect_single():
    """Single relevant item at rank 1 → AP = 1.0."""
    retrieved = ["A", "B", "C"]
    ap = calculate_ap(retrieved, gt_class="A", total_relevant_in_gallery=1)
    assert ap == 1.0


def test_ap_multi_positive():
    """
    Two relevant items at ranks 1 and 3, with R=2 (multi-positive query).

    AP = (1/1 + 2/3) / 2 = 0.8333...
    """
    retrieved = ["A", "X", "A"]
    ap = calculate_ap(retrieved, gt_class="A", total_relevant_in_gallery=2)
    expected = (1.0 + 2 / 3) / 2
    assert abs(ap - expected) < 1e-9


def test_ap_no_relevant():
    """When total_relevant_in_gallery is 0, AP should be 0.0."""
    retrieved = ["A", "B"]
    ap = calculate_ap(retrieved, gt_class="A", total_relevant_in_gallery=0)
    assert ap == 0.0
