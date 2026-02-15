"""
Unit tests for VectorIndex search (src.index.VectorIndex).
"""

from src.index import VectorIndex


def test_search_top1_exact_match(gallery_data):
    """Querying with an exact gallery vector should return that item as top-1."""
    index = VectorIndex(gallery_data)

    query = gallery_data["embeddings"][0]
    results = index.search(query, top_k=1, threshold=0.0)

    assert len(results) == 1
    assert results[0]["id"] == "img_001"
    assert results[0]["class"] == "park"


def test_search_top_k_ordering(gallery_data):
    """Results should be sorted by descending similarity score."""
    index = VectorIndex(gallery_data)
    query = gallery_data["embeddings"][0]

    results = index.search(query, top_k=5, threshold=0.0)
    scores = [r["score"] for r in results]

    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]
