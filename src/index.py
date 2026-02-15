import torch
import numpy as np
from src.logger import Logger

logger = Logger(__name__)


class VectorIndex:
    def __init__(self, gallery_data: dict):
        """
        Args:
            gallery_data (dict): Dictionary containing keys:
                - 'embeddings': Tensor of shape [N, D] (Normalized)
                - 'classes': List of class names
                - 'ids': List of image IDs
        """
        self.gallery_vectors = gallery_data["embeddings"]
        self.gallery_classes = gallery_data["classes"]
        self.gallery_ids = gallery_data["ids"]
        self.gallery_paths = gallery_data["paths"]

        # Ensure vectors are on CPU for numpy operations
        if torch.is_tensor(self.gallery_vectors):
            self.gallery_vectors = self.gallery_vectors.cpu().numpy()

        logger.info(f"Index built with {len(self.gallery_classes)} gallery images.")

    def search(self, query_vector: torch.Tensor, top_k: int = 1, threshold: float = 0.0):
        """
        Search the gallery for the query_vector.

        Args:
            query_vector (Tensor): Shape [D] or [1, D].
            top_k (int): Number of results to return.
            threshold (float): Similarity threshold for "UNKNOWN" detection.

        Returns:
            List of dicts: [{'class': str, 'score': float, 'id': str}, ...]
        """
        # Ensure query is numpy
        if torch.is_tensor(query_vector):
            query = query_vector.cpu().numpy().flatten()
        else:
            query = query_vector.flatten()

        scores = self.gallery_vectors @ query  # Cosine Similarity
        sorted_indices = np.argsort(-scores)  # Negative for descending

        results = []
        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i]
            score = float(scores[idx])

            # Unknown Handling
            if score < threshold:
                pred_class = "UNKNOWN"
            else:
                pred_class = self.gallery_classes[idx]

            results.append(
                {
                    "rank": i + 1,
                    "class": pred_class,
                    "score": round(score, 4),
                    "id": self.gallery_ids[idx],
                    "path": self.gallery_paths[idx],
                }
            )

        return results
