import torch
import numpy as np
import faiss
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
        self.gallery_classes = gallery_data["classes"]
        self.gallery_ids = gallery_data["ids"]
        self.gallery_paths = gallery_data["paths"]

        vectors = gallery_data["embeddings"]
        if torch.is_tensor(vectors):
            vectors = vectors.cpu().numpy()

        # Build FAISS index (Inner Product = Cosine Similarity for normalized vectors)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors.astype(np.float32))

        logger.info(f"FAISS index built with {len(self.gallery_classes)} gallery images.")

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
        if torch.is_tensor(query_vector):
            query = query_vector.cpu().numpy().flatten()
        else:
            query = query_vector.flatten()

        query = query.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = float(scores[0][i])

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
