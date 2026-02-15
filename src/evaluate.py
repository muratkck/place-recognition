from collections import Counter
from .model import extract_features
from src.logger import Logger

logger = Logger(__name__)


def calculate_ap(retrieved_classes: list, gt_class: str, total_relevant_in_gallery: int) -> float:
    """
    Computes Average Precision (AP) for a single query.

    AP = (1 / R) * Sum(Precision@k * rel_k)
    where R is the total number of relevant documents in the gallery.

    Args:
        retrieved_classes: List of class names sorted by similarity (rank 1 to N).
        gt_class: The true class of the query.
        total_relevant_in_gallery: Total number of images of this class in the gallery (R).
    """
    if total_relevant_in_gallery == 0:
        return 0.0

    relevant_found = 0
    precision_sum = 0.0

    for k, pred_class in enumerate(retrieved_classes):
        # k is 0-indexed -> rank is k+1
        if pred_class == gt_class:
            relevant_found += 1
            precision_at_k = relevant_found / (k + 1)
            precision_sum += precision_at_k

    return precision_sum / total_relevant_in_gallery


def compute_metrics(index, query_loader, model, device, top_k=[1, 5, 10]):
    """
    Evaluates the system using Recall@K and Mean Average Precision (mAP).
    Separates 'Closed-Set' (known places) from 'Open-Set' (unknown places).
    """
    logger.info("Extracting query features...")
    query_data = extract_features(model, query_loader, device)

    query_vectors = query_data["embeddings"]
    query_classes = query_data["classes"]

    gallery_class_counts = Counter(index.gallery_classes)  # the denominator for mAP calculation.

    recalls = {k: 0 for k in top_k}
    aps = []
    failures = []

    known_count = 0
    unknown_count = 0
    correct_unknowns = 0

    logger.info("Running similarity search...")

    for i in range(len(query_classes)):
        q_vec = query_vectors[i]
        gt_class = query_classes[i]

        search_limit = max(max(top_k), 100)

        results = index.search(q_vec, top_k=search_limit, threshold=0.6)
        top_pred_class = results[0]["class"] if results else "UNKNOWN"

        # The Query is not in the Gallery (Open Set)
        if gt_class not in gallery_class_counts:
            unknown_count += 1
            # Success if the top result is "UNKNOWN"
            if top_pred_class == "UNKNOWN":
                correct_unknowns += 1
            continue

        # The Query is in the Gallery (Closed Set)
        known_count += 1

        # Calculate Recall@K
        retrieved_classes = [r["class"] for r in results]

        for k in top_k:
            if gt_class in retrieved_classes[:k]:
                recalls[k] += 1

        # Track failures (wrong at Recall@1)
        if not retrieved_classes or retrieved_classes[0] != gt_class:
            failures.append(
                {
                    "query_id": query_data["ids"][i],
                    "gt_class": gt_class,
                    "pred_class": top_pred_class,
                    "score": results[0]["score"] if results else 0.0,
                }
            )

        # Calculate Average Precision (AP)
        total_relevant = gallery_class_counts[gt_class]
        ap = calculate_ap(retrieved_classes, gt_class, total_relevant)
        aps.append(ap)

    # Evaluation Report
    logger.info("=" * 40)
    logger.info("       EVALUATION REPORT       ")
    logger.info("=" * 40)

    if known_count > 0:
        logger.info(f"Known Queries (Closed Set): {known_count}")

        for k in top_k:
            acc = (recalls[k] / known_count) * 100
            logger.info(f"  Recall@{k:<2}: {acc:.2f}%")

        mean_ap = (sum(aps) / len(aps)) * 100
        logger.info(f"  mAP:        {mean_ap:.2f}%")

    else:
        logger.info("No known queries found in this split.")

    if unknown_count > 0:
        logger.info(f"Unknown Queries (Open Set): {unknown_count}")
        rej_rate = (correct_unknowns / unknown_count) * 100
        logger.info(f"Rejection Accuracy: {rej_rate:.2f}% (Correctly identified as UNKNOWN)")
    else:
        logger.info("No unknown queries (distractors) found.")

    # Failure Details
    if failures:
        logger.info(f"Failure Details ({len(failures)} misclassified at Rank-1):")
        for f in failures:
            logger.info(
                f"Query {f['query_id']}: expected '{f['gt_class']}' "
                f"→ got '{f['pred_class']}' (score: {f['score']})"
            )

    logger.info("=" * 40)
