# 5. Evaluation & Reporting

## Metrics Implementation

I implemented two standard retrieval metrics in `evaluate.py`:

**Recall@K** — For each query, I check whether the ground-truth class appears anywhere in the top-K retrieved results. If it does, the query counts as a success. I compute this for K = 1, 5, and 10.

**Mean Average Precision (mAP)** — I compute Average Precision for each query using the formula:

```
AP = (1 / R) × Σ (Precision@k × rel_k)
```

where R is the total number of relevant images for that class in the gallery. This is important because some queries have **multiple valid matches** in the gallery (e.g., Grand Canyon has 20 gallery images, Eiffel Tower has 17). By dividing by R instead of the number of retrieved hits, the AP score correctly penalizes the system when it fails to retrieve all relevant images.

I pre-compute R for each class using `Counter(index.gallery_classes)` so that the mAP calculation reflects the true number of gallery images per class.

---

## Handling Multi-Positive Queries

The `calculate_ap` function takes a `total_relevant_in_gallery` parameter that represents the total number of images of the query's class in the gallery. This ensures:

- If a class has 20 images in the gallery but only 5 appear in the top results, the AP is divided by 20 (not 5), penalizing missing results.
- If all relevant images are retrieved at the top ranks, AP approaches 1.0.

---

## Unit Tests

I wrote three unit tests in `test_metrics.py` to validate the metric calculations:

| Test | Scenario | Expected AP |
|---|---|---|
| `test_ap_perfect_single` | Single relevant item at rank 1 | 1.0 |
| **`test_ap_multi_positive`** | **Two relevant items at ranks 1 and 3 (R=2)** | **(1/1 + 2/3) / 2 = 0.833** |
| `test_ap_no_relevant` | No relevant items in gallery (R=0) | 0.0 |

The **multi-positive test** (`test_ap_multi_positive`) specifically validates that the AP calculation handles queries with multiple valid matches correctly, as required by the project specification.

To run the tests:
```bash
uv run pytest tests/test_metrics.py -v
```

---

## Evaluation Results

I ran the evaluation pipeline on the full dataset (37 gallery images, 29 query images) using a pretrained ResNet101 model for feature extraction and cosine similarity for retrieval, with a similarity threshold of 0.6 for open-set detection.

```
Known Queries (Closed Set): 20
  Recall@1 : 80.00%
  Recall@5 : 80.00%
  Recall@10: 80.00%
  mAP:       20.34%

Unknown Queries (Open Set): 9
  Rejection Accuracy: 100.00%
```

The system correctly identified all 9 open-set queries (landmarks not in the gallery) as UNKNOWN, achieving 100% rejection accuracy.

---

## Failure Mode Analysis

Out of 20 known queries, 4 were misclassified at Rank-1. I manually inspected each failed query image and categorized the failure modes:

| Query | Expected | Predicted | Score | Failure Mode |
|---|---|---|---|---|
| `bb5b9cacefa9b2f1` | Grand Canyon | UNKNOWN | 0.51 | Seasonal change (snow) |
| `3b3e774767a97fe5` | Eiffel Tower | UNKNOWN | 0.52 | Scale (landmark too small in frame) |
| `48e2a1df0a4463de` | Eiffel Tower | UNKNOWN | 0.44 | Viewpoint (photo taken from the tower) |
| `65522a18893d373d` | Eiffel Tower | UNKNOWN | 0.53 | Lighting (sunset/near-night) |

All four failures were classified as UNKNOWN rather than misidentified as a different landmark. The system's threshold prevented false positives but was too conservative for these challenging queries.

**Key observations:**

- **Lighting and seasonal changes** reduce feature similarity below the threshold, as the pretrained ResNet101 features are sensitive to color and texture differences.
- **Extreme viewpoint changes**, such as `48e2a1df0a4463de` where the photo was taken from inside the Eiffel Tower rather than of it, produce fundamentally different visual content. This is arguably a correct rejection since the landmark is not visible in the image.
- **Scale differences**, where the landmark occupies a small portion of the image, weaken the global feature representation since ResNet101 averages features across the entire image.
- The similarity threshold (0.6) controls the tradeoff between rejecting unknown places and accepting difficult known queries. Lowering it would recover some of these failures but may risk misclassifying true unknowns.

---

## Suggested Improvements

The failures shows that all 4 errors were caused by the pre-trained ResNet101 model being sensitive to visual changes like lighting, seasons, size, and viewpoint. The most effective improvements would be:

1. **Using a stronger feature extractor** — Our system uses a pretrained model that was not trained specifically for landmark recognition. Replacing it with a backbone that produces features more robust to lighting, weather, and viewpoint changes would directly fix the main cause of our failures. Fine-tuning the backbone on landmark-specific data would further improve accuracy, since the model would learn which visual details matter most for this task.
2. **Multi-scale feature extraction** — In the scale failure case (query `3b3e774767a97fe5`, where the Eiffel Tower occupies a small portion of the frame), the landmark takes up only a small part of the image. Since our model averages features over the entire image, the landmark signal gets lost. Extracting features at multiple crop levels would help the system focus on the landmark area regardless of its size in the frame.
3. **Gallery augmentation** — Adding gallery images with different conditions (night, snow, various angles) would close the gap between gallery and query images, making the system more robust without changing the model.
