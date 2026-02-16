# Evaluation & Reporting

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

## Performance Benchmark
| Metric | Value |
|---|---|
| Feature Extraction (29 queries) | 7.64s |
| FAISS Search (29 queries) | <0.01s |
| Search Throughput | 5,954 queries/sec |
| Device | CPU |

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

1. **Using a stronger feature extractor** — The system uses a pretrained model that was not trained specifically for landmark recognition. Replacing it with a backbone that produces features more robust to lighting, weather, and viewpoint changes would directly fix the main cause of our failures. Fine-tuning the backbone on landmark-specific data would further improve accuracy, since the model would learn which visual details matter most for this task.
2. **Multi-scale feature extraction** — In the scale failure case (query `3b3e774767a97fe5`, where the Eiffel Tower occupies a small portion of the frame), the landmark takes up only a small part of the image. Since our model averages features over the entire image, the landmark signal gets lost. Extracting features at multiple crop levels would help the system focus on the landmark area regardless of its size in the frame.
3. **Gallery augmentation** — Adding gallery images with different conditions (night, snow, various angles) would close the gap between gallery and query images, making the system more robust without changing the model.
