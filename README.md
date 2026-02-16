# Place Recognition & Retrieval System

Image-based place recognition system using a pretrained ResNet101 backbone for feature extraction and cosine similarity for retrieval.

## Installation

```bash
git clone https://github.com/muratkck/place-recognition.git
cd place-recognition
uv sync
```

This installs all dependencies (PyTorch CPU, OpenCV, NumPy, scikit-learn, tqdm) via `uv`.

## Data Setup

Place your images under `data/landmarks/` and define the gallery/query split in `data/manifest.csv`:

```csv
mapping_line,class_name,class_dir,landmark_id,split,image_id,relpath,url
44797,Grand Canyon,Grand_Canyon__lid_44795,44795,gallery,abc123,landmarks/Grand_Canyon/gallery/abc123.jpg,https://...
44797,Grand Canyon,Grand_Canyon__lid_44795,44795,query,def456,landmarks/Grand_Canyon/query/def456.jpg,https://...
```

Directory layout:

```
data/
├── manifest.csv                          # gallery/query split definitions
└── landmarks/
    ├── Eiffel_Tower__lid_47378/
    │   ├── gallery/                      # reference images
    │   │   └── *.jpg
    │   └── query/                        # test images
    │       └── *.jpg
    ├── Grand_Canyon__lid_44795/
    │   ├── gallery/
    │   └── query/
    └── ...                               # one folder per landmark
```

## Usage

### Build the Gallery Index

Extracts embeddings for all gallery images and caches them to `cache/`:

```bash
uv run python main.py index
```

### Search for a Query Image

Find the top-K most similar gallery images for a given query image:

```bash
uv run python main.py search --input path/to/image.jpg --top-k 5
```

### Run Evaluation

Computes Recall@1/5/10, mAP, open-set rejection accuracy, and per-query failure details:

```bash
uv run python main.py evaluate
```

### Run Tests

```bash
uv run pytest -v
```

## Notes

- **Near-duplicate check** runs automatically before every mode. It uses pHash to detect potential data leakage between gallery and query splits.
- **Gallery embeddings are cached** in `cache/gallery_embeddings.pt`. Delete this file to force re-extraction.
- **Corrupted, grayscale, and too-small images** (below 32×32) are automatically logged and skipped.
- **Open-set detection** uses a similarity threshold of 0.6. Queries with no match above this threshold are classified as UNKNOWN.
