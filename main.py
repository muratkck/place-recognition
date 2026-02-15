import argparse
import torch
import os
from torch.utils.data import DataLoader
from src.dataset import PlaceDataset, collate_fn
from src.model import load_model, get_embeddings_with_cache
from src.index import VectorIndex
from src.evaluate import compute_metrics
from src.dedup import check_leakage
from src.search import search_image
from src.logger import Logger

logger = Logger(__name__)

# Configuration
DATA_DIR = "data/"
MANIFEST_PATH = "data/manifest.csv"
CACHE_DIR = "cache/"
BATCH_SIZE = 32


def main():
    parser = argparse.ArgumentParser(description="Place Recognition CLI")
    parser.add_argument(
        "mode",
        choices=["index", "evaluate", "search"],
        help="Mode: 'index' | 'evaluate' | 'search'",
    )
    parser.add_argument(
        "--input", type=str, help="Path to a query image or folder (for 'search' mode)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top matches to return (default: 5)"
    )
    args = parser.parse_args()

    check_leakage(MANIFEST_PATH, DATA_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    logger.info("Initializing Gallery Dataset...")
    gallery_dataset = PlaceDataset(MANIFEST_PATH, DATA_DIR, split="gallery")
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    gallery_cache_path = os.path.join(CACHE_DIR, "gallery_embeddings.pt")

    gallery_data = get_embeddings_with_cache(gallery_cache_path, load_model, gallery_loader, device)

    index = VectorIndex(gallery_data)

    if args.mode == "index":
        logger.info("Index built and cached successfully.")
        logger.info(f"Gallery contains {len(gallery_data['classes'])} images.")
        return

    model = load_model(device)

    if args.mode == "evaluate":
        logger.info("Initializing Query Dataset...")
        query_dataset = PlaceDataset(MANIFEST_PATH, DATA_DIR, split="query")
        query_loader = DataLoader(
            query_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn
        )
        compute_metrics(index, query_loader, model, device)

    if args.mode == "search":
        if not args.input:
            logger.error("'search' mode requires --input argument.")
            return
        search_image(args.input, model, index, device, top_k=args.top_k)


if __name__ == "__main__":
    main()
