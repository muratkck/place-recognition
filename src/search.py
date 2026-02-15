"""
Search module for Top-K retrieval on a single query image.
"""

import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms
from src.logger import Logger

logger = Logger(__name__)

# Same transforms used in PlaceDataset
TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_image(image_path: str) -> dict | None:
    """
    Loads a single image and returns a dict with 'path' and 'tensor' keys.
    Returns None if the image cannot be read.
    """
    path = Path(image_path)

    if not path.is_file():
        logger.error(f"Image not found: {image_path}")
        return None

    img = cv2.imread(str(path))
    if img is None:
        logger.error(f"Could not read image: {image_path}")
        return None

    # Handle grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = TRANSFORM(img)
    return {"path": str(path), "tensor": tensor}


def search_image(input_path: str, model, index, device, top_k: int = 5):
    """
    Searches the gallery index for Top-K matches for a single query image.
    Logs the results.
    """
    img_data = load_image(input_path)

    if img_data is None:
        return

    # Extract features
    batch = img_data["tensor"].unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(batch)
        features = F.normalize(features, p=2, dim=1)

    query_vec = features.cpu()[0]

    # Search the index
    results = index.search(query_vec, top_k=top_k)

    # Log results
    logger.info(f"Query: {img_data['path']}")
    logger.info(f"  Top-{top_k} matches:")
    for rank, match in enumerate(results, 1):
        logger.info(
            f"    {rank}. [{match['class']}] {match['path']} (score: {match['score']:.4f})"
        )
