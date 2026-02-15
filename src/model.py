import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights
from tqdm import tqdm
import os
from src.logger import Logger

logger = Logger(__name__)


def load_model(device: torch.device) -> nn.Module:
    """
    Loads ResNet101 with pretrained weights and replaces the classification
    head with Identity to extract raw features (2048-dim).
    """
    logger.info("Loading ResNet101 weights...")

    weights = ResNet101_Weights.DEFAULT
    model = resnet101(weights=weights)

    # Remove the Classification Head
    model.fc = nn.Identity()

    model.to(device)
    model.eval()

    return model


def extract_features(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
):
    """
    Runs inference on the dataloader and returns normalized embeddings.
    """
    all_embeddings = []
    all_labels = []
    all_image_ids = []
    all_relpaths = []

    logger.info(f"Starting inference on {len(dataloader.dataset)} images...")

    # No gradients needed for inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            if batch is None:
                continue

            images = batch["image"].to(device)

            # Forward Pass
            features = model(images)  # Shape: [Batch_Size, 2048]

            # L2 Normalization
            features = F.normalize(features, p=2, dim=1)

            all_embeddings.append(features.cpu())
            all_labels.extend(batch["class_name"])
            all_image_ids.extend(batch["image_id"])
            all_relpaths.extend(batch["relpath"])

    # Concatenate all batches into a single Tensor [Total_Images, 2048]
    embeddings_tensor = torch.cat(all_embeddings, dim=0)

    return {
        "embeddings": embeddings_tensor,
        "classes": all_labels,
        "ids": all_image_ids,
        "paths": all_relpaths,
    }


def get_embeddings_with_cache(cache_path: str, model_loader_func, dataloader, device):
    """
    Checks if embeddings exist on disk. If yes, loads them.
    If no, runs inference and saves them.
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading cached embeddings from {cache_path}")
        return torch.load(cache_path)

    logger.info("Cache not found. Computing embeddings...")
    model = model_loader_func(device)

    data = extract_features(model, dataloader, device)

    logger.info(f"Saving embeddings to {cache_path}")
    torch.save(data, cache_path)

    return data
