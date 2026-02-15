import cv2
import torch
import csv
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from src.logger import Logger

logger = Logger(__name__)


class PlaceDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str, split: str, transform=None):
        """
        Args:
            csv_path (str): Path to the manifest.csv file.
            root_dir (str): Directory where images are stored (e.g. 'data/').
            split (str): 'gallery' or 'query'.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split

        self.data = self._load_manifest(csv_path, split)

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def _load_manifest(self, csv_path: str, split: str) -> list[dict]:
        """Parses CSV and filters by split, checking file existence."""
        records = []
        path_obj = Path(csv_path)

        if not path_obj.exists():
            logger.error(f"Manifest file not found: {csv_path}")
            raise FileNotFoundError(f"Manifest file not found: {csv_path}")

        with open(path_obj, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue

                relpath = row.get("relpath", "").strip()
                if not relpath:
                    continue

                abspath = self.root_dir / relpath

                if not abspath.is_file():
                    logger.warning(f"File not found, skipping: {abspath}")
                    continue

                records.append(
                    {
                        "image_id": row["image_id"].strip(),
                        "class_name": row["class_name"].strip(),
                        "relpath": relpath,
                        "abspath": str(abspath),
                    }
                )

        logger.info(f"[{split.upper()}] Loaded {len(records)} images from {csv_path}")
        return records

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        img_path = row["abspath"]

        image = cv2.imread(img_path)

        # Handle Corruption
        if image is None:
            logger.warning(f"Corrupted image, skipping: {img_path}")
            return None

        # Handle very small resolutions
        h, w = image.shape[:2]
        if h < 32 or w < 32:
            logger.warning(f"Image too small ({w}x{h}), skipping: {img_path}")
            return None

        # Handle grayscale images (2D array → 3-channel)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "image_id": row["image_id"],
            "class_name": row["class_name"],
            "relpath": row["relpath"],
        }


def collate_fn(batch):
    """Filter out None values from failed loads."""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
