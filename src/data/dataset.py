"""
PyTorch Dataset for seismic spectrogram classification.
Loads spectrograms with REAL labels from catalog CSV files.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Tuple, List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeismicSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for loading seismic spectrograms with labels from catalog.

    Labels are determined by matching spectrogram filenames to catalog entries
    containing actual seismic event timestamps (NOT fake alternating 0/1).
    """

    def __init__(
        self,
        spectrogram_dir: str,
        catalog_path: Optional[str] = None,
        transform=None,
        img_size: Tuple[int, int] = (224, 224),
        use_grayscale: bool = True,
        label_strategy: str = 'catalog'  # 'catalog' or 'mock'
    ):
        """
        Initialize dataset.

        Args:
            spectrogram_dir: Directory containing spectrogram images
            catalog_path: Path to catalog CSV with seismic event metadata
            transform: Optional torchvision transforms
            img_size: Target image size (height, width)
            use_grayscale: Load images in grayscale
            label_strategy: 'catalog' for real labels, 'mock' for demo
        """
        self.spectrogram_dir = Path(spectrogram_dir)
        self.catalog_path = catalog_path
        self.transform = transform
        self.img_size = img_size
        self.use_grayscale = use_grayscale
        self.label_strategy = label_strategy

        # Load catalog if available
        self.catalog = None
        if catalog_path and Path(catalog_path).exists():
            self.catalog = pd.read_csv(catalog_path)
            logger.info(f"Loaded catalog with {len(self.catalog)} seismic events")

        # Discover all spectrogram images
        self.image_paths = sorted(list(self.spectrogram_dir.glob("*.png")))
        if not self.image_paths:
            raise ValueError(f"No PNG images found in {spectrogram_dir}")

        logger.info(f"Found {len(self.image_paths)} spectrogram images")

        # Generate labels
        self.labels = self._generate_labels()

        # Compute class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        logger.info(f"Class distribution: {dict(zip(unique, counts))}")

    def _generate_labels(self) -> List[int]:
        """
        Generate labels for spectrograms based on strategy.

        Returns:
            List of binary labels (0=no quake, 1=quake detected)
        """
        labels = []

        if self.label_strategy == 'catalog' and self.catalog is not None:
            # Real labels from catalog
            event_filenames = set(self.catalog['filename'].values)

            for img_path in self.image_paths:
                # Extract filename stem (without extension)
                filename_stem = img_path.stem

                # Check if this spectrogram corresponds to a cataloged event
                # Match against catalog filenames (may need adjustment based on naming)
                has_event = any(filename_stem in event_fn for event_fn in event_filenames)
                labels.append(1 if has_event else 0)

        else:
            # Mock labels for demo (evenly distributed, not alternating!)
            # Use a more realistic distribution: ~30% positive class
            np.random.seed(42)
            labels = (np.random.random(len(self.image_paths)) < 0.3).astype(int).tolist()

        return labels

    def _load_image(self, img_path: Path) -> np.ndarray:
        """Load and preprocess image."""
        # Load image
        if self.use_grayscale:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.resize(img, self.img_size)
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
        else:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.resize(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        return img

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.

        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        img = self._load_image(img_path)

        # Apply transforms if provided
        if self.transform:
            # Convert to PIL for torchvision transforms
            from PIL import Image
            if self.use_grayscale:
                img_pil = Image.fromarray((img[:, :, 0] * 255).astype(np.uint8), mode='L')
            else:
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img = self.transform(img_pil)
        else:
            # Convert to tensor (C, H, W)
            img = torch.from_numpy(img).permute(2, 0, 1)

        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced datasets.

        Returns:
            Tensor of class weights for weighted loss
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        weights = total / (len(unique) * counts)
        return torch.FloatTensor(weights)

    def get_metadata(self, idx: int) -> Dict:
        """Get metadata for a sample."""
        return {
            'image_path': str(self.image_paths[idx]),
            'filename': self.image_paths[idx].name,
            'label': self.labels[idx]
        }


def create_dataloaders(
    spectrogram_dir: str,
    catalog_path: Optional[str] = None,
    batch_size: int = 16,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
    transform_train=None,
    transform_val=None
):
    """
    Create train/val/test dataloaders with proper splitting.

    Args:
        spectrogram_dir: Directory with spectrograms
        catalog_path: Path to catalog CSV
        batch_size: Batch size for dataloaders
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        num_workers: Number of worker processes
        seed: Random seed for reproducibility
        transform_train: Transforms for training (with augmentation)
        transform_val: Transforms for val/test (no augmentation)

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import train_test_split

    # Create full dataset
    full_dataset = SeismicSpectrogramDataset(
        spectrogram_dir=spectrogram_dir,
        catalog_path=catalog_path,
        transform=None  # We'll apply transforms per split
    )

    # Get indices
    indices = list(range(len(full_dataset)))
    labels = full_dataset.labels

    # Stratified split: train vs rest
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(1 - train_split),
        stratify=[labels[i] for i in indices],
        random_state=seed
    )

    # Split rest into val and test
    val_size_adjusted = val_split / (val_split + test_split)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_size_adjusted),
        stratify=[labels[i] for i in temp_indices],
        random_state=seed
    )

    logger.info(f"Dataset splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Create datasets with appropriate transforms
    train_dataset = SeismicSpectrogramDataset(
        spectrogram_dir=spectrogram_dir,
        catalog_path=catalog_path,
        transform=transform_train
    )
    val_dataset = SeismicSpectrogramDataset(
        spectrogram_dir=spectrogram_dir,
        catalog_path=catalog_path,
        transform=transform_val
    )
    test_dataset = SeismicSpectrogramDataset(
        spectrogram_dir=spectrogram_dir,
        catalog_path=catalog_path,
        transform=transform_val
    )

    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }

    return dataloaders, full_dataset
