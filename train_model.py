"""
Main training script for seismic detection model.
Production-ready ML pipeline with proper logging, checkpointing, and reproducibility.
"""

import torch
from torchvision import transforms
import numpy as np
import random
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data import create_dataloaders
from src.models import create_model
from src.training import SeismicTrainer
from src.training import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def get_transforms(img_size=224):
    """
    Create data augmentation transforms.

    Returns:
        tuple: (train_transform, val_transform)
    """
    # Training transforms with aggressive augmentation for small dataset
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
    ])

    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, val_transform


def main(args):
    """Main training function."""
    logger.info("="*80)
    logger.info("NASA Seismic Detection - Production ML Pipeline")
    logger.info("="*80)

    # Set seed for reproducibility
    set_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")

    # Create output directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    config['device'] = str(device)
    with open(Path(args.log_dir) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Get transforms
    train_transform, val_transform = get_transforms(args.img_size)

    # Create dataloaders
    logger.info(f"Loading data from {args.data_dir}")
    dataloaders, full_dataset = create_dataloaders(
        spectrogram_dir=args.data_dir,
        catalog_path=args.catalog_path if args.catalog_path else None,
        batch_size=args.batch_size,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        num_workers=args.num_workers,
        seed=args.seed,
        transform_train=train_transform,
        transform_val=val_transform
    )

    logger.info(f"Total samples: {len(full_dataset)}")
    logger.info(f"Batch size: {args.batch_size}")

    # Create model
    logger.info(f"Creating {args.model_type} model")
    model = create_model(
        model_type=args.model_type,
        num_classes=1,
        pretrained=args.pretrained,
        dropout_rate=args.dropout
    )

    # Create trainer
    trainer = SeismicTrainer(
        model=model,
        dataloaders=dataloaders,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_class_weights=args.use_class_weights,
        mixed_precision=args.mixed_precision,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    # Train model
    logger.info(f"Starting training for {args.epochs} epochs")
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience
    )

    logger.info("Training complete!")

    # Evaluate on test set
    if args.evaluate:
        logger.info("\nEvaluating on test set...")

        # Load best model
        best_model_path = Path(args.checkpoint_dir) / 'best_model_loss.pth'
        if best_model_path.exists():
            trainer.load_checkpoint(best_model_path)
            logger.info(f"Loaded best model from {best_model_path}")

        # Evaluate
        evaluator = ModelEvaluator(
            model=trainer.model,
            dataloader=dataloaders['test'],
            device=device,
            output_dir=args.log_dir
        )
        results = evaluator.evaluate()

        # Save final metrics
        final_metrics = {
            'training_history': history,
            'test_results': results
        }
        with open(Path(args.log_dir) / 'final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)

    logger.info("\nAll outputs saved to:")
    logger.info(f"  Checkpoints: {args.checkpoint_dir}")
    logger.info(f"  Logs/Visualizations: {args.log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train seismic detection model")

    # Data parameters
    parser.add_argument('--data_dir', type=str,
                       default='entrainement.ipynb',
                       help='Directory containing spectrogram images')
    parser.add_argument('--catalog_path', type=str,
                       default=None,
                       help='Path to catalog CSV (optional)')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size (default: 224)')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='resnet18',
                       choices=['resnet18', 'custom_cnn', 'efficient_cnn'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights (ResNet only)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (small dataset)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                       help='Use class weights for imbalanced data')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='Early stopping patience')

    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Output parameters
    parser.add_argument('--checkpoint_dir', type=str, default='models',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='assets',
                       help='Directory to save logs and visualizations')
    parser.add_argument('--evaluate', action='store_true', default=True,
                       help='Evaluate on test set after training')

    args = parser.parse_args()

    main(args)
