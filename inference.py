"""
Inference script for seismic detection.
Load trained model and make predictions on new spectrograms.
"""

import torch
from torchvision import transforms
import cv2
import argparse
from pathlib import Path
import json
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.models import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeismicPredictor:
    """Inference class for seismic detection."""

    def __init__(self, model_path, device='cuda', img_size=224):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model checkpoint
            device: Device for inference
            img_size: Input image size
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size

        # Load model
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Create model (ResNet18 by default)
        self.model = create_model('resnet18', num_classes=1, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        logger.info(f"Model loaded successfully on {self.device}")

    def preprocess_image(self, image_path):
        """Load and preprocess image."""
        # Load image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Normalize to [0, 255] uint8
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Apply transforms
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        return img_tensor

    def predict(self, image_path, return_probability=True):
        """
        Make prediction on single image.

        Args:
            image_path: Path to spectrogram image
            return_probability: Return probability instead of binary label

        Returns:
            float or int: Probability or binary prediction
        """
        # Preprocess
        img_tensor = self.preprocess_image(image_path)
        img_tensor = img_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(img_tensor).squeeze()
            probability = torch.sigmoid(output).item()

        if return_probability:
            return probability
        else:
            return 1 if probability >= 0.5 else 0

    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images.

        Args:
            image_paths: List of image paths

        Returns:
            list: List of (probability, prediction) tuples
        """
        results = []

        for image_path in image_paths:
            try:
                prob = self.predict(image_path, return_probability=True)
                pred = 1 if prob >= 0.5 else 0
                results.append({
                    'image': str(image_path),
                    'probability': float(prob),
                    'prediction': int(pred),
                    'label': 'Quake' if pred == 1 else 'No Quake'
                })
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'image': str(image_path),
                    'error': str(e)
                })

        return results


def main(args):
    """Main inference function."""
    # Initialize predictor
    predictor = SeismicPredictor(
        model_path=args.model_path,
        device=args.device,
        img_size=args.img_size
    )

    # Get image paths
    image_dir = Path(args.image_dir)
    if image_dir.is_file():
        image_paths = [image_dir]
    else:
        image_paths = sorted(list(image_dir.glob("*.png")))

    if not image_paths:
        logger.error(f"No images found in {args.image_dir}")
        return

    logger.info(f"Processing {len(image_paths)} images...")

    # Make predictions
    results = predictor.predict_batch(image_paths)

    # Print results
    logger.info("\nPredictions:")
    for result in results:
        if 'error' not in result:
            logger.info(f"{Path(result['image']).name}: "
                       f"{result['label']} (prob={result['probability']:.4f})")
        else:
            logger.error(f"{Path(result['image']).name}: ERROR - {result['error']}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")

    # Summary statistics
    successful = [r for r in results if 'error' not in r]
    if successful:
        quakes = sum(1 for r in successful if r['prediction'] == 1)
        no_quakes = len(successful) - quakes
        logger.info(f"\nSummary: {quakes} quakes detected, {no_quakes} no quakes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on seismic spectrograms")

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images or path to single image')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size')

    args = parser.parse_args()
    main(args)
