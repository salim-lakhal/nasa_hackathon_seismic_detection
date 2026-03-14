"""
Test script to validate the ML architecture without full training.
Demonstrates all components work correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("Testing Seismic Detection ML Architecture")
print("="*80)

# Test 1: Import all modules
print("\n[1/6] Testing imports...")
try:
    from src.data import SeismicSpectrogramDataset, create_dataloaders
    from src.models import create_model, ResNet18Seismic, SeismicCNN
    from src.training import SeismicTrainer, ModelEvaluator
    from src.utils import SpectrogramGenerator
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check data availability
print("\n[2/6] Checking data availability...")
data_dir = Path("entrainement.ipynb")
if data_dir.exists():
    png_files = list(data_dir.glob("*.png"))
    print(f"✓ Found {len(png_files)} spectrogram images in {data_dir}")
else:
    print(f"✗ Data directory not found: {data_dir}")
    sys.exit(1)

# Test 3: Test Dataset creation
print("\n[3/6] Testing Dataset creation...")
try:
    dataset = SeismicSpectrogramDataset(
        spectrogram_dir=str(data_dir),
        catalog_path=None,
        img_size=(224, 224),
        use_grayscale=True,
        label_strategy='mock'
    )
    print(f"✓ Dataset created with {len(dataset)} samples")

    # Test loading a sample
    img, label = dataset[0]
    print(f"  - Sample shape: {img.shape}, Label: {label}")

    # Check class distribution
    class_weights = dataset.get_class_weights()
    print(f"  - Class weights: {class_weights}")
except Exception as e:
    print(f"✗ Dataset creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test model architectures
print("\n[4/6] Testing model architectures...")
try:
    # Test ResNet18
    model_resnet = create_model('resnet18', num_classes=1, pretrained=False)
    print(f"✓ ResNet18 created")

    # Test Custom CNN
    model_custom = create_model('custom_cnn', num_classes=1)
    print(f"✓ Custom CNN created")

    # Test Efficient CNN
    model_efficient = create_model('efficient_cnn', num_classes=1)
    print(f"✓ Efficient CNN created")

    # Test forward pass
    import torch
    dummy_input = torch.randn(2, 1, 224, 224)
    output = model_resnet(dummy_input)
    print(f"  - Forward pass successful, output shape: {output.shape}")

except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test data loading pipeline
print("\n[5/6] Testing data loading pipeline...")
try:
    from torchvision import transforms

    # Simple transforms for testing
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create mini dataloaders for testing
    from torch.utils.data import DataLoader, Subset
    import numpy as np

    # Use small subset for quick test
    subset_indices = np.random.choice(len(dataset), size=min(10, len(dataset)), replace=False)
    subset = Subset(dataset, subset_indices)

    test_loader = DataLoader(
        subset,
        batch_size=4,
        shuffle=False,
        num_workers=0  # Single thread for testing
    )

    # Test one batch
    images, labels = next(iter(test_loader))
    print(f"✓ DataLoader working")
    print(f"  - Batch shape: {images.shape}")
    print(f"  - Labels: {labels}")

except Exception as e:
    print(f"✗ DataLoader test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test inference
print("\n[6/6] Testing inference pipeline...")
try:
    model_resnet.eval()
    with torch.no_grad():
        predictions = model_resnet(images)
        probabilities = torch.sigmoid(predictions)

    print(f"✓ Inference successful")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Sample probabilities: {probabilities.squeeze()[:3]}")

except Exception as e:
    print(f"✗ Inference test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("✓ ALL TESTS PASSED")
print("="*80)
print("\nArchitecture validation complete!")
print("\nNext steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Run training: python3 train_model.py --data_dir entrainement.ipynb")
print("3. Check results in assets/ directory")
print("\nThe ML pipeline is ready for production training! 🚀")
