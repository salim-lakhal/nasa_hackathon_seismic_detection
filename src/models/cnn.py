"""
CNN architectures for seismic detection.
Includes both custom CNN and ResNet18 transfer learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeismicCNN(nn.Module):
    """
    Custom 3-block CNN architecture for seismic spectrogram classification.
    Designed for grayscale 224x224 spectrograms.
    """

    def __init__(self, num_classes=1, dropout_rate=0.5):
        """
        Initialize custom CNN.

        Args:
            num_classes: Number of output classes (1 for binary with BCEWithLogitsLoss)
            dropout_rate: Dropout probability for regularization
        """
        super(SeismicCNN, self).__init__()

        # Block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 224 -> 112

        # Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112 -> 56

        # Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56 -> 28

        # Global Average Pooling instead of flatten for better generalization
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        """Forward pass."""
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Global pooling and flatten
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        # Fully connected
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResNet18Seismic(nn.Module):
    """
    ResNet18 with transfer learning for seismic detection.
    Pretrained on ImageNet, adapted for grayscale input and binary classification.
    """

    def __init__(self, num_classes=1, pretrained=True, freeze_backbone=False):
        """
        Initialize ResNet18 with transfer learning.

        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze all layers except final classifier
        """
        super(ResNet18Seismic, self).__init__()

        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Modify first conv layer to accept grayscale input (1 channel)
        # Average the pretrained RGB weights to create single channel weights
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            # Initialize new conv1 with averaged RGB weights
            with torch.no_grad():
                self.resnet.conv1.weight = nn.Parameter(
                    original_conv1.weight.mean(dim=1, keepdim=True)
                )

        # Freeze backbone if requested (for fine-tuning)
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

        # If backbone is frozen, ensure final layer is trainable
        if freeze_backbone:
            for param in self.resnet.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        """Forward pass."""
        return self.resnet(x)

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.resnet.parameters():
            param.requires_grad = True
        logger.info("Unfroze all ResNet layers for fine-tuning")


class EfficientSeismicCNN(nn.Module):
    """
    More efficient CNN with depthwise separable convolutions.
    Better for small datasets and edge deployment.
    """

    def __init__(self, num_classes=1, dropout_rate=0.5):
        """Initialize efficient CNN."""
        super(EfficientSeismicCNN, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Depthwise separable convolutions
        self.depthwise1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.pointwise1 = nn.Conv2d(32, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.depthwise2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.pointwise2 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """Forward pass."""
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))

        # Depthwise separable block 1
        x = self.depthwise1(x)
        x = F.relu(self.bn2(self.pointwise1(x)))
        x = self.pool1(x)

        # Depthwise separable block 2
        x = self.depthwise2(x)
        x = F.relu(self.bn3(self.pointwise2(x)))
        x = self.pool2(x)

        # Global pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        # Classifier
        x = self.dropout(x)
        x = self.fc(x)

        return x


def create_model(
    model_type='resnet18',
    num_classes=1,
    pretrained=True,
    **kwargs
):
    """
    Factory function to create models.

    Args:
        model_type: 'resnet18', 'custom_cnn', or 'efficient_cnn'
        num_classes: Number of output classes
        pretrained: Use pretrained weights (ResNet only)
        **kwargs: Additional model-specific arguments

    Returns:
        nn.Module: Initialized model
    """
    if model_type == 'resnet18':
        model = ResNet18Seismic(
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
        logger.info("Created ResNet18 model (transfer learning from ImageNet)")

    elif model_type == 'custom_cnn':
        model = SeismicCNN(
            num_classes=num_classes,
            **kwargs
        )
        logger.info("Created custom 3-block CNN")

    elif model_type == 'efficient_cnn':
        model = EfficientSeismicCNN(
            num_classes=num_classes,
            **kwargs
        )
        logger.info("Created efficient CNN with depthwise separable convolutions")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model
