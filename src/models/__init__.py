"""Model architectures for seismic detection."""

from .cnn import (
    SeismicCNN,
    ResNet18Seismic,
    EfficientSeismicCNN,
    create_model
)

__all__ = [
    'SeismicCNN',
    'ResNet18Seismic',
    'EfficientSeismicCNN',
    'create_model'
]
