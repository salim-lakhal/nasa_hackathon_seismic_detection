# Seismic Detection ML Pipeline - Training Guide

## Production-Ready ML System for NASA Hackathon

This is a complete, production-quality machine learning pipeline for detecting seismic events (moonquakes/marsquakes) from spectrogram images.

---

## System Architecture

```
nasa_hackathon_seismic_detection/
├── src/                          # Production ML code
│   ├── data/                     # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── dataset.py           # PyTorch Dataset with REAL catalog labels
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   └── cnn.py               # ResNet18, Custom CNN, Efficient CNN
│   ├── training/                 # Training and evaluation
│   │   ├── __init__.py
│   │   ├── train.py             # Training loop with metrics/checkpointing
│   │   └── evaluate.py          # Evaluation with visualizations
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       └── spectrogram.py       # Spectrogram generation from miniseed
├── train_model.py               # Main training script
├── inference.py                 # Inference script for predictions
├── models/                      # Saved model checkpoints
└── assets/                      # Training logs and visualizations

```

---

## Setup Instructions

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
```

---

## Data Preparation

### Option 1: Use Existing Spectrograms

The system can work with pre-generated spectrograms (77 images in `entrainement.ipynb/`).

### Option 2: Generate from Miniseed Files

If you have the actual seismic data:

```python
from src.utils import SpectrogramGenerator

# Initialize generator
generator = SpectrogramGenerator(output_dir='spectrograms', img_size=(224, 224))

# Generate from miniseed files
mseed_files = ['path/to/file1.mseed', 'path/to/file2.mseed']
results = generator.batch_generate(mseed_files, output_dir='spectrograms')
```

### Label Strategy

**REAL Labels (Production):**
- Labels are derived from `apollo12_catalog_GradeA_final.csv`
- Matches spectrogram filenames to catalog entries
- Binary classification: 0=No Quake, 1=Quake Detected

**Mock Labels (Demo):**
- If catalog unavailable, uses realistic distribution (~30% positive class)
- NOT the fake alternating 0/1 pattern from notebooks

---

## Training

### Quick Start (Default Settings)

```bash
python3 train_model.py \
  --data_dir entrainement.ipynb \
  --epochs 50 \
  --batch_size 8 \
  --model_type resnet18 \
  --pretrained
```

### Full Training Command with All Options

```bash
python3 train_model.py \
  --data_dir entrainement.ipynb \
  --catalog_path data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv \
  --model_type resnet18 \
  --pretrained \
  --epochs 50 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --dropout 0.5 \
  --use_class_weights \
  --mixed_precision \
  --early_stopping_patience 15 \
  --seed 42 \
  --checkpoint_dir models \
  --log_dir assets \
  --evaluate
```

### Model Options

1. **ResNet18 (Transfer Learning - RECOMMENDED)**
   - Pre-trained on ImageNet
   - Adapted for grayscale spectrograms
   - Best for small datasets
   - `--model_type resnet18 --pretrained`

2. **Custom 3-Block CNN**
   - Lightweight architecture
   - Good baseline
   - `--model_type custom_cnn`

3. **Efficient CNN**
   - Depthwise separable convolutions
   - Edge deployment ready
   - `--model_type efficient_cnn`

---

## Training Features

### Data Augmentation (Small Dataset Optimization)
- **Rotation:** ±30 degrees
- **Flips:** Horizontal and vertical
- **Translation:** ±10%
- **Scaling:** 0.9x to 1.1x
- **Shear:** ±10 degrees
- **Brightness/Contrast:** ±20%

### Regularization
- L2 weight decay (1e-4)
- Dropout (0.5)
- Batch normalization
- Early stopping

### Training Monitoring
- **Metrics tracked:** Loss, Accuracy, Precision, Recall, F1, AUC-ROC
- **Checkpointing:** Best model (loss), Best model (accuracy), Latest
- **Learning rate scheduling:** ReduceLROnPlateau
- **Mixed precision:** Faster training with AMP

### Class Imbalance Handling
- Weighted loss function
- Stratified train/val/test split (70/15/15)

---

## Outputs

### Model Checkpoints (`models/`)
- `best_model_loss.pth` - Best validation loss
- `best_model_acc.pth` - Best validation accuracy
- `latest_model.pth` - Latest epoch

### Visualizations (`assets/`)
- `training_curves.png` - Loss, accuracy, F1, learning rate
- `confusion_matrix.png` - Raw confusion matrix
- `confusion_matrix_norm.png` - Normalized confusion matrix
- `roc_curve.png` - ROC curve with AUC score
- `precision_recall_curve.png` - PR curve
- `sample_predictions.png` - Correct/incorrect predictions

### Logs (`assets/`)
- `config.json` - Training configuration
- `training_history.json` - Epoch-by-epoch metrics
- `evaluation_results.json` - Test set metrics
- `classification_report.txt` - Detailed classification report
- `final_metrics.json` - Complete results

---

## Inference

### Single Image

```bash
python3 inference.py \
  --model_path models/best_model_loss.pth \
  --image_dir path/to/spectrogram.png \
  --output predictions.json
```

### Batch Prediction

```bash
python3 inference.py \
  --model_path models/best_model_loss.pth \
  --image_dir path/to/spectrograms/ \
  --output batch_predictions.json
```

### Programmatic Usage

```python
from inference import SeismicPredictor

predictor = SeismicPredictor(model_path='models/best_model_loss.pth')
probability = predictor.predict('spectrogram.png')
print(f"Quake probability: {probability:.4f}")
```

---

## Model Architecture

### ResNet18 Transfer Learning (Recommended)

```
Input: 1x224x224 grayscale spectrogram
  ↓
ResNet18 Backbone (pretrained on ImageNet)
  - Conv1: 1→64 (modified for grayscale)
  - Layer1: 64→64
  - Layer2: 64→128
  - Layer3: 128→256
  - Layer4: 256→512
  ↓
Global Average Pooling
  ↓
Dropout (0.5)
  ↓
Fully Connected: 512→1
  ↓
Output: Logit (BCEWithLogitsLoss)
```

**Why ResNet18?**
- Transfer learning from ImageNet helps with small dataset (77 samples)
- Residual connections prevent vanishing gradients
- Proven architecture for image classification
- ~11M parameters

---

## Performance Metrics

### What to Expect (77 Sample Dataset)

With proper training on 77 spectrograms:
- **Validation Accuracy:** 70-85% (depends on data quality)
- **F1 Score:** 0.65-0.80
- **AUC-ROC:** 0.75-0.90

**Note:** Performance will significantly improve with:
- More training samples (>500 recommended)
- Real catalog labels (not mock)
- Actual seismic data from NASA

### Interpreting Results

- **High Recall:** Model catches most quakes (few false negatives)
- **High Precision:** Model doesn't cry wolf (few false positives)
- **High F1:** Balanced performance
- **High AUC-ROC:** Good separation between classes

---

## Production Deployment

### Model Export

```python
# Export to ONNX for cross-platform deployment
import torch
model = torch.load('models/best_model_loss.pth')
dummy_input = torch.randn(1, 1, 224, 224)
torch.onnx.export(model, dummy_input, 'models/seismic_detector.onnx')
```

### Edge Deployment (TensorFlow Lite)

Convert PyTorch → ONNX → TensorFlow → TFLite for mobile/edge devices.

### API Serving (FastAPI)

See `my_model_demo/app.py` for Flask example, upgrade to FastAPI for production.

---

## Reproducibility

All random operations are seeded (default: 42):
- NumPy random
- PyTorch random
- CUDA random
- Train/val/test split

Configuration saved to `assets/config.json` for every run.

---

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` to 4 or 2
- Use `--mixed_precision` (enabled by default)

### Overfitting (Train >> Val)
- Increase `--weight_decay`
- Increase `--dropout`
- More aggressive augmentation
- Reduce model complexity

### Underfitting (Both Low)
- Increase epochs
- Reduce regularization
- Use pretrained model
- Check data quality

### Class Imbalance
- Enable `--use_class_weights` (default)
- Monitor F1 and AUC-ROC, not just accuracy

---

## Advanced Usage

### 5-Fold Cross-Validation

Modify `train_model.py` to use sklearn's `KFold` for more robust evaluation on small datasets.

### Ensemble Methods

Train multiple models and average predictions:
```python
models = [model1, model2, model3]
predictions = [model(x) for model in models]
ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
```

### Fine-tuning Strategy

1. **Phase 1:** Freeze backbone, train head (10 epochs)
2. **Phase 2:** Unfreeze all, fine-tune (40 epochs)

---

## Citation

If you use this pipeline in your research:

```
NASA Space Apps Challenge 2024 - Seismic Detection Across the Solar System
Production ML Pipeline by [Your Team]
```

---

## Contact & Support

For issues or questions:
1. Check this guide
2. Review `assets/training.log`
3. Inspect `assets/config.json`
4. Examine visualization outputs

---

## Next Steps

1. ✅ Verify installation
2. ✅ Run training with default settings
3. ✅ Check `assets/` for visualizations
4. ✅ Evaluate on test set
5. ✅ Deploy model for inference
6. 🚀 Scale to production data

**Good luck with your seismic detection mission!** 🌙🔴
