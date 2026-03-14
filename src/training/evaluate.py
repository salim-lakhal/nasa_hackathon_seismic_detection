"""
Evaluation script for seismic detection model.
Generates confusion matrix, ROC curve, and sample predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from pathlib import Path
import json
import logging
from tqdm import tqdm
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with visualizations."""

    def __init__(self, model, dataloader, device='cuda', output_dir='assets'):
        """
        Initialize evaluator.

        Args:
            model: Trained PyTorch model
            dataloader: DataLoader for evaluation
            device: Device for inference
            output_dir: Directory to save evaluation results
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.predictions = None
        self.probabilities = None
        self.targets = None
        self.results = {}

    def predict(self):
        """Run inference and collect predictions."""
        self.model.eval()

        all_probs = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in tqdm(self.dataloader, desc="Evaluating"):
                images = images.to(self.device)

                # Forward pass
                outputs = self.model(images).squeeze()
                probs = torch.sigmoid(outputs)

                # Store results
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend((probs >= 0.5).cpu().numpy().astype(int))
                all_targets.extend(labels.numpy())

        self.probabilities = np.array(all_probs)
        self.predictions = np.array(all_preds)
        self.targets = np.array(all_targets)

        logger.info(f"Predictions collected: {len(self.predictions)} samples")

    def compute_metrics(self):
        """Compute comprehensive evaluation metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        metrics = {
            'accuracy': accuracy_score(self.targets, self.predictions),
            'precision': precision_score(self.targets, self.predictions, zero_division=0),
            'recall': recall_score(self.targets, self.predictions, zero_division=0),
            'f1': f1_score(self.targets, self.predictions, zero_division=0),
            'specificity': None,
            'auc_roc': None,
            'average_precision': None
        }

        # Compute specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(self.targets, self.predictions).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)

        # AUC-ROC and Average Precision
        if len(np.unique(self.targets)) > 1:
            metrics['auc_roc'] = roc_auc_score(self.targets, self.probabilities)
            metrics['average_precision'] = average_precision_score(self.targets, self.probabilities)
        else:
            metrics['auc_roc'] = 0.0
            metrics['average_precision'] = 0.0

        self.results['metrics'] = metrics
        return metrics

    def plot_confusion_matrix(self, normalize=False):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(self.targets, self.predictions)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=['No Quake', 'Quake'],
                   yticklabels=['No Quake', 'Quake'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)

        filename = 'confusion_matrix_norm.png' if normalize else 'confusion_matrix.png'
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix to {save_path}")

    def plot_roc_curve(self):
        """Plot and save ROC curve."""
        if len(np.unique(self.targets)) <= 1:
            logger.warning("Cannot plot ROC curve with only one class")
            return

        fpr, tpr, thresholds = roc_curve(self.targets, self.probabilities)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        save_path = self.output_dir / 'roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved ROC curve to {save_path}")

        # Store ROC data
        self.results['roc'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc)
        }

    def plot_precision_recall_curve(self):
        """Plot and save Precision-Recall curve."""
        if len(np.unique(self.targets)) <= 1:
            logger.warning("Cannot plot PR curve with only one class")
            return

        precision, recall, thresholds = precision_recall_curve(self.targets, self.probabilities)
        avg_precision = average_precision_score(self.targets, self.probabilities)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        save_path = self.output_dir / 'precision_recall_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Precision-Recall curve to {save_path}")

    def plot_sample_predictions(self, num_samples=16):
        """Plot sample predictions (correct and incorrect)."""
        # Get indices of correct and incorrect predictions
        correct_idx = np.where(self.predictions == self.targets)[0]
        incorrect_idx = np.where(self.predictions != self.targets)[0]

        # Sample from each category
        num_correct = min(num_samples // 2, len(correct_idx))
        num_incorrect = min(num_samples // 2, len(incorrect_idx))

        sampled_correct = np.random.choice(correct_idx, size=num_correct, replace=False) if len(correct_idx) > 0 else []
        sampled_incorrect = np.random.choice(incorrect_idx, size=num_incorrect, replace=False) if len(incorrect_idx) > 0 else []

        sampled_indices = np.concatenate([sampled_correct, sampled_incorrect])

        # Get actual images from dataset
        dataset = self.dataloader.dataset
        if hasattr(dataset, 'dataset'):  # Handle Subset wrapper
            dataset = dataset.dataset

        # Create figure
        n_samples = len(sampled_indices)
        cols = 4
        rows = (n_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if n_samples > 1 else [axes]

        for idx, sample_idx in enumerate(sampled_indices):
            # Get image from dataset
            img_path = dataset.image_paths[sample_idx]
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))

            # Get predictions
            true_label = self.targets[sample_idx]
            pred_label = self.predictions[sample_idx]
            prob = self.probabilities[sample_idx]

            # Determine color (green for correct, red for incorrect)
            color = 'green' if true_label == pred_label else 'red'

            # Plot
            axes[idx].imshow(img, cmap='gray')
            axes[idx].axis('off')
            title = f'True: {"Quake" if true_label else "No Quake"}\n'
            title += f'Pred: {"Quake" if pred_label else "No Quake"} ({prob:.2f})'
            axes[idx].set_title(title, color=color, fontsize=10)

        # Hide unused subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        save_path = self.output_dir / 'sample_predictions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved sample predictions to {save_path}")

    def generate_classification_report(self):
        """Generate and save detailed classification report."""
        report = classification_report(
            self.targets,
            self.predictions,
            target_names=['No Quake', 'Quake'],
            digits=4
        )

        logger.info("\nClassification Report:")
        logger.info("\n" + report)

        # Save to file
        report_path = self.output_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved classification report to {report_path}")

        return report

    def save_results(self):
        """Save all evaluation results to JSON."""
        results_path = self.output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved evaluation results to {results_path}")

    def evaluate(self):
        """Run complete evaluation pipeline."""
        logger.info("Starting model evaluation...")

        # Collect predictions
        self.predict()

        # Compute metrics
        metrics = self.compute_metrics()
        logger.info("\nEvaluation Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        # Generate visualizations
        self.plot_confusion_matrix(normalize=False)
        self.plot_confusion_matrix(normalize=True)
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_sample_predictions(num_samples=16)

        # Generate classification report
        self.generate_classification_report()

        # Save results
        self.save_results()

        logger.info("\nEvaluation complete!")
        return self.results


def evaluate_model(model_path, dataloader, device='cuda', output_dir='assets'):
    """
    Convenience function to evaluate a saved model.

    Args:
        model_path: Path to saved model checkpoint
        dataloader: DataLoader for evaluation
        device: Device for inference
        output_dir: Directory to save results

    Returns:
        dict: Evaluation results
    """
    from ..models import create_model

    # Load model
    checkpoint = torch.load(model_path, map_location=device)

    # Create model (assuming ResNet18 by default)
    model = create_model('resnet18', num_classes=1, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    evaluator = ModelEvaluator(model, dataloader, device, output_dir)
    results = evaluator.evaluate()

    return results
