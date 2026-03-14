"""
Training script for seismic detection model.
Includes proper metrics, checkpointing, early stopping, and visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and compute training/validation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []

    def update(self, predictions, targets, loss):
        """Update metrics with batch results."""
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())
        self.losses.append(loss)

    def compute(self):
        """Compute final metrics."""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        # Convert logits/probabilities to binary predictions
        binary_preds = (predictions >= 0.5).astype(int)

        metrics = {
            'loss': np.mean(self.losses),
            'accuracy': accuracy_score(targets, binary_preds),
            'precision': precision_score(targets, binary_preds, zero_division=0),
            'recall': recall_score(targets, binary_preds, zero_division=0),
            'f1': f1_score(targets, binary_preds, zero_division=0),
        }

        # Add AUC-ROC if we have both classes
        if len(np.unique(targets)) > 1:
            metrics['auc_roc'] = roc_auc_score(targets, predictions)
        else:
            metrics['auc_roc'] = 0.0

        return metrics


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """Check if should stop training."""
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")

        return self.early_stop


class SeismicTrainer:
    """Trainer class for seismic detection models."""

    def __init__(
        self,
        model,
        dataloaders,
        device='cuda',
        learning_rate=1e-3,
        weight_decay=1e-4,
        use_class_weights=True,
        mixed_precision=True,
        checkpoint_dir='models',
        log_dir='assets'
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            dataloaders: Dict with 'train', 'val', 'test' dataloaders
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            use_class_weights: Use class weights for imbalanced data
            mixed_precision: Use automatic mixed precision training
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory to save logs and metrics
        """
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.device = device
        self.use_mixed_precision = mixed_precision

        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Loss function with optional class weights
        if use_class_weights:
            # Get class weights from dataset
            try:
                class_weights = dataloaders['train'].dataset.dataset.get_class_weights()
                pos_weight = class_weights[1] / class_weights[0]
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
                logger.info(f"Using class weights - positive weight: {pos_weight:.4f}")
            except Exception as e:
                logger.warning(f"Could not compute class weights: {e}. Using unweighted loss.")
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None

        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': []
        }

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        metrics_tracker = MetricsTracker()

        pbar = tqdm(self.dataloaders['train'], desc=f'Epoch {epoch} [Train]')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.float().to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_mixed_precision):
                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Update metrics
            probs = torch.sigmoid(outputs)
            metrics_tracker.update(probs, labels, loss.item())

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Compute epoch metrics
        metrics = metrics_tracker.compute()
        return metrics

    def validate(self, epoch, phase='val'):
        """Validate model."""
        self.model.eval()
        metrics_tracker = MetricsTracker()

        with torch.no_grad():
            pbar = tqdm(self.dataloaders[phase], desc=f'Epoch {epoch} [{phase.capitalize()}]')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.float().to(self.device)

                # Forward pass
                with autocast(enabled=self.use_mixed_precision):
                    outputs = self.model(images).squeeze()
                    loss = self.criterion(outputs, labels)

                # Update metrics
                probs = torch.sigmoid(outputs)
                metrics_tracker.update(probs, labels, loss.item())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Compute metrics
        metrics = metrics_tracker.compute()
        return metrics

    def save_checkpoint(self, epoch, metrics, filename='best_model.pth'):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['metrics']

    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(self.history['train_acc'], label='Train Accuracy', marker='o')
        axes[0, 1].plot(self.history['val_acc'], label='Val Accuracy', marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # F1 curves
        axes[1, 0].plot(self.history['train_f1'], label='Train F1', marker='o')
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', marker='s')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning rate
        axes[1, 1].plot(self.history['learning_rates'], marker='o')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)

        plt.tight_layout()
        save_path = self.log_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training curves to {save_path}")

    def train(self, num_epochs=50, early_stopping_patience=15):
        """
        Full training loop.

        Args:
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping

        Returns:
            dict: Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")

        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch, phase='val')

            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rates'].append(current_lr)

            # Log metrics
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                       f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc_roc']:.4f}")
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc_roc']:.4f}")
            logger.info(f"LR: {current_lr:.6f}")

            # Save best model based on validation loss
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics, 'best_model_loss.pth')

            # Save best model based on validation accuracy
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, val_metrics, 'best_model_acc.pth')

            # Save latest checkpoint
            self.save_checkpoint(epoch, val_metrics, 'latest_model.pth')

            # Check early stopping
            if early_stopping(val_metrics['loss']):
                logger.info("Early stopping triggered!")
                break

        # Plot training curves
        self.plot_training_curves()

        # Save training history
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

        return self.history
