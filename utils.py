"""
Utility Functions for Seismic Facies Classification
Based on: "A deep learning framework for seismic facies classification" (Kaur et al., 2022)

Includes:
- Performance metrics (Precision, Recall, F1-score)
- Uncertainty estimation using Bayesian approximation
- Visualization utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings


# ============================================================================
# Loss Functions
# ============================================================================

class CombinedLoss(nn.Module):
    """
    Combined loss for GAN training as described in the paper.
    L = L_mce - λ(L_bce(real) + L_bce(fake))
    
    Args:
        lambda_adv: Weight for adversarial loss (default: 0.1)
                   Comment: Not specified in paper, using common value
    """
    def __init__(self, lambda_adv=0.1, num_classes=6):
        super(CombinedLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.mce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.num_classes = num_classes
    
    def forward(self, pred_seg, true_labels, disc_real, disc_fake):
        """
        Args:
            pred_seg: Predicted segmentation (B, num_classes, H, W)
            true_labels: Ground truth labels (B, H, W)
            disc_real: Discriminator output for real pairs
            disc_fake: Discriminator output for fake pairs
        """
        # Multiclass cross-entropy loss
        mce = self.mce_loss(pred_seg, true_labels)
        
        # Adversarial loss
        real_target = torch.ones_like(disc_real)
        fake_target = torch.zeros_like(disc_fake)
        
        adv_real = self.bce_loss(disc_real, real_target)
        adv_fake = self.bce_loss(disc_fake, fake_target)
        
        # Combined loss
        total_loss = mce - self.lambda_adv * (adv_real + adv_fake)
        
        return total_loss, mce, adv_real, adv_fake


# ============================================================================
# Performance Metrics
# ============================================================================

def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, 
                             num_classes: int) -> torch.Tensor:
    """
    Compute confusion matrix for segmentation.
    
    Args:
        pred: Predicted labels (B, H, W) or (N,)
        target: Ground truth labels (B, H, W) or (N,)
        num_classes: Number of classes
    
    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Filter out invalid labels
    mask = (target >= 0) & (target < num_classes)
    pred = pred[mask]
    target = target[mask]
    
    # Compute confusion matrix
    indices = num_classes * target + pred
    cm = torch.bincount(indices, minlength=num_classes**2)
    cm = cm.reshape(num_classes, num_classes)
    
    return cm


def compute_metrics(confusion_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute precision, recall, and F1 score from confusion matrix.
    As described in equations (4), (5), (6) in the paper.
    
    Args:
        confusion_matrix: Confusion matrix (num_classes, num_classes)
    
    Returns:
        Dictionary containing precision, recall, and F1 score per class
    """
    # True positives are on the diagonal
    tp = torch.diag(confusion_matrix)
    
    # False positives: sum of column - true positive
    fp = confusion_matrix.sum(dim=0) - tp
    
    # False negatives: sum of row - true positive
    fn = confusion_matrix.sum(dim=1) - tp
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp + 1e-10)
    
    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn + 1e-10)
    
    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mean_precision': precision.mean(),
        'mean_recall': recall.mean(),
        'mean_f1': f1.mean()
    }


def evaluate_model(model: nn.Module, dataloader, device: str, 
                   num_classes: int = 6) -> Dict[str, torch.Tensor]:
    """
    Evaluate model on a dataset and compute metrics.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    with torch.no_grad():
        for seismic, labels in dataloader:
            seismic = seismic.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(seismic)
            preds = torch.argmax(outputs, dim=1)
            
            # Update confusion matrix
            cm = compute_confusion_matrix(preds.cpu(), labels.cpu(), num_classes)
            total_cm += cm
    
    # Compute metrics
    metrics = compute_metrics(total_cm)
    metrics['confusion_matrix'] = total_cm
    
    return metrics


# ============================================================================
# Uncertainty Estimation
# ============================================================================

def estimate_uncertainty(model: nn.Module, input_data: torch.Tensor, 
                        num_samples: int = 20, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate epistemic uncertainty using Monte Carlo Dropout.
    As described in the paper using Bayesian approximation with dropout.
    
    Args:
        model: Trained model with dropout layers
        input_data: Input tensor (B, C, H, W)
        num_samples: Number of MC samples (default: 20)
                    Comment: Not specified in paper, using common value
        device: Device to run on
    
    Returns:
        Tuple of (mean_prediction, uncertainty_map)
        - mean_prediction: Mean of predictions (B, H, W)
        - uncertainty_map: Variance of predictions (B, H, W)
    """
    model.eval()  # Keep in eval mode but enable dropout
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Forward pass with dropout enabled
            output = model(input_data.to(device), use_dropout=True)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
            predictions.append(pred.cpu())
    
    # Stack predictions
    predictions = torch.stack(predictions, dim=0)  # (num_samples, B, H, W)
    
    # Compute mean prediction
    # For categorical data, use mode (most frequent class)
    mean_pred = torch.mode(predictions, dim=0)[0]
    
    # Compute uncertainty as variance
    # Convert predictions to one-hot and compute variance
    one_hot = F.one_hot(predictions, num_classes=6).float()  # (num_samples, B, H, W, num_classes)
    one_hot = one_hot.permute(1, 2, 3, 0, 4)  # (B, H, W, num_samples, num_classes)
    
    # Variance across samples
    variance = one_hot.var(dim=3)  # (B, H, W, num_classes)
    uncertainty = variance.max(dim=3)[0]  # (B, H, W) - max variance across classes
    
    return mean_pred, uncertainty


def compute_prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute prediction entropy as another uncertainty measure.
    
    Args:
        logits: Model output logits (B, num_classes, H, W)
    
    Returns:
        Entropy map (B, H, W)
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    
    # Entropy = -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=1)
    
    return entropy


# ============================================================================
# Visualization Utilities
# ============================================================================

def visualize_prediction(seismic: np.ndarray, true_labels: np.ndarray, 
                        pred_labels: np.ndarray, uncertainty: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None, facies_names: Optional[List[str]] = None):
    """
    Visualize seismic data, predictions, and uncertainty.
    
    Args:
        seismic: Seismic image (H, W)
        true_labels: Ground truth labels (H, W)
        pred_labels: Predicted labels (H, W)
        uncertainty: Uncertainty map (H, W), optional
        save_path: Path to save figure, optional
        facies_names: Names of facies classes
    """
    if facies_names is None:
        facies_names = [
            'Basement rocks',
            'Slope mudstone A', 
            'Mass-transport complex',
            'Slope mudstone B',
            'Slope valley',
            'Submarine canyon'
        ]
    
    n_plots = 3 if uncertainty is None else 4
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    # Seismic
    axes[0].imshow(seismic, cmap='seismic', aspect='auto')
    axes[0].set_title('Seismic Section')
    axes[0].axis('off')
    
    # True labels
    im1 = axes[1].imshow(true_labels, cmap='tab10', vmin=0, vmax=5, aspect='auto')
    axes[1].set_title('True Labels')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Predicted labels
    im2 = axes[2].imshow(pred_labels, cmap='tab10', vmin=0, vmax=5, aspect='auto')
    axes[2].set_title('Predicted Labels')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Uncertainty
    if uncertainty is not None:
        im3 = axes[3].imshow(uncertainty, cmap='hot', aspect='auto')
        axes[3].set_title('Uncertainty')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def plot_metrics(metrics: Dict[str, torch.Tensor], facies_names: Optional[List[str]] = None,
                save_path: Optional[str] = None):
    """
    Plot precision, recall, and F1 score for each class.
    
    Args:
        metrics: Dictionary containing precision, recall, f1_score
        facies_names: Names of facies classes
        save_path: Path to save figure
    """
    if facies_names is None:
        facies_names = [
            'Basement\nrocks',
            'Slope\nmudstone A', 
            'Mass-transport\ncomplex',
            'Slope\nmudstone B',
            'Slope\nvalley',
            'Submarine\ncanyon'
        ]
    
    precision = metrics['precision'].cpu().numpy()
    recall = metrics['recall'].cpu().numpy()
    f1 = metrics['f1_score'].cpu().numpy()
    
    x = np.arange(len(facies_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, recall, width, label='Recall', alpha=0.8)
    ax.bar(x, precision, width, label='Precision', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1 Score', alpha=0.8)
    
    ax.set_xlabel('Facies Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics by Facies Class', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(facies_names, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics plot to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm: torch.Tensor, facies_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (num_classes, num_classes)
        facies_names: Names of facies classes
        save_path: Path to save figure
    """
    if facies_names is None:
        facies_names = [
            'Basement',
            'Slope mud. A', 
            'MTC',
            'Slope mud. B',
            'Slope valley',
            'Sub. canyon'
        ]
    
    cm_np = cm.cpu().numpy()
    
    # Normalize by row (true labels)
    cm_norm = cm_np.astype('float') / (cm_np.sum(axis=1, keepdims=True) + 1e-10)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto')
    
    ax.set_xticks(np.arange(len(facies_names)))
    ax.set_yticks(np.arange(len(facies_names)))
    ax.set_xticklabels(facies_names, rotation=45, ha='right')
    ax.set_yticklabels(facies_names)
    
    # Add text annotations
    for i in range(len(facies_names)):
        for j in range(len(facies_names)):
            text = ax.text(j, i, f'{cm_norm[i, j]:.2f}\n({cm_np[i, j]})',
                          ha="center", va="center", 
                          color="white" if cm_norm[i, j] > 0.5 else "black",
                          fontsize=9)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


# ============================================================================
# Checkpoint Management
# ============================================================================

def save_checkpoint(model, optimizer, epoch, loss, metrics, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device='cuda'):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {filepath}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint.get('metrics', {})


if __name__ == "__main__":
    # Test metrics computation
    print("Testing metrics computation...")
    
    # Create dummy predictions and targets
    num_classes = 6
    pred = torch.randint(0, num_classes, (4, 200, 200))
    target = torch.randint(0, num_classes, (4, 200, 200))
    
    # Compute confusion matrix
    cm = compute_confusion_matrix(pred, target, num_classes)
    print(f"Confusion matrix shape: {cm.shape}")
    
    # Compute metrics
    metrics = compute_metrics(cm)
    print("\nMetrics:")
    for key, value in metrics.items():
        if 'mean' in key or isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✓ Utils test passed!")
