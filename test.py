"""
Testing and Inference Script for Seismic Facies Classification
Based on: "A deep learning framework for seismic facies classification" (Kaur et al., 2022)

Includes:
- Model evaluation on test data
- Uncertainty estimation using Monte Carlo Dropout
- Visualization of results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt

from model import DeepLabV3Plus, GANSegmentation, get_model
from utils import (
    evaluate_model,
    estimate_uncertainty,
    compute_prediction_entropy,
    visualize_prediction,
    plot_metrics,
    plot_confusion_matrix,
    load_checkpoint
)
from data_loader import SeismicFaciesDataset


class Tester:
    """
    Tester class for seismic facies classification models.
    
    Supports inference, evaluation, and uncertainty estimation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        device: str = 'cuda',
        num_classes: int = 6
    ):
        self.model = model.to(device)
        self.model_type = model_type.lower()
        self.device = device
        self.num_classes = num_classes
        
        # Facies names as per paper (Table 1)
        self.facies_names = [
            'Basement rocks',
            'Slope mudstone A',
            'Mass-transport complex',
            'Slope mudstone B',
            'Slope valley',
            'Submarine canyon'
        ]
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        load_checkpoint(self.model, None, checkpoint_path, self.device)
        print(f"Loaded model from {checkpoint_path}")
    
    def predict(self, seismic: torch.Tensor, use_dropout: bool = False) -> torch.Tensor:
        """
        Make predictions on seismic data.
        
        Args:
            seismic: Input seismic data (B, C, H, W)
            use_dropout: Whether to use dropout (for uncertainty estimation)
        
        Returns:
            Predicted labels (B, H, W)
        """
        self.model.eval()
        
        with torch.no_grad():
            seismic = seismic.to(self.device)
            
            if self.model_type == 'gan':
                outputs = self.model.generator(seismic, use_dropout=use_dropout)
            else:
                outputs = self.model(seismic, use_dropout=use_dropout)
            
            preds = torch.argmax(outputs, dim=1)
        
        return preds
    
    def predict_with_uncertainty(
        self, 
        seismic: torch.Tensor, 
        num_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation using MC Dropout.
        As described in the paper using Bayesian approximation.
        
        Args:
            seismic: Input seismic data (B, C, H, W)
            num_samples: Number of Monte Carlo samples
        
        Returns:
            Tuple of (predictions, uncertainty_map)
        """
        predictions, uncertainty = estimate_uncertainty(
            self.model if self.model_type != 'gan' else self.model.generator,
            seismic,
            num_samples=num_samples,
            device=self.device
        )
        
        return predictions, uncertainty
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for test/validation data
        
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating model...")
        
        if self.model_type == 'gan':
            eval_model = self.model.generator
        else:
            eval_model = self.model
        
        metrics = evaluate_model(
            eval_model,
            dataloader,
            self.device,
            self.num_classes
        )
        
        return metrics
    
    def test_and_visualize(
        self,
        test_loader: DataLoader,
        save_dir: str = './results',
        num_samples_to_visualize: int = 5,
        estimate_uncertainty_flag: bool = True,
        num_mc_samples: int = 20
    ):
        """
        Test model and visualize results.
        
        Args:
            test_loader: DataLoader for test data
            save_dir: Directory to save results
            num_samples_to_visualize: Number of samples to visualize
            estimate_uncertainty_flag: Whether to estimate uncertainty
            num_mc_samples: Number of MC samples for uncertainty estimation
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate model
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)
        
        metrics = self.evaluate(test_loader)
        
        print("\nOverall Metrics:")
        print(f"  Mean Precision: {metrics['mean_precision']:.4f}")
        print(f"  Mean Recall: {metrics['mean_recall']:.4f}")
        print(f"  Mean F1 Score: {metrics['mean_f1']:.4f}")
        
        print("\nPer-Class Metrics:")
        for i, name in enumerate(self.facies_names):
            print(f"  {name}:")
            print(f"    Precision: {metrics['precision'][i]:.4f}")
            print(f"    Recall: {metrics['recall'][i]:.4f}")
            print(f"    F1 Score: {metrics['f1_score'][i]:.4f}")
        
        # Plot metrics
        print("\nPlotting metrics...")
        plot_metrics(
            metrics,
            facies_names=self.facies_names,
            save_path=save_dir / f'{self.model_type}_metrics.png'
        )
        
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            facies_names=self.facies_names,
            save_path=save_dir / f'{self.model_type}_confusion_matrix.png'
        )
        
        # Visualize predictions
        print("\n" + "="*70)
        print("VISUALIZATION")
        print("="*70)
        
        self.model.eval()
        
        sample_count = 0
        for batch_idx, (seismic, labels) in enumerate(test_loader):
            if sample_count >= num_samples_to_visualize:
                break
            
            seismic = seismic.to(self.device)
            labels = labels.cpu().numpy()
            
            # Make predictions
            if estimate_uncertainty_flag:
                print(f"\nProcessing sample {sample_count + 1} with uncertainty estimation...")
                preds, uncertainty = self.predict_with_uncertainty(
                    seismic, 
                    num_samples=num_mc_samples
                )
                preds = preds.cpu().numpy()
                uncertainty = uncertainty.cpu().numpy()
            else:
                preds = self.predict(seismic).cpu().numpy()
                uncertainty = None
            
            # Visualize each sample in batch
            for i in range(seismic.size(0)):
                if sample_count >= num_samples_to_visualize:
                    break
                
                seismic_np = seismic[i, 0].cpu().numpy()
                label_np = labels[i]
                pred_np = preds[i]
                uncert_np = uncertainty[i] if uncertainty is not None else None
                
                save_path = save_dir / f'{self.model_type}_sample_{sample_count + 1}.png'
                
                visualize_prediction(
                    seismic_np,
                    label_np,
                    pred_np,
                    uncertainty=uncert_np,
                    save_path=str(save_path),
                    facies_names=self.facies_names
                )
                
                sample_count += 1
        
        print("\n" + "="*70)
        print(f"Results saved to {save_dir}")
        print("="*70)
    
    def predict_full_volume(
        self,
        seismic_volume: np.ndarray,
        batch_size: int = 16,
        estimate_uncertainty_flag: bool = False,
        num_mc_samples: int = 20
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict facies for a full 3D seismic volume.
        
        Args:
            seismic_volume: 3D seismic volume (N, H, W)
            batch_size: Batch size for prediction
            estimate_uncertainty_flag: Whether to estimate uncertainty
            num_mc_samples: Number of MC samples for uncertainty
        
        Returns:
            Tuple of (predictions, uncertainty) volumes
        """
        print(f"\nPredicting facies for volume of shape {seismic_volume.shape}...")
        
        # Create dataset
        dataset = SeismicFaciesDataset(seismic_volume, labels=None, normalize=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        uncertainties = [] if estimate_uncertainty_flag else None
        
        self.model.eval()
        
        with torch.no_grad():
            for seismic, _ in tqdm(dataloader, desc='Predicting'):
                seismic = seismic.to(self.device)
                
                if estimate_uncertainty_flag:
                    preds, uncert = self.predict_with_uncertainty(
                        seismic, 
                        num_samples=num_mc_samples
                    )
                    uncertainties.append(uncert.cpu().numpy())
                else:
                    preds = self.predict(seismic)
                
                predictions.append(preds.cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(predictions, axis=0)
        
        if estimate_uncertainty_flag:
            uncertainties = np.concatenate(uncertainties, axis=0)
        
        return predictions, uncertainties


def test_model(
    model_type: str,
    checkpoint_path: str,
    test_loader: DataLoader,
    device: str = 'cuda',
    save_dir: str = './results',
    visualize: bool = True,
    estimate_uncertainty: bool = True,
    num_mc_samples: int = 20
) -> Dict:
    """
    Convenience function to test a model.
    
    Args:
        model_type: 'deeplabv3+' or 'gan'
        checkpoint_path: Path to model checkpoint
        test_loader: Test data loader
        device: Device to run on
        save_dir: Directory to save results
        visualize: Whether to visualize results
        estimate_uncertainty: Whether to estimate uncertainty
        num_mc_samples: Number of MC samples for uncertainty
    
    Returns:
        Evaluation metrics dictionary
    """
    # Create model
    model = get_model(model_type, in_channels=1, num_classes=6)
    
    # Create tester
    tester = Tester(model, model_type, device=device)
    
    # Load checkpoint
    tester.load_checkpoint(checkpoint_path)
    
    # Test and visualize
    if visualize:
        tester.test_and_visualize(
            test_loader,
            save_dir=save_dir,
            num_samples_to_visualize=5,
            estimate_uncertainty_flag=estimate_uncertainty,
            num_mc_samples=num_mc_samples
        )
    
    # Evaluate
    metrics = tester.evaluate(test_loader)
    
    return metrics


def compare_models(
    deeplab_checkpoint: str,
    gan_checkpoint: str,
    test_loader: DataLoader,
    device: str = 'cuda',
    save_dir: str = './comparison'
):
    """
    Compare DeepLabv3+ and GAN models side by side.
    As described in the paper for comparative analysis.
    
    Args:
        deeplab_checkpoint: Path to DeepLabv3+ checkpoint
        gan_checkpoint: Path to GAN checkpoint
        test_loader: Test data loader
        device: Device to run on
        save_dir: Directory to save comparison results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON: DeepLabv3+ vs GAN")
    print("="*70)
    
    # Create models
    deeplab_model = get_model('deeplabv3+', in_channels=1, num_classes=6)
    gan_model = get_model('gan', in_channels=1, num_classes=6)
    
    # Create testers
    deeplab_tester = Tester(deeplab_model, 'deeplabv3+', device=device)
    gan_tester = Tester(gan_model, 'gan', device=device)
    
    # Load checkpoints
    deeplab_tester.load_checkpoint(deeplab_checkpoint)
    gan_tester.load_checkpoint(gan_checkpoint)
    
    # Evaluate both models
    print("\nEvaluating DeepLabv3+...")
    deeplab_metrics = deeplab_tester.evaluate(test_loader)
    
    print("\nEvaluating GAN...")
    gan_metrics = gan_tester.evaluate(test_loader)
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\n{'Metric':<30} {'DeepLabv3+':<15} {'GAN':<15}")
    print("-" * 70)
    print(f"{'Mean Precision':<30} {deeplab_metrics['mean_precision']:.4f}{'':<11} {gan_metrics['mean_precision']:.4f}")
    print(f"{'Mean Recall':<30} {deeplab_metrics['mean_recall']:.4f}{'':<11} {gan_metrics['mean_recall']:.4f}")
    print(f"{'Mean F1 Score':<30} {deeplab_metrics['mean_f1']:.4f}{'':<11} {gan_metrics['mean_f1']:.4f}")
    
    print("\nPer-Class F1 Scores:")
    print("-" * 70)
    facies_names = [
        'Basement rocks',
        'Slope mudstone A',
        'Mass-transport complex',
        'Slope mudstone B',
        'Slope valley',
        'Submarine canyon'
    ]
    
    for i, name in enumerate(facies_names):
        print(f"{name:<30} {deeplab_metrics['f1_score'][i]:.4f}{'':<11} {gan_metrics['f1_score'][i]:.4f}")
    
    # Side-by-side visualization
    print("\nCreating side-by-side visualizations...")
    
    for batch_idx, (seismic, labels) in enumerate(test_loader):
        if batch_idx >= 3:  # Visualize first 3 samples
            break
        
        seismic = seismic.to(device)
        labels = labels.cpu().numpy()
        
        # Get predictions from both models
        deeplab_preds = deeplab_tester.predict(seismic).cpu().numpy()
        gan_preds = gan_tester.predict(seismic).cpu().numpy()
        
        # Visualize first sample in batch
        seismic_np = seismic[0, 0].cpu().numpy()
        label_np = labels[0]
        deeplab_pred = deeplab_preds[0]
        gan_pred = gan_preds[0]
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(seismic_np, cmap='seismic', aspect='auto')
        axes[0].set_title('Seismic Section', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(label_np, cmap='tab10', vmin=0, vmax=5, aspect='auto')
        axes[1].set_title('True Labels', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(deeplab_pred, cmap='tab10', vmin=0, vmax=5, aspect='auto')
        axes[2].set_title('DeepLabv3+ Prediction', fontsize=12)
        axes[2].axis('off')
        
        axes[3].imshow(gan_pred, cmap='tab10', vmin=0, vmax=5, aspect='auto')
        axes[3].set_title('GAN Prediction', fontsize=12)
        axes[3].axis('off')
        
        plt.tight_layout()
        save_path = save_dir / f'comparison_sample_{batch_idx + 1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved comparison {batch_idx + 1} to {save_path}")
    
    print("\n" + "="*70)
    print(f"Comparison results saved to {save_dir}")
    print("="*70)
    
    return {
        'deeplab': deeplab_metrics,
        'gan': gan_metrics
    }


if __name__ == "__main__":
    # Test inference
    print("Testing inference...")
    
    from data_loader import create_dummy_data
    
    # Create dummy test data
    test_seismic, test_labels = create_dummy_data(num_samples=10, patch_size=200)
    test_dataset = SeismicFaciesDataset(test_seismic, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Create and test model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model('deeplabv3+', in_channels=1, num_classes=6)
    
    tester = Tester(model, 'deeplabv3+', device=device)
    
    # Test prediction
    seismic, _ = next(iter(test_loader))
    preds = tester.predict(seismic)
    print(f"Prediction shape: {preds.shape}")
    
    # Test uncertainty estimation
    preds_unc, uncertainty = tester.predict_with_uncertainty(seismic, num_samples=5)
    print(f"Uncertainty shape: {uncertainty.shape}")
    
    print("\n✓ Test inference passed!")
