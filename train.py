"""
Training Script for Seismic Facies Classification
Based on: "A deep learning framework for seismic facies classification" (Kaur et al., 2022)

Training parameters from paper:
- Batch size: 32
- Epochs: 60 (for GAN)
- Optimizer: Adam
- Training patches: 27,648
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
from typing import Dict, Optional, Tuple

from model import DeepLabV3Plus, GANSegmentation
from utils import (
    CombinedLoss, 
    compute_confusion_matrix, 
    compute_metrics,
    save_checkpoint,
    load_checkpoint
)


class Trainer:
    """
    Trainer class for seismic facies classification models.
    
    Supports both DeepLabv3+ and GAN-based training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_type: str,  # 'deeplabv3+' or 'gan'
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,  # Default Adam lr, not specified in paper
        num_epochs: int = 60,  # As per paper for GAN
        checkpoint_dir: str = './checkpoints',
        num_classes: int = 6
    ):
        self.model = model.to(device)
        self.model_type = model_type.lower()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer and loss
        if self.model_type == 'deeplabv3+' or self.model_type == 'deeplab':
            # DeepLabv3+ training
            self.optimizer = optim.Adam(
                model.parameters(), 
                lr=learning_rate,
                weight_decay=1e-4  # Standard weight decay
            )
            self.criterion = nn.CrossEntropyLoss()
            self.discriminator_optimizer = None
            
        elif self.model_type == 'gan':
            # GAN training - separate optimizers for generator and discriminator
            self.optimizer = optim.Adam(
                model.generator.parameters(),
                lr=learning_rate,
                betas=(0.5, 0.999)  # Standard GAN parameters
            )
            self.discriminator_optimizer = optim.Adam(
                model.discriminator.parameters(),
                lr=learning_rate,
                betas=(0.5, 0.999)
            )
            self.criterion = CombinedLoss(lambda_adv=0.1, num_classes=num_classes)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Learning rate scheduler
        # Comment: Not specified in paper, using ReduceLROnPlateau for adaptive learning
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        self.best_val_loss = float('inf')
        self.start_epoch = 0
    
    def train_epoch_deeplab(self, epoch: int) -> Tuple[float, Dict]:
        """Train one epoch for DeepLabv3+."""
        self.model.train()
        total_loss = 0.0
        total_cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs} [Train]')
        
        for batch_idx, (seismic, labels) in enumerate(pbar):
            seismic = seismic.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(seismic)
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                cm = compute_confusion_matrix(preds.cpu(), labels.cpu(), self.num_classes)
                total_cm += cm
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_metrics(total_cm)
        
        return avg_loss, metrics
    
    def train_epoch_gan(self, epoch: int) -> Tuple[float, Dict]:
        """Train one epoch for GAN-based segmentation."""
        self.model.train()
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        total_cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs} [Train]')
        
        for batch_idx, (seismic, labels) in enumerate(pbar):
            seismic = seismic.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = seismic.size(0)
            
            # ===============================================
            # Train Discriminator
            # ===============================================
            self.discriminator_optimizer.zero_grad()
            
            # Generate fake labels
            with torch.no_grad():
                fake_logits = self.model.generator(seismic)
                fake_labels = torch.argmax(fake_logits, dim=1)
            
            # Discriminator outputs
            real_output = self.model.discriminator(seismic, labels)
            fake_output = self.model.discriminator(seismic, fake_labels)
            
            # Discriminator loss
            real_target = torch.ones_like(real_output)
            fake_target = torch.zeros_like(fake_output)
            
            bce_loss = nn.BCEWithLogitsLoss()
            disc_loss = (bce_loss(real_output, real_target) + 
                        bce_loss(fake_output, fake_target)) / 2
            
            disc_loss.backward()
            self.discriminator_optimizer.step()
            
            # ===============================================
            # Train Generator
            # ===============================================
            self.optimizer.zero_grad()
            
            # Generate labels
            gen_logits = self.model.generator(seismic)
            gen_labels = torch.argmax(gen_logits, dim=1)
            
            # Generator wants discriminator to think fake labels are real
            disc_fake_output = self.model.discriminator(seismic, gen_labels)
            
            # Combined loss
            gen_loss, mce_loss, _, _ = self.criterion(
                gen_logits, labels, real_output.detach(), disc_fake_output
            )
            
            gen_loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_gen_loss += gen_loss.item()
            total_disc_loss += disc_loss.item()
            
            with torch.no_grad():
                preds = torch.argmax(gen_logits, dim=1)
                cm = compute_confusion_matrix(preds.cpu(), labels.cpu(), self.num_classes)
                total_cm += cm
            
            # Update progress bar
            pbar.set_postfix({
                'gen_loss': gen_loss.item(),
                'disc_loss': disc_loss.item()
            })
        
        avg_loss = total_gen_loss / len(self.train_loader)
        metrics = compute_metrics(total_cm)
        
        return avg_loss, metrics
    
    def validate(self, epoch: int) -> Tuple[float, Dict]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        
        criterion = nn.CrossEntropyLoss()
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.num_epochs} [Val]')
        
        with torch.no_grad():
            for seismic, labels in pbar:
                seismic = seismic.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.model_type == 'gan':
                    outputs = self.model.generator(seismic)
                else:
                    outputs = self.model(seismic)
                
                # Compute loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Track metrics
                preds = torch.argmax(outputs, dim=1)
                cm = compute_confusion_matrix(preds.cpu(), labels.cpu(), self.num_classes)
                total_cm += cm
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(total_cm)
        
        return avg_loss, metrics
    
    def train(self):
        """Main training loop."""
        print(f"\nTraining {self.model_type.upper()} model...")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print("-" * 70)
        
        for epoch in range(self.start_epoch, self.num_epochs):
            start_time = time.time()
            
            # Train
            if self.model_type == 'gan':
                train_loss, train_metrics = self.train_epoch_gan(epoch + 1)
            else:
                train_loss, train_metrics = self.train_epoch_deeplab(epoch + 1)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch + 1)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.num_epochs} Summary:")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train F1: {train_metrics['mean_f1']:.4f} | Val F1: {val_metrics['mean_f1']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = self.checkpoint_dir / f'{self.model_type}_best.pth'
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    val_loss,
                    val_metrics,
                    checkpoint_path
                )
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.checkpoint_dir / f'{self.model_type}_epoch_{epoch+1}.pth'
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    val_loss,
                    val_metrics,
                    checkpoint_path
                )
            
            print("-" * 70)
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        epoch, loss, metrics = load_checkpoint(
            self.model, 
            self.optimizer, 
            checkpoint_path, 
            self.device
        )
        self.start_epoch = epoch
        print(f"Resuming training from epoch {epoch}")


def train_model(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 60,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
    checkpoint_dir: str = './checkpoints',
    resume_from: Optional[str] = None
) -> Dict:
    """
    Convenience function to train a model.
    
    Args:
        model_type: 'deeplabv3+' or 'gan'
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs (default: 60 as per paper)
        learning_rate: Learning rate (default: 1e-4)
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
    
    Returns:
        Training history dictionary
    """
    # Create model
    if model_type.lower() in ['deeplabv3+', 'deeplab']:
        model = DeepLabV3Plus(in_channels=1, num_classes=6)
    elif model_type.lower() == 'gan':
        model = GANSegmentation(in_channels=1, num_classes=6)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    if resume_from:
        trainer.resume_from_checkpoint(resume_from)
    
    # Train
    history = trainer.train()
    
    return history


if __name__ == "__main__":
    # Test training setup
    print("Testing training setup...")
    
    from data_loader import create_dummy_data, get_dataloaders
    
    # Create dummy data
    train_seismic, train_labels = create_dummy_data(num_samples=100, patch_size=200)
    val_seismic, val_labels = create_dummy_data(num_samples=20, patch_size=200)
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        train_seismic, train_labels,
        val_seismic, val_labels,
        batch_size=4,
        num_workers=0
    )
    
    # Test DeepLabv3+ training
    print("\nTesting DeepLabv3+ training (2 epochs)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    history = train_model(
        model_type='deeplabv3+',
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        device=device,
        checkpoint_dir='./test_checkpoints'
    )
    
    print("\n✓ Training test passed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
