"""
Data Loader for Seismic Facies Classification
Based on: "A deep learning framework for seismic facies classification" (Kaur et al., 2022)

This module handles loading and preprocessing of 3D seismic data.
Paper details:
- Patch size: 200 x 200 pixels
- Number of facies classes: 6
- Data: Parihaka Basin, New Zealand
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List
import warnings

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    warnings.warn("h5py not installed. HDF5 functionality will not be available.")


class SeismicFaciesDataset(Dataset):
    """
    Dataset class for seismic facies classification.
    
    Expected data format:
    - Seismic data: (N, H, W) where N is number of patches, H=W=200
    - Labels: (N, H, W) with integer values 0-5 for 6 facies classes
    
    Args:
        seismic_data: numpy array of seismic patches (N, H, W) or path to .npy file
        labels: numpy array of facies labels (N, H, W) or path to .npy file
        transform: Optional data augmentation transforms
        normalize: Whether to normalize seismic data (default: True)
    """
    
    def __init__(
        self,
        seismic_data,
        labels=None,
        transform=None,
        normalize=True
    ):
        # Load data if paths are provided
        if isinstance(seismic_data, (str, Path)):
            self.seismic_data = np.load(seismic_data)
        else:
            self.seismic_data = seismic_data
            
        if labels is not None:
            if isinstance(labels, (str, Path)):
                self.labels = np.load(labels)
            else:
                self.labels = labels
        else:
            self.labels = None
        
        # Validate data shapes
        if self.seismic_data.ndim != 3:
            raise ValueError(f"Expected 3D seismic data (N, H, W), got shape {self.seismic_data.shape}")
        
        if self.labels is not None:
            if self.seismic_data.shape != self.labels.shape:
                raise ValueError(
                    f"Seismic data shape {self.seismic_data.shape} doesn't match "
                    f"labels shape {self.labels.shape}"
                )
        
        self.transform = transform
        self.normalize = normalize
        
        # Compute normalization statistics if needed
        if self.normalize:
            self.mean = np.mean(self.seismic_data)
            self.std = np.std(self.seismic_data)
            if self.std == 0:
                self.std = 1.0
                warnings.warn("Standard deviation is 0, setting to 1.0")
    
    def __len__(self) -> int:
        return len(self.seismic_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Get seismic patch
        seismic = self.seismic_data[idx].astype(np.float32)
        
        # Normalize
        if self.normalize:
            seismic = (seismic - self.mean) / self.std
        
        # Add channel dimension (C=1 for grayscale seismic data)
        seismic = seismic[np.newaxis, ...]  # (1, H, W)
        
        # Convert to tensor
        seismic = torch.from_numpy(seismic)
        
        if self.labels is not None:
            label = self.labels[idx].astype(np.int64)
            label = torch.from_numpy(label)
            
            # Apply transforms if provided
            if self.transform is not None:
                seismic, label = self.transform(seismic, label)
            
            return seismic, label
        else:
            # Apply transforms if provided (inference mode)
            if self.transform is not None:
                seismic, _ = self.transform(seismic, None)
            
            return seismic, torch.tensor(-1)  # Dummy label for inference


def create_patches_from_volume(
    volume: np.ndarray,
    patch_size: int = 200,
    stride: int = 200,
    axis: int = 0
) -> np.ndarray:
    """
    Extract 2D patches from a 3D seismic volume.
    
    Args:
        volume: 3D seismic volume (D, H, W) where D is depth/inline/crossline
        patch_size: Size of square patches (default: 200 as per paper)
        stride: Stride for patch extraction (default: 200 for non-overlapping)
        axis: Axis along which to extract slices (0, 1, or 2)
    
    Returns:
        Array of patches (N, patch_size, patch_size)
    """
    patches = []
    
    # Get slices along specified axis
    if axis == 0:
        slices = volume
    elif axis == 1:
        slices = np.transpose(volume, (1, 0, 2))
    elif axis == 2:
        slices = np.transpose(volume, (2, 0, 1))
    else:
        raise ValueError(f"Invalid axis {axis}, must be 0, 1, or 2")
    
    # Extract patches from each slice
    for slice_2d in slices:
        h, w = slice_2d.shape
        
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = slice_2d[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
    
    return np.array(patches)


def get_dataloaders(
    train_seismic,
    train_labels,
    val_seismic,
    val_labels,
    batch_size: int = 32,  # As per paper
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_seismic: Training seismic data or path
        train_labels: Training labels or path
        val_seismic: Validation seismic data or path
        val_labels: Validation labels or path
        batch_size: Batch size (default: 32 as per paper)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = SeismicFaciesDataset(
        train_seismic,
        train_labels,
        normalize=True
    )
    
    val_dataset = SeismicFaciesDataset(
        val_seismic,
        val_labels,
        normalize=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def create_dummy_data(
    num_samples: int = 1000,
    patch_size: int = 200,
    num_classes: int = 6,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dummy seismic data for testing purposes.
    
    Args:
        num_samples: Number of samples to generate
        patch_size: Size of each patch (default: 200)
        num_classes: Number of facies classes (default: 6)
        save_path: Optional path to save the data
    
    Returns:
        Tuple of (seismic_data, labels)
    """
    # Generate synthetic seismic data with some structure
    seismic_data = np.random.randn(num_samples, patch_size, patch_size).astype(np.float32)
    
    # Add some horizontal layering to mimic seismic data
    for i in range(num_samples):
        for j in range(patch_size):
            seismic_data[i, j, :] += np.sin(j / 10) * 2
    
    # Generate labels with some spatial coherence
    labels = np.random.randint(0, num_classes, (num_samples, patch_size, patch_size)).astype(np.int64)
    
    # Add spatial smoothing to labels for more realistic facies
    from scipy.ndimage import gaussian_filter
    for i in range(num_samples):
        # Smooth and re-discretize
        smoothed = gaussian_filter(labels[i].astype(float), sigma=3)
        labels[i] = (smoothed * num_classes / smoothed.max()).astype(np.int64)
        labels[i] = np.clip(labels[i], 0, num_classes - 1)
    
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        np.save(Path(save_path) / "seismic_data.npy", seismic_data)
        np.save(Path(save_path) / "labels.npy", labels)
        print(f"Saved dummy data to {save_path}")
    
    return seismic_data, labels


if __name__ == "__main__":
    # Test data loader
    print("Testing SeismicFaciesDataset...")
    
    # Create dummy data
    seismic, labels = create_dummy_data(num_samples=100, patch_size=200)
    
    # Create dataset
    dataset = SeismicFaciesDataset(seismic, labels)
    print(f"Dataset size: {len(dataset)}")
    
    # Test getting an item
    sample_seismic, sample_label = dataset[0]
    print(f"Sample seismic shape: {sample_seismic.shape}")
    print(f"Sample label shape: {sample_label.shape}")
    print(f"Unique labels: {torch.unique(sample_label)}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_seismic, batch_labels = next(iter(loader))
    print(f"\nBatch seismic shape: {batch_seismic.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    
    print("\n✓ Data loader test passed!")
