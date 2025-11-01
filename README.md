# Seismic Facies Classification Using Deep Learning

PyTorch implementation of:

**"A deep learning framework for seismic facies classification"**  
*Harpreet Kaur, Nam Pham, Sergey Fomel, et al.*  
*Interpretation, Vol. 11, No. 1 (February 2023)*

## 📋 Overview

This repository implements two state-of-the-art deep learning models for automatic seismic facies classification:
- **DeepLabv3+**: Uses atrous convolutions and encoder-decoder architecture for sharp facies boundaries
- **GAN-based Segmentation**: Combines multiclass cross-entropy and adversarial loss for better facies continuity

The framework includes uncertainty estimation using Bayesian approximation with Monte Carlo Dropout.

## 🏗️ Architecture

### DeepLabv3+
- Modified Xception backbone with atrous separable convolutions
- Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context
- Encoder-decoder with low-level feature fusion
- Produces sharp facies boundaries

### GAN Segmentation
- U-Net generator with skip connections
- PatchGAN discriminator
- Combined MCE + adversarial loss
- Better facies continuity

## 📊 Dataset

**Source**: Parihaka Basin, New Zealand (provided by New Zealand Crown Minerals)
**Training labels**: Provided by Chevron U.S.A. Inc.

### Facies Classes (6 types)
0. Basement rocks
1. Slope mudstone A
2. Mass-transport complex
3. Slope mudstone B
4. Slope valley
5. Submarine canyon

### Data Format
- **Input**: 2D seismic patches (200 × 200 pixels)
- **Output**: Pixel-wise facies classification (200 × 200)
- **Training patches**: 27,648 (as per paper)
- **Validation patches**: Remaining from 3D volume

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd webapp

# Install dependencies
pip install torch torchvision numpy matplotlib scipy tqdm h5py
# Or if you have a requirements.txt:
pip install -r requirements.txt
```

### Project Structure

```
webapp/
├── data_loader.py      # Data loading and preprocessing
├── model.py            # DeepLabv3+ and GAN models
├── utils.py            # Metrics, uncertainty, visualization
├── train.py            # Training logic
├── test.py             # Testing and inference
├── main.ipynb          # Complete workflow notebook
├── README.md           # This file
├── data/               # Data directory (create if needed)
├── checkpoints/        # Saved models
└── results/            # Output results and visualizations
```

### Usage

#### Option 1: Jupyter Notebook (Recommended)

Open and run `main.ipynb` for the complete workflow:
```bash
jupyter notebook main.ipynb
```

The notebook includes:
1. Data loading and visualization
2. Model training (DeepLabv3+ and GAN)
3. Evaluation and comparison
4. Uncertainty analysis
5. Result visualization

#### Option 2: Python Scripts

**Train a model:**
```python
from train import train_model
from data_loader import get_dataloaders

# Load your data
train_loader, val_loader = get_dataloaders(
    train_seismic, train_labels,
    val_seismic, val_labels,
    batch_size=32
)

# Train DeepLabv3+
history = train_model(
    model_type='deeplabv3+',
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=60,
    device='cuda'
)
```

**Test a model:**
```python
from test import test_model

metrics = test_model(
    model_type='deeplabv3+',
    checkpoint_path='checkpoints/deeplabv3+_best.pth',
    test_loader=test_loader,
    device='cuda',
    visualize=True,
    estimate_uncertainty=True
)
```

**Compare models:**
```python
from test import compare_models

results = compare_models(
    deeplab_checkpoint='checkpoints/deeplabv3+_best.pth',
    gan_checkpoint='checkpoints/gan_best.pth',
    test_loader=test_loader,
    device='cuda'
)
```

## 🔧 Configuration

Key hyperparameters from the paper:

```python
CONFIG = {
    'patch_size': 200,        # Patch size (200×200)
    'num_classes': 6,         # Number of facies classes
    'batch_size': 32,         # Batch size
    'num_epochs': 60,         # Training epochs
    'learning_rate': 1e-4,    # Adam learning rate
    'num_mc_samples': 20,     # MC dropout samples for uncertainty
}
```

### Notes on Hyperparameters

- **Batch size**: 32 (as specified in paper)
- **Epochs**: 60 for GAN (as specified in paper)
- **Optimizer**: Adam (specified in paper)
- **Learning rate**: 1e-4 (not specified, using standard value)
- **MC samples**: 20 (not specified, using common value)
- **Middle flow repetitions**: 4 (reduced from 8 for efficiency)

## 📈 Results

### Performance Metrics

Models are evaluated using:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Key Findings (from paper)

1. **DeepLabv3+**:
   - Sharper boundaries between facies
   - Better at capturing transitions
   - Uses ASPP for multi-scale features

2. **GAN**:
   - Better continuity of predicted facies
   - Smoother predictions
   - Less sensitive to noise

3. **Uncertainty**:
   - High at facies boundaries
   - Indicates misprediction regions
   - Useful for quality control

## 🧪 Testing the Implementation

Test individual modules:

```bash
# Test data loader
cd /home/user/webapp && python data_loader.py

# Test models
cd /home/user/webapp && python model.py

# Test utils
cd /home/user/webapp && python utils.py

# Test training (short run)
cd /home/user/webapp && python train.py

# Test inference
cd /home/user/webapp && python test.py
```

## 📊 Uncertainty Estimation

The framework implements epistemic uncertainty using Monte Carlo Dropout:

```python
from test import Tester
from model import get_model

model = get_model('deeplabv3+')
tester = Tester(model, 'deeplabv3+')
tester.load_checkpoint('checkpoints/deeplabv3+_best.pth')

# Predict with uncertainty
predictions, uncertainty = tester.predict_with_uncertainty(
    seismic_data,
    num_samples=20  # MC samples
)
```

## 📁 Data Format

### Input Data
Load your seismic data as numpy arrays:

```python
import numpy as np

# Seismic data: (N, 200, 200) - N patches of 200×200 pixels
train_seismic = np.load('train_seismic.npy')

# Labels: (N, 200, 200) - Integer values 0-5
train_labels = np.load('train_labels.npy')
```

### Creating Patches from 3D Volume

```python
from data_loader import create_patches_from_volume

# 3D volume: (depth, height, width)
volume = np.load('seismic_volume.npy')

# Extract patches
patches = create_patches_from_volume(
    volume,
    patch_size=200,
    stride=200,  # Non-overlapping
    axis=0       # Extract along depth
)
```

## 🎯 Model Outputs

### DeepLabv3+
- Sharp facies boundaries
- Better edge detection
- ASPP captures multi-scale features

### GAN
- Smooth facies transitions
- Better spatial continuity
- Adversarial training improves coherence

### Joint Analysis
Combining both models (as recommended in paper):
- Leverage strengths of both approaches
- More robust predictions
- Better interpretation confidence

## 📝 Citation

If you use this code, please cite the original paper:

```bibtex
@article{kaur2023deep,
  title={A deep learning framework for seismic facies classification},
  author={Kaur, Harpreet and Pham, Nam and Fomel, Sergey and Geng, Zhicheng and Decker, Luke and Gremillion, Ben and Jervis, Michael and Abma, Ray and Gao, Shuang},
  journal={Interpretation},
  volume={11},
  number={1},
  pages={T107--T116},
  year={2023},
  publisher={Society of Exploration Geophysicists and American Association of Petroleum Geologists}
}
```

## 🔍 Key Features

- ✅ Complete implementation of DeepLabv3+ for seismic data
- ✅ GAN-based segmentation with adversarial training
- ✅ Uncertainty estimation using Bayesian approximation
- ✅ Comprehensive evaluation metrics (Precision, Recall, F1)
- ✅ Visualization tools for results and uncertainty
- ✅ Model comparison utilities
- ✅ Full workflow in Jupyter notebook
- ✅ Modular code structure for easy extension

## 🛠️ Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- SciPy
- tqdm
- h5py (optional, for HDF5 data)

## 📞 Contact

For questions or issues, please refer to the original paper or open an issue in this repository.

## 📄 License

This implementation is for research and educational purposes. Please refer to the original paper for data usage rights.

---

**Paper**: Kaur et al. (2023), "A deep learning framework for seismic facies classification", *Interpretation*, 11(1), T107-T116.

**DOI**: 10.1190/INT-2022-0048.1
