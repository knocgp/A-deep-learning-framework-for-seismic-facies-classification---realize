# Seismic Facies Classification Using Deep Learning

> **[한국어 문서 (Korean Documentation)](./README_KR.md)** | **[Technical Details](./DOCUMENTATION.md)** | **[Data Guide](./DATA_GUIDE.md)**

PyTorch implementation of:

**"A deep learning framework for seismic facies classification"**  
*Harpreet Kaur, Nam Pham, Sergey Fomel, et al.*  
*Interpretation, Vol. 11, No. 1 (February 2023)*  
**DOI**: 10.1190/INT-2022-0048.1

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/knocgp/A-deep-learning-framework-for-seismic-facies-classification---realize)
[![Paper](https://img.shields.io/badge/Paper-Interpretation-green)](https://doi.org/10.1190/INT-2022-0048.1)
[![License](https://img.shields.io/badge/License-Research-orange)](LICENSE)

---

## 📋 Overview

> **Source**: Kaur et al. (2023) - Introduction section

This repository implements two state-of-the-art deep learning models for automatic seismic facies classification from seismic reflection data.

**Problem Statement** (from paper):
> "Conventional seismic facies identification requires manual analysis of seismic images and gathers by interpreters. For large 3D data sets, manual interpretation of seismic facies becomes labor-intensive and time-consuming. In addition, manual interpretation is subjective, relying on the interpreter's experience and skill."

**Solution**:
- **DeepLabv3+**: Sharp facies boundaries with ASPP multi-scale features
- **GAN-based Segmentation**: Better facies continuity with adversarial training
- **Uncertainty Estimation**: Bayesian approximation using Monte Carlo Dropout

---

## 🏗️ Model Architectures

### DeepLabv3+

> **Source**: Chen et al. (2018), "Encoder-decoder with atrous separable convolution for semantic image segmentation", ECCV

**Key Components** (from paper):

1. **Atrous Convolution**
   - "Atrous convolution refers to convolution with upsampled filters"
   - Enlarges field of view without increasing parameters

2. **ASPP (Atrous Spatial Pyramid Pooling)**
   - "Uses multiple parallel atrous convolution layers with different sampling rates"
   - Rates: 6, 12, 18 (as implemented)

3. **Modified Xception Backbone**
   - "All of the max-pooling operations are replaced by atrous separable convolutions"

4. **Encoder-Decoder Structure**
   - "Refines results for semantic segmentation, especially along the object boundaries"

**Characteristics**:
- ✅ Sharp facies boundaries
- ✅ Multi-scale feature extraction
- ✅ Precise edge detection

---

### GAN-based Segmentation

> **Sources**: 
> - Goodfellow et al. (2014), "Generative adversarial networks"
> - Luc et al. (2016), "Semantic segmentation using adversarial networks"

**Loss Function** (Paper Equation 3):

```
L = Σ l_mce(s(x_n), y_n) - λ(l_bce(a(x_n, y_n), 1) + l_bce(a(x_n, s(x_n)), 0))
```

Where:
- l_mce: Multiclass cross-entropy loss
- l_bce: Binary cross-entropy loss
- λ: Adversarial loss weight
- s(x): Segmentation model output
- a(x, y): Adversarial network output

**Components**:
- **Generator**: U-Net with skip connections
- **Discriminator**: PatchGAN (70×70 receptive field)

**Characteristics**:
- ✅ Better spatial continuity
- ✅ Natural-looking facies patterns
- ✅ Smooth transitions

---

## 📊 Dataset

> **Source**: Kaur et al. (2023) - Numerical examples section

**Data Provider**:
- 3D Seismic Volume: New Zealand Crown Minerals
- Training Labels: Chevron U.S.A. Inc.
- Location: Parihaka Basin, New Zealand

**Volume Specifications** (from paper):

```
Dimensions: 1006 × 590 × 782
- Inlines: 590 traces
- Crosslines: 782 traces
- Samples: 1006 time samples

Patch Size: 200 × 200 pixels
Training Patches: 27,648 (as specified in paper)
Validation: Remaining patches from same volume
Test Volume: 782 × 251 (adjacent volume)
```

---

### Facies Classes (6 types)

> **Source**: Kaur et al. (2023), Table 1

| Class | Facies Type | Seismic Characteristics |
|-------|-------------|------------------------|
| 0 | **Basement rocks** | Low SNR; few internal reflectors |
| 1 | **Slope mudstone A** | High-amplitude boundaries; low-amplitude continuous internal reflectors |
| 2 | **Mass-transport complex** | Mix of chaotic facies and low-amplitude parallel reflectors |
| 3 | **Slope mudstone B** | High-amplitude parallel reflectors; low-continuity scour surfaces |
| 4 | **Slope valley** | High-amplitude incised channels; relatively low relief |
| 5 | **Submarine canyon** | Low-amplitude mix of parallel surfaces and chaotic reflectors |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/knocgp/A-deep-learning-framework-for-seismic-facies-classification---realize.git
cd A-deep-learning-framework-for-seismic-facies-classification---realize

# Install dependencies
pip install torch torchvision numpy matplotlib scipy tqdm
# Or use requirements file
pip install -r requirements.txt
```

### Project Structure

```
├── data_loader.py      # Data loading and preprocessing
├── model.py            # DeepLabv3+ and GAN models
├── utils.py            # Metrics, uncertainty, visualization
├── train.py            # Training logic
├── test.py             # Testing and inference
├── main.ipynb          # Complete workflow notebook
├── README.md           # This file (English)
├── README_KR.md        # Korean documentation
├── DOCUMENTATION.md    # Technical details
└── DATA_GUIDE.md       # Data preparation guide
```

---

## 💻 Usage

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook main.ipynb
```

The notebook includes:
1. Data loading and visualization
2. Model training (DeepLabv3+ and GAN)
3. Evaluation and comparison
4. Uncertainty analysis
5. Result visualization

---

### Option 2: Python Scripts

**Train a model:**

```python
from train import train_model
from data_loader import get_dataloaders

# Load your data
train_loader, val_loader = get_dataloaders(
    train_seismic, train_labels,
    val_seismic, val_labels,
    batch_size=32  # As specified in paper
)

# Train DeepLabv3+
history = train_model(
    model_type='deeplabv3+',
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=60,  # As specified in paper
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

---

## 🔧 Configuration

### Hyperparameters (From Paper)

> **Source**: Kaur et al. (2023) - Implementation section

**Specified in Paper**:
```python
batch_size = 32          # "batch size of 32"
num_epochs = 60          # "60 epochs" for GAN
optimizer = "Adam"       # "Adam optimizer"
patch_size = 200         # "200 × 200 samples"
num_classes = 6          # 6 facies types
```

**Not Specified (Using Standard Values)**:
```python
learning_rate = 1e-4     # Standard Adam learning rate
mc_samples = 20          # For uncertainty estimation
```

---

## 📈 Results

### Performance Metrics

> **Source**: Kaur et al. (2023) - Performance metric section (Equations 4, 5, 6)

```
Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

---

### Key Findings (From Paper)

**DeepLabv3+ Characteristics**:
> "DeepLab v3+ output captures sharper boundaries between the facies by gradually capturing spatial information using ASPP"

**GAN Characteristics**:
> "GAN output shows improved continuity of predicted facies"

**Comparative Analysis** (Figures 3, 4, 5):
- "DeepLab v3+ has picked up sharper facies boundaries"
- "The facies boundaries picked by GAN are smooth"
- "The continuity of predicted facies is better preserved by GAN"

**Recommendation** (Conclusion):
> "The joint analysis of the output of multiple networks provides a more accurate interpretation of predicted facies"

---

## 🧪 Uncertainty Estimation

> **Source**: Gal & Ghahramani (2016), "Dropout as a Bayesian approximation"

**Method** (from paper):
> "We use dropout at the inference time and compute multiple predictions for each pixel. The use of dropout layers in neural networks is equivalent to Bayesian approximation"

**Implementation**:

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

**Interpretation** (from paper):
> "Uncertainty values are overall low except at the boundaries of the facies, which is the transition zone from one facies type to another"

---

## 📁 Data Format

### Input Data Format

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

For detailed data preparation instructions, see **[DATA_GUIDE.md](./DATA_GUIDE.md)**

---

## 📚 References

### Primary Paper
Kaur, H., Pham, N., Fomel, S., Geng, Z., Decker, L., Gremillion, B., Jervis, M., Abma, R., & Gao, S. (2023). A deep learning framework for seismic facies classification. *Interpretation*, 11(1), T107-T116. doi:10.1190/INT-2022-0048.1

### Seismic Facies Methodology
Sheriff, R.E. (1976). Inferring stratigraphy from seismic data. *AAPG Bulletin*, 60(4), 528-542. doi:10.1306/83D923F7-16C7-11D7-8645000102C1865D

### DeepLabv3+ Architecture
Chen, L.C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. *Proceedings of the European Conference on Computer Vision (ECCV)*, 801-818.

Chen, L.C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A.L. (2017). DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40, 834-848. doi:10.1109/TPAMI.2017.2699184

### GAN for Segmentation
Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial networks. *arXiv preprint*, arXiv:1406.2661.

Luc, P., Couprie, C., Chintala, S., & Verbeek, J. (2016). Semantic segmentation using adversarial networks. *arXiv preprint*, arXiv:1611.08408.

### Bayesian Deep Learning
Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *Proceedings of the 33rd International Conference on Machine Learning*, 1050-1059.

Kendall, A. & Gal, Y. (2017). What uncertainties do we need in bayesian deep learning for computer vision? *Advances in Neural Information Processing Systems*, 30.

---

## ⚠️ Important Notes

### Verified Information
The following information is **directly from the paper**:
- ✅ Dataset specifications (volume size, patch numbers)
- ✅ 6 facies class descriptions (Table 1)
- ✅ Training parameters (batch size, epochs)
- ✅ Model architectures
- ✅ Performance metric equations

### Implementation Details
The following were **not specified in the paper**:
- ⚠️ Learning rate (using standard value: 1e-4)
- ⚠️ MC samples for uncertainty (using common value: 20)

### Data Usage Rights
- Parihaka 3D data: Owned by New Zealand Crown Minerals
- Training labels: Provided by Chevron U.S.A. Inc.
- Commercial use requires permission from respective organizations

---

## 🔍 Key Features

- ✅ Complete implementation of DeepLabv3+ for seismic data
- ✅ GAN-based segmentation with adversarial training
- ✅ Uncertainty estimation using Bayesian approximation
- ✅ Comprehensive evaluation metrics (Precision, Recall, F1)
- ✅ Visualization tools for results and uncertainty
- ✅ Model comparison utilities
- ✅ Full workflow in Jupyter notebook
- ✅ Modular code structure for easy extension

---

## 🛠️ Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- SciPy
- tqdm
- h5py (optional, for HDF5 data)

---

## 📖 Additional Documentation

- **[README_KR.md](./README_KR.md)**: Korean documentation with detailed explanations
- **[DOCUMENTATION.md](./DOCUMENTATION.md)**: Comprehensive technical documentation
- **[DATA_GUIDE.md](./DATA_GUIDE.md)**: Complete data preparation guide
- **[PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md)**: Project completion summary

---

## 📧 Contact

For questions or issues:
1. Check the paper first
2. Refer to the documentation files
3. Open an issue on GitHub

---

## 📄 License

This implementation is for research and educational purposes.

**Data Usage**:
- Parihaka 3D: Follow New Zealand Crown Minerals policies
- Training labels: Provided by Chevron U.S.A. Inc., check usage permissions

---

## 📝 Citation

If you use this code, please cite the original paper:

```bibtex
@article{kaur2023deep,
  title={A deep learning framework for seismic facies classification},
  author={Kaur, Harpreet and Pham, Nam and Fomel, Sergey and 
          Geng, Zhicheng and Decker, Luke and Gremillion, Ben and 
          Jervis, Michael and Abma, Ray and Gao, Shuang},
  journal={Interpretation},
  volume={11},
  number={1},
  pages={T107--T116},
  year={2023},
  publisher={Society of Exploration Geophysicists and 
            American Association of Petroleum Geologists},
  doi={10.1190/INT-2022-0048.1}
}
```

---

**Repository**: https://github.com/knocgp/A-deep-learning-framework-for-seismic-facies-classification---realize

**Last Updated**: November 1, 2025
