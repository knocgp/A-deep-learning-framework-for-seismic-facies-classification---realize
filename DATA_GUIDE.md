# Data Usage Guide

## 📊 Data Requirements and Format

### Expected Data Structure

This implementation requires seismic data and corresponding facies labels in the following format:

```
data/
├── train/
│   ├── seismic_train.npy    # Training seismic patches
│   └── labels_train.npy     # Training facies labels
├── validation/
│   ├── seismic_val.npy      # Validation seismic patches
│   └── labels_val.npy       # Validation facies labels
└── test/
    ├── seismic_test.npy     # Test seismic patches
    └── labels_test.npy      # Test facies labels (optional)
```

---

## 🗂️ Data Format Specifications

### Seismic Data Format

**File type**: NumPy binary format (`.npy`)

**Array shape**: `(N, 200, 200)` where:
- `N` = Number of 2D patches
- `200 × 200` = Patch dimensions (as specified in paper)

**Data type**: `float32` or `float64`

**Value range**: 
- Raw amplitude values (will be normalized during loading)
- Typical range: [-1000, 1000] or normalized [-1, 1]

**Example creation**:
```python
import numpy as np

# Example: Create dummy seismic data
num_patches = 1000
seismic_data = np.random.randn(num_patches, 200, 200).astype(np.float32)

# Save
np.save('data/train/seismic_train.npy', seismic_data)
```

### Label Data Format

**File type**: NumPy binary format (`.npy`)

**Array shape**: `(N, 200, 200)` where:
- `N` = Number of 2D patches (must match seismic data)
- `200 × 200` = Patch dimensions

**Data type**: `int64` or `int32`

**Value range**: `[0, 1, 2, 3, 4, 5]` representing:
```python
FACIES_CLASSES = {
    0: "Basement rocks",
    1: "Slope mudstone A",
    2: "Mass-transport complex",
    3: "Slope mudstone B",
    4: "Slope valley",
    5: "Submarine canyon"
}
```

**Example creation**:
```python
# Example: Create dummy labels
labels = np.random.randint(0, 6, (num_patches, 200, 200)).astype(np.int64)

# Save
np.save('data/train/labels_train.npy', labels)
```

---

## 📦 Obtaining Real Seismic Data

### Option 1: Parihaka 3D Dataset (Paper Dataset)

**Source**: New Zealand Crown Minerals

**Access**: 
- Website: https://data.nzpam.govt.nz/
- Search: "Parihaka 3D"
- Free download (requires registration)

**File format**: SEG-Y or similar seismic formats

**Processing steps**:
1. Download Parihaka 3D seismic volume
2. Load SEG-Y data using `segyio` or `ObsPy`
3. Extract 2D inline/crossline sections
4. Create 200×200 patches
5. Match with provided interpretation labels from Chevron

**Example code**:
```python
import segyio
import numpy as np

# Load SEG-Y file
with segyio.open('parihaka_3d.sgy', 'r') as f:
    # Read all traces
    seismic_volume = segyio.tools.cube(f)
    
# seismic_volume shape: (n_inlines, n_crosslines, n_samples)
# Extract patches using the create_patches_from_volume() function
from data_loader import create_patches_from_volume

patches = create_patches_from_volume(
    seismic_volume, 
    patch_size=200, 
    stride=200,
    axis=0  # Extract along inline direction
)
```

### Option 2: Public Seismic Datasets

**F3 Block Dataset (Netherlands)**
- Website: https://terranubis.com/datainfo/F3-Demo-2020
- Format: SEG-Y
- Size: ~2 GB
- Free, no registration required

**SEAM Phase II Dataset**
- Synthetic seismic data
- Controlled geology
- Ground truth available
- Good for testing algorithms

**Penobscot 3D (Canada)**
- Open source via OpendTect
- Well-documented
- Multiple attributes available

### Option 3: Generate Synthetic Data

For algorithm testing and development:

```python
from data_loader import create_dummy_data

# Generate synthetic seismic-like data
train_seismic, train_labels = create_dummy_data(
    num_samples=27648,  # As per paper
    patch_size=200,
    num_classes=6,
    save_path='./data/train'
)
```

The synthetic data includes:
- Layered structure mimicking seismic reflections
- Gaussian noise
- Spatially coherent facies labels
- Suitable for code testing and development

---

## 🔄 Converting 3D Volume to Patches

### Method 1: Non-overlapping Patches

**Use case**: Training data to avoid data leakage

```python
from data_loader import create_patches_from_volume

# Load 3D volume (inline, crossline, time/depth)
volume = np.load('seismic_3d_volume.npy')  # Shape: (590, 782, 1006)

# Extract non-overlapping patches
patches = create_patches_from_volume(
    volume=volume,
    patch_size=200,
    stride=200,        # Non-overlapping
    axis=0             # Extract along inline direction
)

# Save patches
np.save('seismic_patches.npy', patches)
```

### Method 2: Overlapping Patches

**Use case**: Inference/prediction for smoother results

```python
# Extract overlapping patches
patches_overlap = create_patches_from_volume(
    volume=volume,
    patch_size=200,
    stride=100,        # 50% overlap
    axis=0
)
```

### Method 3: Multi-directional Extraction

**Use case**: Data augmentation

```python
# Extract from different directions
patches_inline = create_patches_from_volume(volume, 200, 200, axis=0)
patches_crossline = create_patches_from_volume(volume, 200, 200, axis=1)
patches_depth = create_patches_from_volume(volume, 200, 200, axis=2)

# Combine for training
all_patches = np.concatenate([patches_inline, patches_crossline, patches_depth])
```

---

## 🏷️ Creating Facies Labels

### Manual Interpretation Workflow

If you have seismic data but no labels:

#### 1. Seismic Interpretation Software
- **Petrel** (Schlumberger)
- **OpendTect** (dGB Earth Sciences) - Free for academic use
- **Kingdom** (IHS Markit)

#### 2. Interpretation Guidelines

Based on seismic attributes:

**Basement Rocks**:
- Strong reflection at base
- Low internal reflectivity
- High amplitude contrast

**Slope Mudstone A/B**:
- Parallel, continuous reflectors
- Moderate amplitude
- Uniform thickness

**Mass-Transport Complex**:
- Chaotic internal structure
- Variable amplitude
- Disrupted reflectors

**Slope Valley**:
- Linear, erosional features
- High-amplitude base
- Narrow, sinuous

**Submarine Canyon**:
- Large-scale erosional features
- Complex internal fill
- Multiple episodes

#### 3. Export Labels

Export interpretations as:
- Horizon surfaces → interpolate to grid
- Polygons → rasterize to patches
- Classification volumes → extract patches

#### 4. Label Format Conversion

```python
def convert_interpretation_to_labels(horizon_surfaces, patch_coords):
    """
    Convert horizon surfaces to facies labels
    
    Args:
        horizon_surfaces: Dict of {facies_id: surface_coordinates}
        patch_coords: Coordinates of extracted patches
    
    Returns:
        labels: numpy array (N, 200, 200)
    """
    labels = np.zeros((len(patch_coords), 200, 200), dtype=np.int64)
    
    for patch_idx, (i, j) in enumerate(patch_coords):
        for facies_id, surface in horizon_surfaces.items():
            # Determine which facies each pixel belongs to
            # Based on its position relative to horizon surfaces
            mask = determine_facies_from_horizons(i, j, surface)
            labels[patch_idx][mask] = facies_id
    
    return labels
```

---

## 🔍 Data Quality Checks

### Before Training

```python
import numpy as np

# Load data
seismic = np.load('seismic_train.npy')
labels = np.load('labels_train.npy')

# Check shapes match
assert seismic.shape == labels.shape, "Shape mismatch!"

# Check dimensions
assert seismic.ndim == 3, "Seismic should be 3D (N, H, W)"
assert seismic.shape[1] == 200, "Height should be 200"
assert seismic.shape[2] == 200, "Width should be 200"

# Check label values
unique_labels = np.unique(labels)
assert np.all((unique_labels >= 0) & (unique_labels < 6)), \
    f"Invalid labels: {unique_labels}"

# Check for NaN/Inf
assert not np.any(np.isnan(seismic)), "NaN values in seismic!"
assert not np.any(np.isinf(seismic)), "Inf values in seismic!"

# Check amplitude range
print(f"Seismic range: [{seismic.min():.2f}, {seismic.max():.2f}]")
print(f"Seismic mean: {seismic.mean():.2f}, std: {seismic.std():.2f}")

# Check class distribution
for i in range(6):
    count = np.sum(labels == i)
    percentage = 100 * count / labels.size
    print(f"Class {i}: {count:,} pixels ({percentage:.2f}%)")
```

---

## 📈 Data Statistics from Paper

### Parihaka Dataset

**Training Data**:
```
Number of patches: 27,648
Volume coverage: ~70% of Parihaka 3D
Dimensions: 200 × 200 pixels per patch
Total pixels: 27,648 × 200 × 200 = 1,105,920,000 pixels
```

**Validation Data**:
```
Remaining patches from same volume
Used for hyperparameter tuning
```

**Test Data**:
```
Adjacent seismic volume: 24,560 patches
Independent test set (not seen during training)
Used for final performance evaluation
```

**Facies Distribution** (approximate):
```
Class 0 (Basement):     ~15% - Regional basement surface
Class 1 (Slope mud A):  ~30% - Most common facies
Class 2 (MTC):          ~8%  - Episodic events
Class 3 (Slope mud B):  ~25% - Common background
Class 4 (Slope valley): ~7%  - Linear features
Class 5 (Canyon):       ~15% - Major systems
```

---

## 🛠️ Data Preprocessing

### Automatic Normalization

The `SeismicFaciesDataset` class automatically normalizes data:

```python
from data_loader import SeismicFaciesDataset

# Normalization is automatic
dataset = SeismicFaciesDataset(
    seismic_data='seismic_train.npy',
    labels='labels_train.npy',
    normalize=True  # Default: z-score normalization
)

# Access normalization parameters
print(f"Mean: {dataset.mean}")
print(f"Std: {dataset.std}")
```

### Custom Preprocessing

If you need custom preprocessing:

```python
import numpy as np

# Load raw data
seismic_raw = np.load('seismic_train.npy')

# Method 1: Z-score normalization (per volume)
seismic_norm = (seismic_raw - seismic_raw.mean()) / seismic_raw.std()

# Method 2: Min-max normalization
seismic_norm = (seismic_raw - seismic_raw.min()) / \
               (seismic_raw.max() - seismic_raw.min())

# Method 3: Per-patch normalization
seismic_norm = np.zeros_like(seismic_raw)
for i in range(len(seismic_raw)):
    patch = seismic_raw[i]
    seismic_norm[i] = (patch - patch.mean()) / patch.std()

# Save preprocessed data
np.save('seismic_train_normalized.npy', seismic_norm)
```

---

## 💾 Storage Requirements

### Disk Space Estimates

**Training data (27,648 patches)**:
```
Seismic: 27,648 × 200 × 200 × 4 bytes (float32) = 4.3 GB
Labels:  27,648 × 200 × 200 × 8 bytes (int64)   = 8.6 GB
Total:   ~13 GB
```

**Full Parihaka 3D volume**:
```
Dimensions: 1006 × 590 × 782
Size: 1006 × 590 × 782 × 4 bytes = 1.8 GB
```

**Recommendations**:
- Minimum: 20 GB free space (data + checkpoints + results)
- Recommended: 50 GB free space (allows for multiple experiments)

---

## 🔗 Additional Resources

### Seismic Data Processing Libraries

**Python**:
- `segyio`: Read/write SEG-Y files
- `obspy`: Seismic data processing
- `bruges`: Geophysical functions
- `segysak`: Seismic analysis

**Installation**:
```bash
pip install segyio obspy bruges segysak
```

### Example: Loading SEG-Y Data

```python
import segyio
import numpy as np

# Open SEG-Y file
with segyio.open('data.sgy', 'r', strict=False) as f:
    # Get dimensions
    n_traces = f.tracecount
    n_samples = len(f.samples)
    
    # Load all data
    data = np.array([f.trace[i] for i in range(n_traces)])
    
    # Reshape to 3D volume (if applicable)
    n_inline = len(f.ilines)
    n_xline = len(f.xlines)
    volume = data.reshape(n_inline, n_xline, n_samples)
    
    print(f"Volume shape: {volume.shape}")
```

---

## ❓ FAQ

**Q: Can I use different patch sizes?**
A: Yes, but you'll need to modify the model architectures. The paper uses 200×200 for optimal performance.

**Q: What if I don't have labels?**
A: You can:
1. Use unsupervised methods (not covered in this implementation)
2. Create labels manually using interpretation software
3. Use transfer learning from labeled datasets
4. Generate synthetic labels for initial testing

**Q: Can I use 3D patches instead of 2D?**
A: The current implementation uses 2D patches as per the paper. For 3D, you'd need to modify the model architectures significantly.

**Q: How do I handle missing data or noise?**
A: Preprocessing options:
1. Mask missing values and exclude from loss calculation
2. Apply denoising filters (median, bilateral, etc.)
3. Use robust normalization methods
4. Augment with synthetic noise during training

**Q: What's the minimum amount of training data?**
A: For reasonable performance:
- Minimum: 1,000 patches
- Recommended: 10,000+ patches  
- Paper: 27,648 patches

---

## 📧 Support

For data-related issues:
1. Check data format matches specifications above
2. Verify shapes and value ranges
3. Run quality checks script
4. See README.md for general troubleshooting

---

*This guide covers all aspects of preparing and using seismic data for facies classification.*
