# Seismic Facies Classification - Technical Documentation

## 📚 Table of Contents
1. [Geophysical Background](#geophysical-background)
2. [Geological Context](#geological-context)
3. [Machine Learning Principles](#machine-learning-principles)
4. [Data Description](#data-description)
5. [Model Architecture Details](#model-architecture-details)
6. [Training Methodology](#training-methodology)

---

## 🌍 Geophysical Background

### Seismic Reflection Method

Seismic facies classification relies on the fundamental principles of **seismic reflection geophysics**:

#### 1. Wave Propagation
- **Seismic waves** are artificially generated (explosives, vibrators) and propagate through Earth's subsurface
- When waves encounter boundaries between rock layers with different **acoustic impedances**, they reflect back
- Acoustic impedance (Z) = ρ × V, where:
  - ρ = rock density (kg/m³)
  - V = seismic wave velocity (m/s)

#### 2. Reflection Coefficient
The amount of energy reflected at an interface depends on the contrast in acoustic impedance:

```
R = (Z₂ - Z₁) / (Z₂ + Z₁)
```

Where:
- R = reflection coefficient
- Z₁ = acoustic impedance of upper layer
- Z₂ = acoustic impedance of lower layer

#### 3. Seismic Attributes
Different rock types exhibit distinct seismic characteristics:

| Attribute | Description | Geological Significance |
|-----------|-------------|------------------------|
| **Amplitude** | Strength of reflected signal | Indicates impedance contrast, fluid content |
| **Frequency** | Number of wave cycles per second | Related to bed thickness, attenuation |
| **Phase** | Timing of wave arrival | Indicates lithology changes |
| **Continuity** | Lateral consistency of reflectors | Shows depositional environment |

#### 4. Resolution Limits
- **Vertical resolution**: λ/4 (typically 10-30 meters)
- **Horizontal resolution**: Fresnel zone radius (50-200 meters)
- These limits affect the detectability of thin beds and small features

### Seismic Data Acquisition

**3D Seismic Surveys** used in this study:
- **Source**: Air guns, vibroseis, or dynamite
- **Receivers**: Hydrophones (marine) or geophones (land)
- **Recording**: Multi-channel recording with thousands of traces
- **Processing**: Migration, stacking, filtering to create 3D volume

---

## 🗻 Geological Context

### Study Area: Parihaka Basin, New Zealand

The **Parihaka 3D seismic volume** is located in the **Taranaki Basin**, offshore New Zealand:

#### Basin Evolution
1. **Late Cretaceous-Paleocene**: Passive margin development
2. **Eocene-Oligocene**: Marine transgression
3. **Miocene-Present**: Active margin tectonics with subduction

#### Depositional Environments
The 6 facies classes represent distinct geological settings:

### 1. Basement Rocks (Class 0)
**Geological Characteristics:**
- **Age**: Pre-Cretaceous crystalline or metamorphic rocks
- **Composition**: Schist, granite, or volcanic rocks
- **Seismic Signature**: 
  - Low signal-to-noise ratio
  - Few internal reflectors
  - High acoustic impedance contrast with overlying sediments

**Geophysical Properties:**
- Density: 2.6-2.9 g/cm³
- P-wave velocity: 5000-6500 m/s
- High impedance creates strong reflection at basement-sediment interface

### 2. Slope Mudstone A (Class 1)
**Geological Characteristics:**
- **Depositional Setting**: Continental slope, deep marine
- **Lithology**: Fine-grained hemipelagic muds
- **Age**: Miocene-Pliocene

**Seismic Signature:**
- High-amplitude upper and lower boundaries
- Low-amplitude continuous/semicontinuous internal reflectors
- Parallel to sub-parallel bedding

**Sedimentological Process:**
- Slow background sedimentation from suspension
- Low energy environment
- Minimal bioturbation preservation

### 3. Mass Transport Complex (MTC) (Class 2)
**Geological Characteristics:**
- **Process**: Submarine landslides and debris flows
- **Trigger**: Seismic activity, over-steepening, rapid sedimentation
- **Scale**: Can extend for kilometers

**Seismic Signature:**
- **Chaotic reflectors**: Disrupted internal structure
- **Low-amplitude parallel reflectors**: Remolded sediments
- **Irregular top surface**: Hummocky morphology
- **Basal shear surface**: Erosional contact

**Geohazard Significance:**
- Important for offshore infrastructure planning
- Indicates paleoseismic activity
- Affects reservoir quality

### 4. Slope Mudstone B (Class 3)
**Geological Characteristics:**
- **Setting**: Mid to lower slope
- **Differences from Type A**: Different age, slightly different composition

**Seismic Signature:**
- High-amplitude parallel reflectors
- Better continuity than Type A
- Low-continuity scour surfaces from bottom currents

**Depositional Processes:**
- Hemipelagic sedimentation
- Occasional turbidity current activity
- Contour current influence

### 5. Slope Valley (Class 4)
**Geological Characteristics:**
- **Genesis**: Erosional features on continental slope
- **Dimensions**: 100s of meters wide, 10s of meters deep
- **Function**: Sediment bypass and transport

**Seismic Signature:**
- High-amplitude incised channels
- Relatively low relief
- V-shaped to U-shaped cross-sections
- Levee deposits on flanks

**Sedimentological Significance:**
- Pathways for turbidity currents
- Connect shelf to deep basin
- Important for sand distribution prediction

### 6. Submarine Canyon (Class 5)
**Geological Characteristics:**
- **Scale**: Kilometers wide, 100s of meters deep
- **Formation**: Long-term erosion by turbidity currents and bottom currents
- **Modern analogs**: Congo Canyon, Monterey Canyon

**Seismic Signature:**
- Low-amplitude mix of parallel surfaces and chaotic reflectors
- Complex internal architecture
- Multiple cut-and-fill sequences
- Channel-levee systems

**Petroleum System Relevance:**
- Major sand fairways (potential reservoirs)
- Heterogeneous facies distribution
- Compartmentalization risks

---

## 🤖 Machine Learning Principles

### Semantic Segmentation Framework

This project addresses **dense prediction** problem:
- **Input**: Seismic image (200×200 pixels)
- **Output**: Per-pixel facies classification (200×200 labels)
- **Goal**: Assign each pixel to one of 6 facies classes

### Model 1: DeepLabv3+

#### Architectural Innovations

##### 1. Atrous (Dilated) Convolutions
**Mathematical Formulation:**
```
y[j] = Σ x[j + r·k] · w[k]
```
Where:
- r = dilation rate (atrous rate)
- k = kernel position
- w = filter weights

**Advantages:**
- Exponentially increases receptive field without increasing parameters
- For rate r=2, effective kernel size 3×3 becomes 5×5
- Captures multi-scale context efficiently

**Seismic Application:**
- Captures both thin beds (r=1) and regional trends (r=6,12,18)
- Important for detecting features at multiple scales

##### 2. Atrous Spatial Pyramid Pooling (ASPP)

Parallel branches with different atrous rates:
```
ASPP = [Conv1×1, AtrousConv(r=6), AtrousConv(r=12), AtrousConv(r=18), GlobalPooling]
```

**Purpose:**
- Encode multi-scale contextual information
- Handle objects (facies) at different scales
- Combine local and global features

**Geological Relevance:**
- Thin beds require small receptive fields
- Regional geology requires large receptive fields
- ASPP captures both simultaneously

##### 3. Encoder-Decoder with Low-Level Features

**Encoder Path:**
- Modified Xception backbone
- Extracts high-level semantic features
- Downsamples by factor of 16

**Decoder Path:**
- Upsamples encoder features by 4×
- Fuses with low-level features (early encoder layers)
- Final 4× upsample to original resolution

**Why This Matters:**
- High-level features: "This looks like a canyon system"
- Low-level features: "Sharp boundary at this exact location"
- Fusion: Semantic understanding + precise localization

##### 4. Depthwise Separable Convolutions

Standard convolution:
```
Parameters = K × K × C_in × C_out
```

Depthwise separable:
```
Depthwise: K × K × C_in (spatial filtering)
Pointwise: 1 × 1 × C_in × C_out (channel mixing)
Total parameters ≈ K × K × C_in + C_in × C_out
```

**Efficiency Gain:**
- ~8-9× fewer parameters for 3×3 convolutions
- Faster training and inference
- Enables deeper networks

#### Training Characteristics

**Loss Function:**
```python
L = CrossEntropy(predictions, targets)
  = -Σ y_true · log(y_pred)
```

**Optimizer: Adam**
- Adaptive learning rates per parameter
- Momentum: β₁ = 0.9
- RMSprop: β₂ = 0.999
- Learning rate: 1e-4 (empirically chosen)

**Why DeepLabv3+ Works Well:**
1. **Sharp boundaries**: Encoder-decoder preserves spatial information
2. **Multi-scale**: ASPP handles features from thin beds to regional structures
3. **Efficient**: Separable convolutions enable deep architecture

---

### Model 2: GAN-based Segmentation

#### Generative Adversarial Network Framework

**Minimax Game:**
```
min_G max_D V(D,G) = E[log D(x,y)] + E[log(1 - D(x,G(x)))]
```

Where:
- G = Generator (U-Net segmentation network)
- D = Discriminator (PatchGAN)
- x = seismic input
- y = true facies labels

#### Generator: U-Net Architecture

**Contracting Path (Encoder):**
```
Input (200×200×1)
  ↓ Conv→BN→ReLU→Conv→BN→ReLU
64×64 features (100×100)
  ↓ MaxPool
128×128 features (50×50)
  ↓ MaxPool
256×256 features (25×25)
  ↓ MaxPool
512×512 features (12×12)
```

**Expanding Path (Decoder):**
```
Upsample + Skip Connection
512 features (25×25)
  ↓
256 features (50×50)
  ↓
128 features (100×100)
  ↓
64 features (200×200)
  ↓
6 class predictions (200×200)
```

**Skip Connections:**
- Concatenate encoder features with decoder features
- Preserve spatial information lost during downsampling
- Enable precise localization

**Why U-Net for Seismic:**
- Preserves spatial relationships in geological data
- Skip connections maintain stratigraphic continuity
- Symmetric architecture matches seismic data characteristics

#### Discriminator: PatchGAN

**Architecture:**
```
Input: [Seismic (1 channel) + Label Map (6 channels one-hot)] = 7 channels

Conv(64, stride=2) → 100×100
  ↓ LeakyReLU
Conv(128, stride=2) → 50×50
  ↓ BN→LeakyReLU
Conv(256, stride=2) → 25×25
  ↓ BN→LeakyReLU
Conv(512, stride=1) → 24×24
  ↓ BN→LeakyReLU
Conv(1, stride=1) → 23×23
  ↓
Per-patch real/fake scores
```

**PatchGAN Philosophy:**
- Instead of global real/fake: evaluate local patches
- Each 23×23 output corresponds to 70×70 input receptive field
- Enforces local spatial consistency

**Why PatchGAN for Geology:**
- Geological patterns are locally coherent
- Forces realistic facies transitions
- Captures spatial continuity better than global discriminator

#### Combined Loss Function

**Total Generator Loss:**
```
L_total = L_MCE + λ · L_adversarial

L_MCE = -Σ Σ y_true[i,j] · log(y_pred[i,j])
       i,j

L_adversarial = -log(D(x, G(x)))
```

Where λ = 0.1 (adversarial weight)

**Two-Stage Training:**

1. **Discriminator Update:**
```python
L_D = -[log D(x, y_true) + log(1 - D(x, G(x)))]
```
- Train D to distinguish real vs. generated facies

2. **Generator Update:**
```python
L_G = L_MCE - λ·log(D(x, G(x)))
```
- Train G to: (1) match true labels, (2) fool discriminator

**Why This Combination Works:**
- MCE ensures correct class prediction
- Adversarial loss enforces realistic spatial patterns
- Result: Geologically plausible facies distributions

#### Comparison: DeepLabv3+ vs GAN

| Aspect | DeepLabv3+ | GAN |
|--------|------------|-----|
| **Boundaries** | Sharp, precise | Smooth, natural |
| **Continuity** | Can be fragmented | Better spatial coherence |
| **Training** | Stable, faster | More complex, slower |
| **Physics** | Learns feature hierarchies | Learns data distribution |
| **Best For** | Precise delineation | Natural-looking predictions |

---

## 📊 Data Description

### Parihaka 3D Seismic Volume

**Survey Specifications:**
```
Survey Name: Parihaka 3D
Location: Taranaki Basin, New Zealand
Provider: New Zealand Crown Minerals
Processing: Pre-stack depth migration
```

**Volume Dimensions:**
```
Inlines:     590 traces
Crosslines:  782 traces  
Samples:     1006 time samples
Sample rate: 4 ms
Total size:  1006 × 590 × 782 ≈ 464 million samples
```

**Data Characteristics:**
- **Amplitude range**: Normalized [-1, 1] or raw amplitude values
- **Frequency content**: Dominant frequency 25-40 Hz
- **Vertical resolution**: ~10-15 meters (λ/4 at 2000 m/s velocity)
- **Signal quality**: High SNR in most areas, degraded near basement

### Training Labels

**Label Source:**
- Manually interpreted by Chevron U.S.A. Inc. geoscientists
- Based on seismic facies analysis methodology (Sheriff, 1976)
- Quality controlled with well log data and regional geology

**Label Format:**
```python
Labels: numpy array (N, 200, 200)
Values: Integer [0, 1, 2, 3, 4, 5]
Encoding:
    0 = Basement rocks
    1 = Slope mudstone A
    2 = Mass-transport complex
    3 = Slope mudstone B
    4 = Slope valley
    5 = Submarine canyon
```

**Training Data Split:**
```
Total 3D volume: 1006 × 590 × 782
Patch extraction: 200×200 non-overlapping patches

Training volume patches:   27,648 (as per paper)
Validation volume patches: 32,952 (remaining from volume)
Test volume (adjacent):    24,560 (independent volume)
```

**Class Distribution (Expected):**
```
Basement rocks:           ~15% (regional, extensive)
Slope mudstone A:         ~30% (dominant background)
Mass-transport complex:   ~8%  (less frequent)
Slope mudstone B:         ~25% (common)
Slope valley:             ~7%  (rare, linear features)
Submarine canyon:         ~15% (moderate, large features)
```

**Data Preprocessing:**
```python
1. Extract 200×200 patches from 3D volume
2. Normalize seismic amplitudes:
   x_norm = (x - mean) / std
3. Convert labels to integer class indices [0-5]
4. No data augmentation (per paper methodology)
```

### Facies Identification Criteria

**Seismic Interpretation Workflow:**

1. **Regional Analysis**: Identify major stratigraphic surfaces
2. **Attribute Analysis**: Compute amplitude, coherence, curvature
3. **Pattern Recognition**: Identify characteristic seismic signatures
4. **Well Calibration**: Tie facies to well logs where available
5. **Quality Control**: Geological consistency checks

**Facies Decision Tree:**
```
IF (amplitude = low AND continuity = low) → Basement
ELSE IF (amplitude = high AND parallel reflectors) → Slope mudstone
    IF (scour surfaces present) → Slope mudstone B
    ELSE → Slope mudstone A
ELSE IF (chaotic reflectors) → Mass-transport complex
ELSE IF (incised low-relief channels) → Slope valley
ELSE IF (complex low-amplitude channels) → Submarine canyon
```

---

## 🏗️ Model Architecture Details

### DeepLabv3+ Implementation

**Input Processing:**
```python
Input: (Batch, 1, 200, 200) - Grayscale seismic
  ↓ Entry Flow
Conv 3×3, stride=2 → (Batch, 32, 100, 100)
Conv 3×3, stride=1 → (Batch, 64, 100, 100) [low-level features]
  ↓ Xception Blocks
Block1 (stride=2) → (Batch, 128, 50, 50)
Block2 (stride=2) → (Batch, 256, 25, 25)
Block3 (stride=2) → (Batch, 728, 12, 12)
  ↓ Middle Flow (4× repetition)
Block4-7 (stride=1, dilation=1) → (Batch, 728, 12, 12)
  ↓ Exit Flow
Block8 (stride=1, dilation=2) → (Batch, 1024, 12, 12)
Block9 (stride=1, dilation=2) → (Batch, 1536, 12, 12)
Block10 (stride=1, dilation=2) → (Batch, 2048, 12, 12)
```

**ASPP Module:**
```python
Branch 1: Conv 1×1 → 256 channels
Branch 2: AtrousConv 3×3 (rate=6) → 256 channels
Branch 3: AtrousConv 3×3 (rate=12) → 256 channels
Branch 4: AtrousConv 3×3 (rate=18) → 256 channels
Branch 5: GlobalAvgPool → 256 channels
  ↓ Concatenate
(Batch, 1280, 12, 12)
  ↓ Project
Conv 1×1 → (Batch, 256, 12, 12)
```

**Decoder:**
```python
ASPP output (Batch, 256, 12, 12)
  ↓ Upsample 4×
(Batch, 256, 50, 50)
  ↓ Process low-level features
Conv 1×1 on low-level → (Batch, 48, 50, 50)
  ↓ Concatenate
(Batch, 304, 50, 50)
  ↓ Separable Convolutions
SepConv 3×3 → (Batch, 256, 50, 50)
SepConv 3×3 → (Batch, 256, 50, 50)
  ↓ Upsample 4×
(Batch, 256, 200, 200)
  ↓ Classifier
Conv 1×1 → (Batch, 6, 200, 200)
```

**Total Parameters: 45,431,582**

### GAN Implementation

**Generator (U-Net) Parameters: 31,037,766**

**Discriminator (PatchGAN) Parameters: 2,769,729**

**Effective Receptive Field:**
- Each discriminator output patch sees 70×70 input region
- Enforces local coherence over ~14 seismic traces

---

## 🎓 Training Methodology

### Hyperparameters (From Paper)

```python
TRAINING_CONFIG = {
    # Data
    'batch_size': 32,              # Paper specified
    'patch_size': 200,             # Paper specified
    'num_classes': 6,              # Paper specified
    
    # Training
    'num_epochs': 60,              # Paper specified for GAN
    'optimizer': 'Adam',           # Paper specified
    'learning_rate': 1e-4,         # Default (not specified)
    'weight_decay': 1e-4,          # L2 regularization
    'beta1': 0.5,                  # GAN momentum (standard)
    'beta2': 0.999,                # GAN RMSprop (standard)
    
    # GAN specific
    'lambda_adversarial': 0.1,     # Adversarial loss weight
    'discriminator_updates': 1,     # D updates per G update
    
    # Scheduler
    'scheduler': 'ReduceLROnPlateau',
    'patience': 5,
    'factor': 0.5,
    
    # Uncertainty
    'mc_dropout': 0.5,             # Dropout probability
    'mc_samples': 20,              # Monte Carlo samples
}
```

### Training Strategy

**DeepLabv3+ Training:**
1. Initialize with random weights (no ImageNet pretraining due to domain difference)
2. Cross-entropy loss on all pixels
3. Adam optimizer with learning rate decay
4. Validation every epoch
5. Save best model based on validation F1 score

**GAN Training:**
1. Alternating optimization:
   ```
   for epoch in epochs:
       for batch in data:
           # Update Discriminator
           loss_D = real_loss + fake_loss
           D.backward()
           
           # Update Generator  
           loss_G = MCE_loss + adversarial_loss
           G.backward()
   ```
2. Balance discriminator and generator strength
3. Monitor both losses for training stability

### Uncertainty Quantification

**Monte Carlo Dropout (Bayesian Deep Learning):**

**Theoretical Foundation:**
- Neural network with dropout ≈ Bayesian approximation
- Dropout at test time ≈ sampling from posterior distribution
- Variance of predictions ≈ epistemic uncertainty

**Implementation:**
```python
def predict_with_uncertainty(model, x, num_samples=20):
    predictions = []
    for _ in range(num_samples):
        # Enable dropout during inference
        pred = model(x, use_dropout=True)
        predictions.append(pred)
    
    # Mean prediction
    mean_pred = mode(predictions)
    
    # Uncertainty (variance)
    uncertainty = variance(predictions)
    
    return mean_pred, uncertainty
```

**Interpretation:**
- **High uncertainty**: Model unsure (often at boundaries)
- **Low uncertainty**: Model confident (within homogeneous facies)

**Geological Application:**
- Identify areas needing expert review
- Flag potential interpretation errors
- Guide additional data acquisition

---

## 📖 References

**Core Paper:**
Kaur, H., Pham, N., Fomel, S., Geng, Z., Decker, L., Gremillion, B., Jervis, M., Abma, R., & Gao, S. (2023). A deep learning framework for seismic facies classification. *Interpretation*, 11(1), T107-T116.

**Seismic Facies Methodology:**
Sheriff, R. E. (1976). Inferring stratigraphy from seismic data. *AAPG Bulletin*, 60(4), 528-542.

**DeepLab Architecture:**
Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. *ECCV*.

**GAN for Segmentation:**
Luc, P., Couprie, C., Chintala, S., & Verbeek, J. (2016). Semantic segmentation using adversarial networks. *arXiv:1611.08408*.

**Bayesian Deep Learning:**
Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*.

---

## 💡 Key Insights

### Why Deep Learning for Seismic Facies?

1. **Complexity**: Traditional methods can't capture subtle amplitude, frequency, and phase patterns simultaneously
2. **Scale**: Manual interpretation of 100s of km² is prohibitively time-consuming
3. **Consistency**: Eliminates interpreter bias and subjectivity
4. **Multi-scale**: Deep networks naturally handle features from meters to kilometers

### Why Two Models?

**DeepLabv3+ Strengths:**
- Precise boundary detection (important for reservoir boundaries)
- Multi-scale feature extraction (handles thin beds to regional structures)
- Sharp, clear predictions

**GAN Strengths:**
- Natural-looking geological patterns
- Better spatial continuity (mimics real geology)
- Smooth transitions between facies

**Best Practice:**
Use both models and combine their strengths through ensemble or expert review.

---

*This documentation provides the theoretical foundation for understanding the seismic facies classification implementation.*
