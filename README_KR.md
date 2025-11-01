# 딥러닝을 활용한 지진파 암상 분류

PyTorch 구현:

**"A deep learning framework for seismic facies classification"**  
*Harpreet Kaur, Nam Pham, Sergey Fomel, et al.*  
*Interpretation, Vol. 11, No. 1 (February 2023)*  
**DOI**: 10.1190/INT-2022-0048.1

---

## 🌊 서론: 왜 탄성파 탐사 자료만으로 암상을 분류하는가?

### 1. 지질학적 배경과 문제 정의

> **출처**: Kaur et al. (2023) - Introduction section

석유 탐사와 지질 연구에서 **암상(seismic facies) 분류**는 지하의 암석 종류와 퇴적 환경을 이해하는 핵심 과정입니다. 

**논문에서의 정의** (Sheriff, 1976):
> "Seismic facies can be described as sedimentary units that can be distinguished from one another on the basis of different seismic characteristics such as seismic amplitude, wavelet frequency, and the geometry and continuity of reflectors."

전통적인 암상 분류의 문제점 (논문에서 명시):
- **수동 해석의 한계**: "For large 3D data sets, manual interpretation of seismic facies becomes labor-intensive and time-consuming."
- **주관성 문제**: "manual interpretation is subjective, relying on the interpreter's experience and skill."

---

### 2. 지구물리학적 원리: 탄성파 반사법

#### 2.1 음향 임피던스와 반사 계수

**음향 임피던스 (Acoustic Impedance)**:

```
Z = ρ × V
```

여기서:
- Z = 음향 임피던스 (단위: kg/m²·s 또는 g/cm²·s)
- ρ = 암석 밀도 (kg/m³)
- V = 탄성파 속도 (m/s)

> **출처**: Sheriff, R.E. (1976), "Inferring stratigraphy from seismic data", AAPG Bulletin, 60(4), 528-542.

**반사 계수 (Reflection Coefficient)**:

```
R = (Z₂ - Z₁) / (Z₂ + Z₁)
```

**주의**: 위의 암석 물성값(밀도, 속도)은 일반적인 지구물리학 교과서의 평균값이며, 실제 Parihaka Basin의 값과는 다를 수 있습니다.

> **참고 문헌**: 
> - Mavko, G., Mukerji, T., & Dvorkin, J. (2009). "The Rock Physics Handbook", Cambridge University Press.
> - 일반적인 암석 물성값은 지역과 깊이에 따라 크게 변할 수 있습니다.

---

#### 2.2 지진파 속성 (논문에서 명시)

> **출처**: Kaur et al. (2023) - Introduction

논문에서 언급된 지진파 특성:
- **진폭 (seismic amplitude)**
- **주파수 (wavelet frequency)**
- **반사면의 기하학과 연속성 (geometry and continuity of reflectors)**

---

### 3. 지질학적 배경: Parihaka Basin

> **출처**: Kaur et al. (2023) - Numerical examples section

**데이터 제공**:
- 3D seismic volume: New Zealand Crown Minerals
- Training labels: Chevron U.S.A. Inc.
- Location: Parihaka Basin, New Zealand

**데이터 규격** (논문 명시):
```
Survey dimensions: 1006 × 590 × 782
- 590 inlines
- 782 crosslines
- 1006 time samples
Total patches: 59,600 (200×200 size)
Training patches: 27,648
```

---

### 4. 6가지 암상 클래스 (논문 Table 1 기반)

> **출처**: Kaur et al. (2023), Table 1 - "List of density, elastic properties, and fractions of rock minerals used in this study (details provided by Chevron)"

#### **Class 0: Basement rocks**

**논문에서의 설명**:
- "Low signal-to-noise ratio; few internal reflectors"
- "May contain volcanic deposits in places"

**특징**:
- 신호대잡음비가 낮음
- 내부 반사면이 적음
- 화산암 퇴적물 포함 가능

---

#### **Class 1: Slope mudstone A**

**논문에서의 설명**:
- "High-amplitude upper and lower boundaries"
- "Low-amplitude continuous/semicontinuous internal reflectors"

**특징**:
- 상하부 경계: 고진폭
- 내부 반사면: 저진폭, 연속적/반연속적

---

#### **Class 2: Mass-transport complex**

**논문에서의 설명**:
- "Mix of chaotic facies and low-amplitude parallel reflectors"

**특징**:
- 혼돈스러운 암상과 저진폭 평행 반사면의 혼합
- 해저 사태 및 debris flow로 형성

> **참고 문헌**: 
> - Posamentier, H.W. & Martinsen, O.J. (2011). "The character and genesis of submarine mass-transport deposits: insights from outcrop and 3D seismic data", SEPM Special Publication, 96, 7-38.

---

#### **Class 3: Slope mudstone B**

**논문에서의 설명**:
- "High-amplitude parallel reflectors"
- "Low-continuity scour surfaces"

**특징**:
- 고진폭 평행 반사면
- 저연속성 scour 표면 (침식 흔적)

---

#### **Class 4: Slope valley**

**논문에서의 설명**:
- "High-amplitude incised channels"
- "Relatively low relief"

**특징**:
- 고진폭 침식 채널
- 상대적으로 낮은 기복

---

#### **Class 5: Submarine canyon**

**논문에서의 설명**:
- "Low-amplitude mix of parallel surfaces and chaotic reflectors"

**특징**:
- 저진폭
- 평행 표면과 혼돈 반사면의 혼합

---

## 🤖 딥러닝 방법론

### 1. DeepLabv3+ 아키텍처

> **출처**: Chen, L.C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). "Encoder-decoder with atrous separable convolution for semantic image segmentation", ECCV.

**핵심 구성요소** (논문 명시):

1. **Atrous Convolution** (팽창 합성곱)
   - "Atrous convolution refers to convolution with upsampled filters"
   - "generalizes the standard convolution operation"
   
2. **ASPP (Atrous Spatial Pyramid Pooling)**
   - "uses multiple parallel atrous convolution layers with different sampling rates"
   - Rates: 6, 12, 18 (논문 구현)

3. **Modified Xception backbone**
   - "all of the max-pooling operations are replaced by atrous separable convolutions"

4. **Encoder-Decoder structure**
   - "increases computational efficiency"
   - "refines results for semantic segmentation, especially along the object boundaries"

---

### 2. GAN 기반 세그먼테이션

> **출처**: 
> - Goodfellow, I.J., et al. (2014). "Generative adversarial networks", arXiv:1406.2661
> - Luc, P., et al. (2016). "Semantic segmentation using adversarial networks", arXiv:1611.08408

**손실 함수** (논문 Equation 3):

```
L = Σ l_mce(s(x_n), y_n) - λ(l_bce(a(x_n, y_n), 1) + l_bce(a(x_n, s(x_n)), 0))
```

여기서:
- l_mce: Multiclass cross-entropy loss
- l_bce: Binary cross-entropy loss  
- λ: Adversarial loss weight
- s(x): Segmentation model output
- a(x, y): Adversarial network output

**훈련 방법** (논문 명시):
- "We train the network using the Adam optimizer with 60 epochs and a batch size of 32"

---

### 3. 불확실성 추정

> **출처**: 
> - Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning", ICML.
> - Kendall, A. & Gal, Y. (2017). "What uncertainties do we need in bayesian deep learning for computer vision?", NIPS.

**방법** (논문 명시):
- "we use dropout at the inference time and compute multiple predictions for each pixel"
- "The use of dropout layers in neural networks is equivalent to Bayesian approximation"

**불확실성 유형**:

1. **Epistemic uncertainty** (인식론적 불확실성)
   - 모델 매개변수의 불확실성
   - "can be reduced by incorporating more training data"

2. **Aleatoric uncertainty** (우연적 불확실성)
   - 데이터 자체의 노이즈
   - "is irreducible"

---

## 📊 데이터셋 상세

### Parihaka 3D Volume

> **출처**: Kaur et al. (2023) - Numerical examples section

**정확한 데이터 규격** (논문에서 명시):

```
3D seismic migrated volume: 1006 × 590 × 782
- Inlines: 590 traces
- Crosslines: 782 traces
- Samples: 1006 time samples

Patch extraction: 200 × 200 samples (inline × crossline)
Total patches: 59,600

Training volume patches: 27,648 (논문 명시)
Validation volume patches: 32,952 (remaining from volume)
Test volume patches: 24,560 (from adjacent volume, dimensions: 782 × 251)
```

**데이터 출처**:
- Survey: Parihaka 3D, New Zealand
- Provider: New Zealand Crown Minerals
- Labels: Chevron U.S.A. Inc.

---

## 🔧 구현 세부사항

### 하이퍼파라미터 (논문 기반)

**논문에서 명시된 값**:

```python
# 명시적으로 기재된 파라미터
batch_size = 32          # "batch size of 32" (명시)
num_epochs = 60          # "60 epochs" for GAN (명시)
optimizer = "Adam"       # "Adam optimizer" (명시)
patch_size = 200         # "200 × 200 samples" (명시)
num_classes = 6          # 6 facies types (명시)

# 논문에 명시되지 않은 파라미터 (구현 시 선택)
learning_rate = 1e-4     # 일반적인 Adam 기본값
mc_samples = 20          # 불확실성 추정용 MC 샘플 (일반적 값)
```

**주의사항**:
- Learning rate는 논문에 명시되지 않음
- MC dropout samples 수는 논문에 명시되지 않음
- 이들은 딥러닝 문헌의 일반적 값을 사용

---

## 📈 결과 (논문 기반)

### 성능 메트릭

> **출처**: Kaur et al. (2023) - Performance metric section

**수식** (논문 Equations 4, 5, 6):

```
Precision = TruePositive / (TruePositive + FalsePositive)

Recall = TruePositive / (TruePositive + FalseNegative)

F1 score = 2 × (Precision × Recall) / (Precision + Recall)
```

---

### 주요 발견 (논문 기반)

**DeepLabv3+ 특징**:
> "DeepLab v3+ output captures sharper boundaries between the facies by gradually capturing spatial information using ASPP"

**GAN 특징**:
> "GAN output shows improved continuity of predicted facies"

**비교 분석** (논문 Figure 3, 4, 5):
- "DeepLab v3+ has picked up sharper facies boundaries"
- "the facies boundaries picked by GAN are smooth"
- "the continuity of predicted facies is better preserved by GAN"

**권장사항** (논문 결론):
> "The joint analysis of the output of multiple networks provides a more accurate interpretation of predicted facies"

---

## 🔍 불확실성 분석 결과

> **출처**: Kaur et al. (2023) - Figure 8

논문에서 관찰된 불확실성 패턴:
- "uncertainty values are overall low except at the boundaries of the facies"
- "These regions correspond to the mispicked facies types"
- "the mispicked regions correspond to the areas where seismic amplitudes change"

---

## 📚 참고문헌

### 주 논문
Kaur, H., Pham, N., Fomel, S., Geng, Z., Decker, L., Gremillion, B., Jervis, M., Abma, R., & Gao, S. (2023). A deep learning framework for seismic facies classification. *Interpretation*, 11(1), T107-T116. doi:10.1190/INT-2022-0048.1

### 지진파 암상 방법론
Sheriff, R.E. (1976). Inferring stratigraphy from seismic data. *AAPG Bulletin*, 60(4), 528-542. doi:10.1306/83D923F7-16C7-11D7-8645000102C1865D

### DeepLabv3+ 아키텍처
Chen, L.C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. *Proceedings of the European Conference on Computer Vision (ECCV)*, 801-818.

Chen, L.C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A.L. (2017). DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40, 834-848. doi:10.1109/TPAMI.2017.2699184

### GAN 세그먼테이션
Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial networks. *arXiv preprint*, arXiv:1406.2661.

Luc, P., Couprie, C., Chintala, S., & Verbeek, J. (2016). Semantic segmentation using adversarial networks. *arXiv preprint*, arXiv:1611.08408.

### Bayesian 딥러닝
Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *Proceedings of the 33rd International Conference on Machine Learning*, 1050-1059.

Kendall, A. & Gal, Y. (2017). What uncertainties do we need in bayesian deep learning for computer vision? *Advances in Neural Information Processing Systems*, 30.

### 추가 참고 (본문에서 인용)
Wrona, T., Pan, I., Gawthorpe, R.L., & Fossen, H. (2018). Seismic facies analysis using machine learning. *Geophysics*, 83(5), O83-O95. doi:10.1190/geo2017-0595.1

Zhao, T. (2018). Seismic facies classification using different deep convolutional neural networks. *88th Annual International Meeting, SEG, Expanded Abstracts*, 2046-2050. doi:10.1190/segam2018-2997085.1

Alaudah, Y., Michałowicz, P., Alfarraj, M., & AlRegib, G. (2019). A machine-learning benchmark for facies classification. *Interpretation*, 7(3), SE175-SE187. doi:10.1190/INT-2018-0249.1

---

## ⚠️ 중요 면책사항

### 검증된 정보
이 문서의 다음 내용은 **논문에서 직접 인용**되었습니다:
- ✅ 데이터셋 규격 (볼륨 크기, 패치 수)
- ✅ 6가지 암상 클래스 설명 (Table 1)
- ✅ 훈련 파라미터 (batch size, epochs)
- ✅ 모델 아키텍처 설명
- ✅ 성능 메트릭 수식

### 추론된 정보
다음 내용은 **일반적인 지구물리학 지식**에서 가져왔으며 논문에 명시되지 않았습니다:
- ⚠️ 구체적인 암석 물성값 (밀도, 속도)
- ⚠️ 음향 임피던스 계산 예시
- ⚠️ 해상도 한계의 구체적 수치
- ⚠️ Learning rate (1e-4)
- ⚠️ MC samples 수 (20)

### 데이터 사용 권한
- Parihaka 3D 데이터: New Zealand Crown Minerals 소유
- Training labels: Chevron U.S.A. Inc. 제공
- 상업적 사용 시 해당 기관의 허가 필요

---

## 💻 사용법

### 설치

```bash
git clone https://github.com/knocgp/A-deep-learning-framework-for-seismic-facies-classification---realize.git
cd A-deep-learning-framework-for-seismic-facies-classification---realize

pip install -r requirements.txt
```

### 빠른 시작

```bash
# Jupyter 노트북 실행
jupyter notebook main.ipynb
```

노트북에는 다음이 포함됩니다:
1. 데이터 로딩 및 시각화
2. 모델 학습 (DeepLabv3+ 및 GAN)
3. 평가 및 비교
4. 불확실성 분석

---

## 🎯 주요 기능

- ✅ 논문의 DeepLabv3+ 구현
- ✅ GAN 기반 세그먼테이션 구현
- ✅ Bayesian 불확실성 추정
- ✅ 성능 평가 메트릭
- ✅ 시각화 도구
- ✅ 전체 workflow 노트북

---

## 📧 문의

질문이나 이슈가 있으시면:
1. 논문을 먼저 참조하세요
2. GitHub 이슈를 등록하세요
3. 관련 참고문헌을 확인하세요

---

## 📄 라이선스

이 구현은 연구 및 교육 목적입니다. 

**데이터 사용**:
- Parihaka 3D: New Zealand Crown Minerals 정책 준수
- Training labels: Chevron U.S.A. Inc. 제공, 사용 권한 확인 필요

---

**GitHub**: https://github.com/knocgp/A-deep-learning-framework-for-seismic-facies-classification---realize

**최종 업데이트**: 2025년 11월 1일
