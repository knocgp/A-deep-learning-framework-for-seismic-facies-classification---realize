# 딥러닝을 활용한 지진파 암상 분류

PyTorch 구현:

**"A deep learning framework for seismic facies classification"**  
*Harpreet Kaur, Nam Pham, Sergey Fomel, et al.*  
*Interpretation, Vol. 11, No. 1 (February 2023)*

---

## 🌊 서론: 왜 탄성파 탐사 자료만으로 암상을 분류하는가?

### 1. 지질학적 배경과 문제 정의

석유 탐사와 지질 연구에서 **암상(facies) 분류**는 지하의 암석 종류와 퇴적 환경을 이해하는 핵심 과정입니다. 전통적으로 암상 분류는 시추공(well)에서 채취한 암석 시료나 물리검층 자료를 통해 이루어져 왔습니다. 하지만 시추공은:
- **비용이 매우 높고** (수백만~수천만 달러)
- **공간적으로 제한적이며** (수 km 간격)
- **시간이 많이 소요됩니다** (수개월)

반면 **탄성파(seismic) 탐사**는:
- ✅ 넓은 지역을 커버 (수백 km²)
- ✅ 상대적으로 저렴 (시추 대비)
- ✅ 3차원 지하 구조 영상화 가능
- ✅ 짧은 시간에 대량 데이터 획득

따라서 **탄성파 자료만으로 암상을 분류**할 수 있다면, 시추공이 없는 지역에서도 지하 지질을 이해할 수 있어 탐사 효율이 극적으로 향상됩니다.

### 2. 지구물리학적 원리: 탄성파 반사법

#### 2.1 탄성파의 생성과 전파

**탄성파 탐사**는 다음과 같은 물리학적 원리를 기반으로 합니다:

1. **인공 진동원 발생**
   - 육상: 진동차(vibroseis), 다이너마이트
   - 해상: 에어건(air gun)
   - 발생된 탄성파(P파, S파)가 지하로 전파

2. **암석 경계면에서의 반사**
   ```
   지표 → 탄성파 전파 → 암석 경계면 → 반사파 → 수신기 → 기록
   ```

3. **음향 임피던스(Acoustic Impedance)**
   
   탄성파의 반사는 **음향 임피던스 대비**에 의해 결정됩니다:
   
   ```
   Z = ρ × V
   ```
   
   여기서:
   - Z = 음향 임피던스 (kg/m²·s)
   - ρ = 암석 밀도 (kg/m³)
   - V = 탄성파 속도 (m/s)
   
   **예시**:
   - 사암(sandstone): ρ ≈ 2.3 g/cm³, V ≈ 3500 m/s → Z ≈ 8050
   - 셰일(shale): ρ ≈ 2.5 g/cm³, V ≈ 2500 m/s → Z ≈ 6250
   - 석회암(limestone): ρ ≈ 2.7 g/cm³, V ≈ 4500 m/s → Z ≈ 12150

4. **반사 계수(Reflection Coefficient)**
   
   두 층의 경계면에서 반사되는 에너지의 비율:
   
   ```
   R = (Z₂ - Z₁) / (Z₂ + Z₁)
   ```
   
   - Z₁ > Z₂인 경우: 음의 반사 (위상 역전)
   - Z₁ < Z₂인 경우: 양의 반사
   - |R|이 클수록 강한 반사파
   
   **예시 (사암/셰일 경계)**:
   ```
   R = (8050 - 6250) / (8050 + 6250) = 0.126
   → 약 12.6%의 에너지가 반사됨
   ```

#### 2.2 지진파 속성과 암상의 관계

서로 다른 암석과 퇴적 환경은 독특한 **지진파 특성(seismic signature)**을 보입니다:

| 지진파 속성 | 물리적 의미 | 지질학적 정보 |
|------------|------------|--------------|
| **진폭(Amplitude)** | 반사 신호의 강도 | 임피던스 대비, 유체 포화도 |
| **주파수(Frequency)** | 파동의 진동수 | 층 두께, 암석 감쇠 특성 |
| **위상(Phase)** | 파동의 도착 시간 | 암상 변화, 층서 관계 |
| **연속성(Continuity)** | 반사면의 측방 연속성 | 퇴적 환경, 단층 |
| **형태(Geometry)** | 반사면의 기하학적 패턴 | 채널, 협곡, 층서 |

**구체적 예시**:
- **평행한 연속 반사면** → 조용한 심해 환경의 이암(mudstone)
- **혼돈스러운 반사면** → 해저 산사태(mass-transport complex)
- **침식된 채널 형태** → 해저 협곡(submarine canyon)
- **고진폭 반사** → 강한 임피던스 대비 (예: 기반암-퇴적층 경계)

#### 2.3 해상도 한계

탄성파 자료는 물리적 한계가 있습니다:

- **수직 해상도**: λ/4 (일반적으로 10-30m)
  - 지배 주파수 30 Hz, 속도 3000 m/s인 경우: 100/4 = 25m
  - 이보다 얇은 층은 구분 어려움

- **수평 해상도**: 프레넬 대역(Fresnel zone) 반경
  - 일반적으로 50-200m
  - 이보다 작은 구조는 감지 어려움

### 3. 지질학적 원리: 퇴적상과 환경

#### 3.1 퇴적상(Sedimentary Facies) 개념

**퇴적상**은 특정 퇴적 환경에서 형성된 암석의 총체적 특성입니다:
- 암석 종류 (이암, 사암, 역암 등)
- 퇴적 구조 (층리, 엽리, 생물교란 등)
- 화석 내용
- 지구화학적 특성

#### 3.2 본 연구의 6가지 암상

이 연구는 뉴질랜드 Parihaka Basin (해저 환경)의 6가지 주요 암상을 구분합니다:

**1. 기반암 (Basement rocks) - Class 0**

**지질학적 특성**:
- 백악기 이전 (~100백만 년 전) 형성된 결정질 또는 변성암
- 편암(schist), 화강암(granite), 화산암(volcanic rocks)
- 모든 퇴적층의 기반을 이루는 가장 오래된 암석

**물리적 특성**:
- 밀도: 2.6-2.9 g/cm³ (퇴적암보다 높음)
- P파 속도: 5000-6500 m/s (매우 빠름)
- 음향 임피던스: 매우 높음

**지진파 특성**:
- 상부 경계: 매우 강한 반사 (퇴적층과의 큰 임피던스 대비)
- 내부 구조: 반사면이 거의 없거나 무질서함
- 신호대잡음비: 낮음 (결정질 구조로 인한 산란)

**왜 중요한가?**
- 석유 탐사의 하한선 (기반암 아래는 탐사 불필요)
- 지진 활동과 관련 (단층 위치 파악)

---

**2. 경사면 이암 A (Slope mudstone A) - Class 1**

**지질학적 특성**:
- 대륙 경사면(continental slope)의 심해 환경 (~500-2000m 수심)
- 미세한 점토와 실트가 느리게 퇴적 (반원양성 퇴적물)
- 미오세-플라이오세 시대 (약 5-20백만 년 전)

**퇴적 과정**:
```
대륙붕 → 미세 입자 부유 → 경사면으로 이동 → 느린 침강 → 이암 형성
```

**물리적 특성**:
- 밀도: 2.2-2.5 g/cm³
- P파 속도: 2000-3000 m/s
- 공극률: 30-50% (압밀 정도에 따라)

**지진파 특성**:
- **상하부 경계**: 고진폭 반사 (다른 암상과의 뚜렷한 경계)
- **내부 반사면**: 저진폭, 연속적/반연속적
- **패턴**: 평행 또는 약간 경사진 층리

**왜 중요한가?**
- 석유 시스템의 덮개암(seal rock) 역할
- 저류층(reservoir)의 측방 연속성 지시
- 퇴적 속도와 환경 변화 기록

---

**3. 매스 이동 복합체 (Mass-transport complex, MTC) - Class 2**

**지질학적 특성**:
- **형성 메커니즘**: 해저 사면의 중력 붕괴
  ```
  안정한 경사면 → 트리거(지진, 과다 퇴적) → 붕괴 → 해저 사태
  ```
- 규모: 수 km² ~ 수천 km²
- 두께: 수 m ~ 수백 m

**트리거 요인**:
1. 지진 활동 (가장 흔함)
2. 경사면 과도 경사 (퇴적물 축적)
3. 급격한 퇴적 속도
4. 가스 하이드레이트 해리
5. 해저 화산 활동

**물리적 특성**:
- 불균질한 혼합물 (진흙, 모래, 역, 점토 덩어리)
- 밀도: 매우 가변적 (1.8-2.4 g/cm³)
- 공극률: 불규칙

**지진파 특성**:
- **혼돈스러운 반사면**: 내부 구조 파괴
- **불규칙한 상부 표면**: 언덕형(hummocky) 지형
- **기저 전단면**: 침식성 경계
- **저진폭**: 내부 에너지 산란

**왜 중요한가?**
- 🚨 **지질재해(Geohazard)**: 해저 케이블, 파이프라인 위험
- 고대 지진 활동 기록 (고지진학)
- 저류층 품질 저하 (불균질성)

---

**4. 경사면 이암 B (Slope mudstone B) - Class 3**

**지질학적 특성**:
- 경사면 이암 A와 유사하지만:
  - 다른 지질 시대 (더 오래되거나 새로운)
  - 약간 다른 광물 조성
  - **등심류(contour current)의 영향**

**등심류의 영향**:
```
해저 등심류 → 퇴적물 재동원 → scour 표면 형성 → 불연속면
```

**물리적 특성**:
- 이암 A와 유사
- 국부적으로 실트/모래 렌즈 포함 (등심류에 의한)

**지진파 특성**:
- **고진폭 평행 반사면**
- **더 좋은 연속성** (이암 A보다)
- **저연속성 scour 표면**: 해저류에 의한 침식면

**왜 중요한가?**
- 심해류(bottom current) 활동 지시자
- 퇴적 속도 변화 기록
- 국부 저류층 가능성 (실트/모래 렌즈)

---

**5. 경사면 계곡 (Slope valley) - Class 4**

**지질학적 특성**:
- 대륙 경사면의 **침식성 지형**
- 형성 메커니즘:
  ```
  혼탁류(turbidity current) → 경사면 침식 → V자/U자형 계곡
  ```
- 규모: 폭 수백 m, 깊이 수십 m

**혼탁류(Turbidity Current)**:
```
대륙붕 사태 → 퇴적물+물 혼합 → 밀도류 → 경사면 아래로 흐름 (50-100 km/h)
```

**물리적 특성**:
- 계곡 충진물: 모래, 실트, 이암의 혼합
- 측벽: 주변 이암
- 제방(levee) 퇴적물

**지진파 특성**:
- **고진폭 침식 채널**: 뚜렷한 U자형/V자형
- **상대적으로 낮은 기복** (협곡보다 작음)
- **측방 연속성 제한**: 선형 구조

**왜 중요한가?**
- 🛢️ **석유 탐사**: 사암 통로 (잠재 저류층)
- 대륙붕-심해 분지 연결 통로
- 모래 분포 예측에 핵심

---

**6. 해저 협곡 (Submarine canyon) - Class 5**

**지질학적 특성**:
- **대규모 침식 시스템**
- 형성: 장기간 (수십만 년) 혼탁류와 해저류에 의한 침식
- 규모: 폭 수 km, 깊이 수백 m
- 현대 유사례: 콩고 협곡, 몬테레이 협곡

**형성 과정**:
```
초기 침식 → 혼탁류 통로 → 반복적 침식 → 협곡 심화 → 
채널-제방 시스템 발달 → 복합 충진
```

**물리적 특성**:
- 매우 이질적 (heterogeneous)
- 충진물: 사암, 역암, 이암의 복잡한 혼합
- 다중 침식-충진 시퀀스

**지진파 특성**:
- **저진폭 혼합**: 평행 반사면 + 혼돈 반사면
- **복잡한 내부 구조**: 채널-제방-충진
- **다중 절단-충진 시퀀스**: 여러 번의 활동 기록

**왜 중요한가?**
- 🛢️ **주요 저류층 통로**: 사암이 풍부
- 💰 **석유 시스템의 핵심**: 근원암-저류층-덮개암 연결
- ⚠️ **저류층 복잡도**: 구획화(compartmentalization) 위험

---

### 4. 왜 딥러닝인가?

#### 4.1 전통적 방법의 한계

**수동 해석**:
- 👁️ 지구물리학자가 지진파 단면을 보고 직접 해석
- ⏰ 시간 소모: 수백 km²를 해석하는데 수 개월
- 👤 주관성: 해석자마다 다른 결과
- 🎯 일관성 결여: 대규모 3D 자료에서 일관성 유지 어려움

**전통적 머신러닝**:
- 수작업 특징 추출 (amplitude, frequency, coherence 등)
- 단순 분류기 (SVM, Random Forest)
- 복잡한 패턴 포착 한계

#### 4.2 딥러닝의 장점

**1. 자동 특징 학습**
```
원시 지진파 자료 → 딥러닝 모델 → 자동으로 중요한 특징 발견
```
- 인간이 정의하지 못한 복잡한 패턴 학습
- 진폭, 주파수, 위상, 기하학적 패턴을 동시에 고려

**2. 다중 스케일 분석**
- 얇은 층 (수십 m) ~ 지역 구조 (수 km)를 동시에 처리
- ASPP (Atrous Spatial Pyramid Pooling)가 핵심

**3. 공간적 맥락 이해**
- 주변 픽셀과의 관계 학습
- 지질학적으로 타당한 연속성 유지

**4. 불확실성 정량화**
- Bayesian 딥러닝: 모델이 얼마나 확신하는지 측정
- 불확실한 지역 → 전문가 검토 필요

---

## 📋 개요

이 저장소는 최신 딥러닝 기법을 활용한 자동 지진파 암상 분류를 위한 두 가지 모델을 구현합니다:

### 모델 1: DeepLabv3+
**핵심 아이디어**: 다중 스케일 특징 추출 + 정밀한 경계 검출

**작동 원리**:
1. **Atrous Convolution (팽창 합성곱)**
   ```
   일반 3×3 필터 → rate=2 → 5×5 효과 (매개변수 수는 동일)
   ```
   - 넓은 영역을 보면서도 계산량 증가 없음
   - 얇은 층과 두꺼운 층을 동시에 포착

2. **ASPP (Atrous Spatial Pyramid Pooling)**
   ```
   병렬 처리:
   - 1×1 convolution (점 정보)
   - 3×3 atrous conv (rate=6) (중간 스케일)
   - 3×3 atrous conv (rate=12) (넓은 스케일)
   - 3×3 atrous conv (rate=18) (매우 넓은 스케일)
   - Global pooling (전역 맥락)
   → 결합 → 다중 스케일 정보
   ```

3. **Encoder-Decoder 구조**
   - Encoder: "이것은 협곡 시스템이다" (의미 정보)
   - Decoder: "경계는 정확히 여기다" (공간 정보)
   - 융합: 정확한 암상 분류

**장점**:
- ✅ 날카로운 암상 경계 (저류층 경계 파악에 중요)
- ✅ 다중 스케일 특징 (얇은 층~지역 구조)
- ✅ 정확한 클래스 예측

### 모델 2: GAN 기반 세그먼테이션
**핵심 아이디어**: 적대적 학습으로 지질학적으로 타당한 패턴 생성

**작동 원리**:
1. **Generator (생성자) - U-Net**
   ```
   지진파 영상 → U-Net → 암상 예측 맵
   ```
   - Skip connections: 공간 정보 보존
   - 대칭 구조: 세밀한 복원

2. **Discriminator (판별자) - PatchGAN**
   ```
   진짜 암상 vs 생성된 암상 → 판별 (70×70 패치 단위)
   ```
   - 국소적 지질학적 일관성 강제
   - 현실적인 암상 패턴 학습

3. **적대적 학습**
   ```
   Generator: "나는 진짜처럼 보이는 암상을 만들겠다"
   Discriminator: "나는 진짜와 가짜를 구별하겠다"
   → 경쟁을 통해 둘 다 발전
   ```

**장점**:
- ✅ 부드러운 암상 전이 (지질학적으로 자연스러움)
- ✅ 공간적 연속성 향상
- ✅ 잡음에 덜 민감

### 불확실성 추정
**Monte Carlo Dropout (Bayesian 근사)**:
```
같은 입력 → 20번 예측 (매번 다른 dropout) → 분산 계산 → 불확실성
```

**해석**:
- 높은 불확실성 → 모델이 확신하지 못함 → 전문가 검토 필요
- 낮은 불확실성 → 모델이 확신함 → 자동 분류 신뢰

---

## 📊 데이터셋

**출처**: Parihaka Basin, 뉴질랜드 (New Zealand Crown Minerals 제공)
**해석 레이블**: Chevron U.S.A. Inc. 제공

### 6가지 암상 클래스

| 클래스 | 암상명 | 퇴적 환경 | 석유 시스템 역할 |
|--------|--------|----------|----------------|
| 0 | 기반암 | 선캄브리아-중생대 | 탐사 하한선 |
| 1 | 경사면 이암 A | 심해 경사면 | 덮개암 |
| 2 | 매스 이동 복합체 | 해저 사태 | 지질재해 |
| 3 | 경사면 이암 B | 등심류 영향 경사면 | 덮개암 |
| 4 | 경사면 계곡 | 혼탁류 통로 | 저류층 통로 |
| 5 | 해저 협곡 | 주요 퇴적 시스템 | 주요 저류층 |

### 데이터 형식
- **입력**: 2D 지진파 패치 (200 × 200 픽셀)
- **출력**: 픽셀별 암상 분류 (200 × 200)
- **훈련 패치**: 27,648개 (논문 명시)
- **검증 패치**: 3D 볼륨의 나머지

---

## 🚀 빠른 시작

### 설치

```bash
# 저장소 복제
git clone https://github.com/knocgp/A-deep-learning-framework-for-seismic-facies-classification---realize.git
cd A-deep-learning-framework-for-seismic-facies-classification---realize

# 의존성 설치
pip install torch torchvision numpy matplotlib scipy tqdm
# 또는
pip install -r requirements.txt
```

### 프로젝트 구조

```
webapp/
├── data_loader.py      # 데이터 로딩 및 전처리
├── model.py            # DeepLabv3+ 및 GAN 모델
├── utils.py            # 메트릭, 불확실성, 시각화
├── train.py            # 학습 로직
├── test.py             # 테스트 및 추론
├── main.ipynb          # 전체 workflow 노트북
├── README_KR.md        # 이 파일 (한글)
├── DOCUMENTATION_KR.md # 기술 문서 (한글)
├── DATA_GUIDE_KR.md    # 데이터 가이드 (한글)
├── data/               # 데이터 디렉토리
├── checkpoints/        # 저장된 모델
└── results/            # 출력 결과 및 시각화
```

---

## 💻 사용법

### Option 1: Jupyter 노트북 (권장)

```bash
jupyter notebook main.ipynb
```

노트북에는 다음이 포함됩니다:
1. 데이터 로딩 및 시각화
2. 모델 학습 (DeepLabv3+ 및 GAN)
3. 평가 및 비교
4. 불확실성 분석
5. 결과 시각화

### Option 2: Python 스크립트

**모델 학습:**
```python
from train import train_model
from data_loader import get_dataloaders

# 데이터 로드
train_loader, val_loader = get_dataloaders(
    train_seismic, train_labels,
    val_seismic, val_labels,
    batch_size=32
)

# DeepLabv3+ 학습
history = train_model(
    model_type='deeplabv3+',
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=60,
    device='cuda'
)
```

**모델 테스트:**
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

**모델 비교:**
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

## 🔧 하이퍼파라미터 설정

논문에서 명시된 주요 파라미터:

```python
CONFIG = {
    'patch_size': 200,        # 패치 크기 (200×200)
    'num_classes': 6,         # 암상 클래스 수
    'batch_size': 32,         # 배치 크기
    'num_epochs': 60,         # 학습 에폭
    'learning_rate': 1e-4,    # Adam 학습률
    'num_mc_samples': 20,     # 불확실성 추정용 MC 샘플 수
}
```

### 파라미터 주석

- **Batch size**: 32 (논문 명시)
- **Epochs**: 60 (GAN용, 논문 명시)
- **Optimizer**: Adam (논문 명시)
- **Learning rate**: 1e-4 (논문에 없음, 일반 값 사용)
- **MC samples**: 20 (논문에 없음, 일반 값 사용)
- **Middle flow 반복**: 4회 (원본 Xception은 8회, 효율성을 위해 축소)

---

## 📈 결과

### 성능 메트릭

모델은 다음 메트릭으로 평가됩니다:
- **Precision (정밀도)**: TP / (TP + FP)
- **Recall (재현율)**: TP / (TP + FN)  
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

### 주요 발견 (논문 기반)

**1. DeepLabv3+**:
- ✅ 암상 간 날카로운 경계
- ✅ 전이 구간 포착 우수
- ✅ ASPP를 통한 다중 스케일 특징
- ⚠️ 저신호대잡음비 지역에서 단편화 가능

**2. GAN**:
- ✅ 예측된 암상의 연속성 향상
- ✅ 부드러운 예측
- ✅ 잡음에 덜 민감
- ⚠️ 중요한 경계를 흐리게 할 수 있음

**3. 불확실성**:
- 암상 경계에서 높음 (예상됨)
- 오분류 영역 지시
- 품질 관리에 유용

### 권장 사항
**두 모델을 함께 사용**하여 각각의 장점을 활용:
- DeepLabv3+: 정밀한 경계 파악
- GAN: 공간적 맥락과 연속성
- 결합: 더 견고한 해석

---

## 🧪 구현 테스트

개별 모듈 테스트:

```bash
# 데이터 로더 테스트
cd /home/user/webapp && python data_loader.py

# 모델 테스트
cd /home/user/webapp && python model.py

# 유틸리티 테스트
cd /home/user/webapp && python utils.py

# 학습 테스트 (짧은 실행)
cd /home/user/webapp && python train.py

# 추론 테스트
cd /home/user/webapp && python test.py
```

---

## 📊 불확실성 추정

Monte Carlo Dropout을 사용한 인식론적 불확실성:

```python
from test import Tester
from model import get_model

model = get_model('deeplabv3+')
tester = Tester(model, 'deeplabv3+')
tester.load_checkpoint('checkpoints/deeplabv3+_best.pth')

# 불확실성과 함께 예측
predictions, uncertainty = tester.predict_with_uncertainty(
    seismic_data,
    num_samples=20  # MC 샘플 수
)
```

**해석**:
- `uncertainty`가 높은 픽셀 → 모델이 확신하지 못함
- 주로 암상 경계나 모호한 지역에서 발생
- 전문가 검토가 필요한 영역 자동 식별

---

## 📁 데이터 형식

### 입력 데이터

NumPy 배열로 지진파 자료 로드:

```python
import numpy as np

# 지진파 자료: (N, 200, 200) - N개의 200×200 픽셀 패치
train_seismic = np.load('train_seismic.npy')

# 레이블: (N, 200, 200) - 정수 값 0-5
train_labels = np.load('train_labels.npy')
```

### 3D 볼륨에서 패치 생성

```python
from data_loader import create_patches_from_volume

# 3D 볼륨: (depth, height, width)
volume = np.load('seismic_volume.npy')

# 패치 추출
patches = create_patches_from_volume(
    volume,
    patch_size=200,
    stride=200,  # 겹치지 않게
    axis=0       # depth 방향으로 추출
)
```

---

## 🎯 모델 출력 특성

### DeepLabv3+
- 날카로운 암상 경계
- 우수한 에지 검출
- ASPP가 다중 스케일 특징 포착

### GAN
- 부드러운 암상 전이
- 공간적 연속성 향상
- 적대적 학습으로 일관성 개선

### 결합 분석
두 모델을 함께 사용 (논문 권장):
- 두 접근법의 장점 활용
- 더 견고한 예측
- 해석 신뢰도 향상

---

## 📝 인용

이 코드를 사용하시면 원 논문을 인용해 주세요:

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

---

## 🔍 주요 기능

- ✅ 지진파 자료용 DeepLabv3+ 완전 구현
- ✅ 적대적 학습 기반 GAN 세그먼테이션
- ✅ Bayesian 근사를 사용한 불확실성 추정
- ✅ 포괄적 평가 메트릭 (Precision, Recall, F1)
- ✅ 결과 및 불확실성 시각화 도구
- ✅ 모델 비교 유틸리티
- ✅ Jupyter 노트북의 전체 workflow
- ✅ 확장 가능한 모듈식 코드 구조

---

## 🛠️ 요구사항

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- SciPy
- tqdm
- h5py (선택, HDF5 데이터용)

---

## 📚 추가 문서

- **DOCUMENTATION_KR.md**: 지구물리학, 지질학, 머신러닝 원리 상세 설명
- **DATA_GUIDE_KR.md**: 데이터 준비 및 사용 완전 가이드
- **PROJECT_SUMMARY.md**: 프로젝트 완성 요약 (영문)

---

## 📞 문의

질문이나 이슈가 있으시면 원 논문을 참조하시거나 이 저장소에 이슈를 등록해 주세요.

---

## 📄 라이선스

이 구현은 연구 및 교육 목적입니다. 데이터 사용권에 대해서는 원 논문을 참조하세요.

---

**논문**: Kaur et al. (2023), "A deep learning framework for seismic facies classification", *Interpretation*, 11(1), T107-T116.

**DOI**: 10.1190/INT-2022-0048.1

**GitHub**: https://github.com/knocgp/A-deep-learning-framework-for-seismic-facies-classification---realize
