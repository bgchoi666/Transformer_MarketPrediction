# 🚀 Transformer 모델 사용 가이드

## 📋 목차
1. [개요](#개요)
2. [파일 구조](#파일-구조)
3. [설치 방법](#설치-방법)
4. [Transformer 모델 종류](#transformer-모델-종류)
5. [사용 방법](#사용-방법)
6. [LSTM vs Transformer 비교](#lstm-vs-transformer-비교)
7. [파라미터 튜닝 가이드](#파라미터-튜닝-가이드)
8. [문제 해결](#문제-해결)

---

## 개요

기존 LSTM 기반 시스템에 **Transformer 모델**을 추가했습니다.

### 왜 Transformer인가?

**LSTM의 한계:**
- 긴 시퀀스에서 기울기 소실 문제
- 순차적 처리로 인한 병렬화 제한
- 먼 과거 정보 손실

**Transformer의 장점:**
- Self-Attention으로 모든 시점 동시 참조
- 병렬 처리로 학습 속도 빠름
- 장기 의존성 포착에 유리
- 최신 NLP/시계열 분야에서 SOTA 성능

---

## 파일 구조

```
프로젝트/
├── transformer_models.py           # Transformer 모델 정의
├── transformer_data_processing.py  # 데이터 처리 모듈
├── transformer_systems.py          # 시스템 통합 클래스
├── mains_transformer.py           # 실행 스크립트
└── transformer_guide.md           # 이 문서
```

### 파일별 역할

1. **transformer_models.py**
   - VanillaTransformer
   - AutoEncoderTransformer
   - HybridTransformerLSTM
   - ResTransformer

2. **transformer_data_processing.py**
   - CreateInputOutput 클래스 확장
   - Transformer용 데이터 전처리

3. **transformer_systems.py**
   - TransformerPredictor 클래스
   - HybridTransformerLSTMPredictor 클래스

4. **mains_transformer.py**
   - 실험 실행 스크립트

---

## 설치 방법

### 1. 기존 파일 업데이트

**FnLearn.py에 추가:**
```python
# transformer_data_processing.py의 메서드들을
# CreateInputOutput 클래스에 추가

@staticmethod
def transformer_input_data(data, n_timestep, future_day, stride=1):
    # ... 코드 복사

@staticmethod
def transformer_future_target(target, n_timestep, future_day, stride=1):
    # ... 코드 복사

# 나머지 메서드들도 추가
```

**models.py에 추가:**
```python
# transformer_models.py의 내용을 models.py 끝에 추가
# 또는 별도 파일로 유지하고 import

from transformer_models import VanillaTransformer, HybridTransformerLSTM
```

**systems.py에 추가:**
```python
# transformer_systems.py의 클래스들을 추가

class TransformerPredictor(MarketPrediction):
    # ... 코드 복사

class HybridTransformerLSTMPredictor(MarketPrediction):
    # ... 코드 복사
```

### 2. 새 파일 생성

또는 제공된 파일들을 그대로 사용:
```bash
# 프로젝트 폴더에 복사
cp transformer_*.py /path/to/your/project/
cp mains_transformer.py /path/to/your/project/
```

---

## Transformer 모델 종류

### 1. VanillaTransformer
**특징:**
- 순수 Transformer 아키텍처
- Multi-head Self-Attention
- Positional Encoding

**언제 사용?**
- 장기 의존성이 중요한 경우
- 데이터가 충분한 경우
- 최신 기법 실험

**파라미터:**
```python
model = VanillaTransformer(
    n_timestep=60,      # 시계열 길이
    n_inputs=300,       # 입력 특징 개수
    d_model=256,        # 모델 차원
    num_heads=8,        # 어텐션 헤드
    num_layers=4,       # Transformer 레이어
    dff=1024,          # FFN 차원
    dropout_rate=0.1   # 드롭아웃
)
```

### 2. HybridTransformerLSTM
**특징:**
- Transformer + LSTM 결합
- Transformer로 전역 패턴 포착
- LSTM으로 지역 시계열 학습

**언제 사용?**
- 두 방식의 장점 활용
- 안정적인 성능 원할 때
- 리소스 제약이 있을 때

**파라미터:**
```python
model = HybridTransformerLSTM(
    n_timestep=60,
    n_inputs=300,
    d_model=256,
    num_heads=8,
    num_transformer_layers=2,  # Transformer 레이어 (적게)
    lstm_units=256,           # LSTM 유닛
    dropout_rate=0.1
)
```

### 3. AutoEncoderTransformer
**특징:**
- AutoEncoder 구조
- 입력 압축 후 복원

**언제 사용?**
- 차원 축소 필요할 때
- 노이즈 제거 원할 때
- 기존 AutoEncoderLSTM 대체

**파라미터:**
```python
model = AutoEncoderTransformer(
    n_timestep=60,
    n_inputs=300,
    coding_size=256,          # 압축 차원
    num_heads=8,
    num_encoder_layers=3,
    num_decoder_layers=3
)
```

### 4. ResTransformer
**특징:**
- 깊은 Residual Connection
- 더 깊은 네트워크 학습

**언제 사용?**
- 복잡한 패턴 학습
- 데이터가 매우 많을 때

---

## 사용 방법

### 방법 1: 기본 사용

```python
from transformer_systems import TransformerPredictor

# 모델 초기화
predictor = TransformerPredictor(
    model_name='VanillaTransformer',
    item_name='삼성전자',
    price_column='수정주가(원)',
    target_type='trend_ratio',
    lookback_days=60,
    prediction_day=20,
    time_interval=1,
    begin_pred_date='2015-01-02',
    d_model=256,
    num_heads=8,
    num_layers=4
)

# 데이터 로드 및 전처리
predictor.data_load()
predictor.preprocessing()

# 모델 생성
predictor.create_model(dff=1024, dropout_rate=0.1, gpu=0)

# 학습
predictor.moving_transfer_learning(
    train_period=500,
    test_period=20,
    batch_size=32,
    learning_rate=0.0005,
    begin_iteration=1000,
    iteration=500
)

# 평가
predictor.evaluation(learning_time)
```

### 방법 2: 스크립트 실행

```bash
python mains_transformer.py
```

### 방법 3: 데이터만 처리

```python
from FnLearn import CreateInputOutput

# Transformer용 입력 생성
X = CreateInputOutput.transformer_input_data(
    data=dataframe.values,
    n_timestep=60,
    future_day=20,
    stride=1
)

# 타겟 생성
y = CreateInputOutput.transformer_future_target(
    target=ratio,
    n_timestep=60,
    future_day=20,
    stride=1
)

# 또는 원스톱
X, y = CreateInputOutput.create_transformer_dataset(
    data=dataframe.values,
    target=ratio,
    n_timestep=60,
    future_day=20,
    output_type='future_target'
)
```

---

## LSTM vs Transformer 비교

### 성능 비교

| 항목 | LSTM | Transformer |
|------|------|-------------|
| 장기 의존성 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 학습 속도 | ⭐⭐⭐ | ⭐⭐⭐⭐ (병렬화) |
| 메모리 사용 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 데이터 효율 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 해석 가능성 | ⭐⭐ | ⭐⭐⭐⭐ (attention) |

### 언제 무엇을 사용할까?

**LSTM 사용:**
- 데이터가 적을 때 (<10,000 샘플)
- 메모리/GPU 제한적
- 안정적인 성능 필요
- 단기 패턴 중요

**Transformer 사용:**
- 데이터가 많을 때 (>50,000 샘플)
- GPU 충분
- 최고 성능 추구
- 장기 패턴 중요

**Hybrid 사용:**
- 중간 데이터 (10,000~50,000)
- 두 장점 활용
- 실전 배포 고려

---

## 파라미터 튜닝 가이드

### 1. 모델 크기 (d_model)

```python
# 작은 모델 (빠름, 메모리 적음)
d_model = 128
num_heads = 4

# 중간 모델 (권장)
d_model = 256
num_heads = 8

# 큰 모델 (느림, 높은 성능)
d_model = 512
num_heads = 8
```

### 2. 레이어 개수 (num_layers)

```python
# 얕은 네트워크
num_layers = 2  # 빠름, 단순 패턴

# 중간
num_layers = 4  # 권장

# 깊은 네트워크
num_layers = 6  # 복잡한 패턴, 오버피팅 주의
```

### 3. 드롭아웃 (dropout_rate)

```python
# 데이터 많음
dropout_rate = 0.1

# 데이터 보통
dropout_rate = 0.2

# 데이터 적음 (오버피팅 방지)
dropout_rate = 0.3
```

### 4. 학습률 (learning_rate)

```python
# Transformer는 일반적으로 낮은 학습률 사용
learning_rate = 0.0001  # 안전
learning_rate = 0.0005  # 권장
learning_rate = 0.001   # 빠르지만 불안정 가능
```

### 5. 배치 크기 (batch_size)

```python
# GPU 메모리 부족 시
batch_size = 16

# 권장
batch_size = 32

# GPU 충분 시
batch_size = 64
```

---

## 실험 예시

### 실험 1: 기본 Transformer

```python
# 60일 데이터로 20일 후 예측
predictor = TransformerPredictor(
    model_name='VanillaTransformer',
    item_name='삼성전자',
    price_column='수정주가(원)',
    target_type='trend_ratio',
    lookback_days=60,
    prediction_day=20,
    time_interval=1,
    begin_pred_date='2015-01-02',
    d_model=256,
    num_heads=8,
    num_layers=4
)
```

### 실험 2: 장기 예측

```python
# 120일 데이터로 더 긴 패턴 포착
predictor = TransformerPredictor(
    lookback_days=120,  # 더 긴 시퀀스
    num_layers=6,       # 더 깊은 네트워크
    d_model=512,        # 더 큰 모델
    # ... 나머지 동일
)
```

### 실험 3: 하이브리드

```python
# Transformer + LSTM 결합
predictor = HybridTransformerLSTMPredictor(
    model_name='HybridTransformerLSTM',
    # ... 기본 파라미터
    d_model=256,
    num_transformer_layers=2,  # Transformer 적게
    lstm_units=256            # LSTM 추가
)
```

---

## 문제 해결

### 1. GPU 메모리 부족

**증상:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**해결:**
```python
# 모델 크기 줄이기
d_model = 128
num_layers = 2

# 배치 크기 줄이기
batch_size = 16

# 시퀀스 길이 줄이기
lookback_days = 40
```

### 2. 학습이 너무 느림

**해결:**
```python
# 레이어 줄이기
num_layers = 2

# 배치 크기 늘리기 (GPU 메모리 허용 시)
batch_size = 64

# Iteration 줄이기
begin_iteration = 500
iteration = 200
```

### 3. 오버피팅

**증상:**
- Train MSE는 낮은데 Test MSE가 높음

**해결:**
```python
# 드롭아웃 높이기
dropout_rate = 0.3

# 모델 크기 줄이기
d_model = 128
num_layers = 2

# 정규화 강화
# models.py에서 l2 regularization 값 증가
```

### 4. 성능이 LSTM보다 안 좋음

**원인:**
- 데이터 부족
- 하이퍼파라미터 튜닝 필요

**해결:**
```python
# 1. 더 많은 데이터 수집
# 2. 하이브리드 모델 시도
predictor = HybridTransformerLSTMPredictor(...)

# 3. 학습률 조정
learning_rate = 0.0001  # 낮춰보기

# 4. Warmup 추가 (고급)
```

---

## 성능 모니터링

### 학습 중 확인사항

```python
# 100 iteration마다 출력되는 정보 확인
iteration: 0  loss = 0.5  train MSE = 0.3  test MSE = 0.4
iteration: 100  loss = 0.3  train MSE = 0.2  test MSE = 0.35
iteration: 200  loss = 0.2  train MSE = 0.15  test MSE = 0.3
```

**좋은 신호:**
- Loss가 꾸준히 감소
- Train/Test MSE 차이가 크지 않음

**나쁜 신호:**
- Loss가 증가하거나 발산
- Train MSE는 낮은데 Test MSE가 계속 높음

---

## 추가 팁

### 1. Attention 시각화

```python
# transformer_models.py에서 attention weights 반환
# 어떤 시점에 집중하는지 확인 가능
```

### 2. 앙상블

```python
# LSTM + Transformer 예측 평균
predictions_lstm = lstm_model.predict(X)
predictions_transformer = transformer_model.predict(X)
final_predictions = (predictions_lstm + predictions_transformer) / 2
```

### 3. Transfer Learning

```python
# 한 종목에서 학습한 모델을 다른 종목에 적용
model.load_weights('삼성전자_model.h5')
# 새 데이터로 fine-tuning
```

---

## 참고 자료

- **논문:** "Attention Is All You Need" (Vaswani et al., 2017)
- **시계열:** "Temporal Fusion Transformers" (Lim et al., 2021)
- **금융:** "Enhancing Time Series Momentum Strategies Using Deep Neural Networks" (Dixon et al., 2019)

---

## 라이선스 및 주의사항

이 코드는 **연구 및 학습 목적**으로 작성되었습니다.

**⚠️ 투자 유의사항:**
- 과거 데이터 기반 예측일 뿐
- 실제 투자 시 신중한 검증 필요
- 리스크 관리 필수
- 금융 상품은 원금 손실 가능

---

## 연락처

문제나 질문이 있으면 이슈를 남겨주세요.

**Happy Trading! 📈🤖**
