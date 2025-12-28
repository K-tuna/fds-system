# Phase 1: 사전 학습

> Phase 1 (FDS 모델링) 시작 전에 필요한 ML 지식

---

## 개요

### 목적
Phase 1 (XGBoost + LSTM 앙상블 FDS) 구현을 위한 ML 기초

### 선수 조건
- Phase 0 (Python, Numpy, Pandas, Matplotlib) 완료

### 사용 데이터
- **Credit Card Fraud Detection** (Kaggle)
- 284,807건 신용카드 거래, 사기 0.17%
- 다운로드: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 위치: `data/raw/creditcard.csv`

### 총 학습 시간
약 5시간

---

## ⭐ 학습-구현 사이클 (중요!)

**원칙**: 학습 전부 → 구현 전부 ❌ | 학습 → 바로 구현 → 학습 → 구현 ✅

| Cycle | 학습 (Study) | 구현 (Impl) | 설명 |
|-------|--------------|-------------|------|
| 1 | 1-S1, 1-S2, 1-S3 | 1-1, 1-2, 1-3 | ML 기초 + XGBoost |
| 2 | 1-S4 | 1-4 | LSTM |
| 3 | 1-S5 | 1-5 | 앙상블 |
| 4 | - | 1-6, 1-7 | SHAP + FastAPI |

**이유**: 배운 것을 바로 적용해야 기억에 남고 효율적

---

## 학습 목차

| 번호 | 주제 | 시간 | Phase 1 연관 |
|------|------|------|-------------|
| 1-S1 | ML 개념 + Sklearn | 2.5h | 모델 학습/평가 |
| 1-S2 | 모델 저장/튜닝/설명 | 1h | XGBoost, SHAP |
| 1-S3 | XGBoost 심화 | 0.5h | 정형 데이터 모델 |
| 1-S4 | LSTM 기초 | 0.5h | 시계열 모델 |
| 1-S5 | 앙상블 개념 | 0.5h | 모델 결합 |

---

## 1-S1: ML 개념 + Sklearn (2.5시간)

### 학습 내용

**ML 기본 개념**
- 분류 vs 회귀
- 과적합 (Overfitting)
- Train/Test 분할 (왜 나누는지)

**Sklearn 흐름**
```python
# 1. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 2. 모델 생성 및 학습
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 3. 예측
y_pred = model.predict(X_test)

# 4. 평가
auc = roc_auc_score(y_test, y_pred)
```

**평가 지표**
- Accuracy: 전체 정확도 (불균형에서 무의미)
- Recall: 실제 양성 중 맞춘 비율
- Precision: 양성 예측 중 맞춘 비율
- AUC: 순위 매기기 능력

**트리 모델 흐름**
- 결정트리: 기본
- 랜덤포레스트: 여러 트리 앙상블
- XGBoost: 부스팅 방식, 성능 최고

### 체크포인트
- [ ] 분류와 회귀의 차이를 안다
- [ ] 왜 train/test를 나누는지 설명할 수 있다
- [ ] `fit()` → `predict()` 흐름을 안다
- [ ] Recall, Precision, AUC 의미를 안다
- [ ] 결정트리 → 랜덤포레스트 → XGBoost 관계를 안다

---

## 1-S2: 모델 저장/튜닝/설명 (1시간)

### 학습 내용

**joblib - 모델 저장**
```python
import joblib
joblib.dump(model, 'model.pkl')  # 저장
model = joblib.load('model.pkl')  # 로드
```

**Optuna - 하이퍼파라미터 튜닝**
- 하이퍼파라미터: 모델 설정값 (트리 개수, 깊이 등)
- Optuna: 자동으로 최적값 탐색

**SHAP - 모델 설명**
- XAI (설명가능한 AI)
- 각 피처가 예측에 얼마나 기여했는지 계산
- 금융에서 규제 때문에 필수

### 체크포인트
- [ ] joblib로 모델 저장/로드할 수 있다
- [ ] 하이퍼파라미터가 뭔지 안다
- [ ] SHAP이 왜 필요한지 설명할 수 있다

---

## 1-S3: XGBoost 심화 (0.5시간)

### 학습 내용

**XGBoost 특징**
- Gradient Boosting 기반
- 정형 데이터에서 최고 성능
- 결측치 자동 처리

**주요 파라미터**
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,      # 트리 개수
    max_depth=6,           # 트리 깊이
    learning_rate=0.1,     # 학습률
    scale_pos_weight=10,   # 클래스 불균형 처리
    use_label_encoder=False,
    eval_metric='auc'
)
```

**불균형 데이터 처리**
- `scale_pos_weight`: 양성 클래스 가중치
- FDS에서 사기 거래가 1% 미만 → 가중치 필요

### 체크포인트
- [ ] XGBoost가 왜 정형 데이터에서 강한지 안다
- [ ] `scale_pos_weight`의 역할을 안다
- [ ] 주요 하이퍼파라미터를 안다

---

## 1-S4: LSTM 기초 (0.5시간)

### 학습 내용

**RNN과 LSTM**
- RNN: 순서가 있는 데이터 처리
- LSTM: 장기 기억 가능한 RNN

**왜 FDS에서 LSTM?**
- 거래 패턴은 시간 순서가 중요
- "최근 10건의 거래"를 하나의 시퀀스로 보고 학습
- 갑자기 평소와 다른 패턴 → 이상 탐지

**PyTorch LSTM**
```python
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return torch.sigmoid(out)
```

### 체크포인트
- [ ] LSTM이 왜 시퀀스 데이터에 적합한지 안다
- [ ] FDS에서 LSTM을 쓰는 이유를 설명할 수 있다
- [ ] PyTorch 기본 구조를 이해한다

---

## 1-S5: 앙상블 개념 (0.5시간)

### 학습 내용

**앙상블이란?**
- 여러 모델의 예측을 결합
- 단일 모델보다 일반화 성능 향상

**앙상블 종류**
| 방식 | 설명 | 예시 |
|------|------|------|
| Bagging | 여러 모델 평균 | RandomForest |
| Boosting | 순차 학습 | XGBoost |
| Stacking | 메타 모델 학습 | 복잡 |
| Voting | 단순 투표/평균 | 직접 구현 |

**FDS 앙상블 전략**
```python
# Weighted Average
xgb_prob = xgb_model.predict_proba(X)[:, 1]
lstm_prob = lstm_model.predict(X)

# XGBoost 0.6 + LSTM 0.4
final_prob = 0.6 * xgb_prob + 0.4 * lstm_prob
```

**왜 XGBoost + LSTM?**
- XGBoost: 정형 피처 (금액, 시간, 카드 종류 등)
- LSTM: 시퀀스 패턴 (거래 이력)
- 상호 보완적 → 성능 향상

### 체크포인트
- [ ] 앙상블의 장점을 안다
- [ ] Weighted Average 방식을 이해한다
- [ ] XGBoost와 LSTM이 왜 상호 보완적인지 설명할 수 있다

---

## 학습 완료 후

### 전체 체크리스트
- [x] 1-S1: ML 개념 + Sklearn ✅
- [x] 1-S2: 모델 저장/튜닝/설명 ✅
- [x] 1-S3: XGBoost 심화 ✅
- [ ] 1-S4: LSTM 기초 (노트북 생성됨, Cycle 2에서 진행)
- [ ] 1-S5: 앙상블 개념 (Cycle 3에서 진행)

### 다음 단계
1-S3 완료 → **1-1, 1-2, 1-3 구현** → 1-S4 학습 → 1-4 구현 → ...
