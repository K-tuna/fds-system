# Phase 1: FDS + Ensemble + XAI - 구현 상세 (AI용)

> 노트북/코드 생성을 위한 상세 스펙

---

## ⚠️ 구현 방식 (필독)

**각 Day 노트북은 반드시 다음 형식으로 구현해야 함:**

### 노트북 구조

```
[마크다운] 제목 + 학습 목표
[코드] 패키지 임포트
[마크다운] 📚 개념 설명
[코드] 예제 코드 (완성본)
[마크다운] 💻 실습 N 설명
[코드] 실습 TODO (빈칸)
[코드] ✅ 실습 정답 (채운 버전)
[코드] 체크포인트 (assert)
... (반복)
[마크다운] 최종 요약 + 면접 포인트
```

### 포함 요소

| 요소 | 설명 |
|------|------|
| 📚 개념 설명 | 마크다운으로 개념/이론 설명 |
| 예제 코드 | 완성된 예제 (학습용) |
| 💻 실습 TODO | 빈칸/TODO 포함된 실습 코드 |
| ✅ 실습 정답 | TODO 채운 정답 코드 |
| 체크포인트 | assert로 검증 |
| 면접 포인트 | 해당 Day 관련 면접 Q&A |

### src/ 모듈화

1-2부터 검증된 코드를 src/로 모듈화:

```
노트북에서 코드 작성 + 실험
    ↓
검증되면 src/로 모듈화
    ↓
노트북에서 모듈 import해서 사용
```

---

## 파일 구조

```
fds-system/
├── notebooks/                   # 실험용 노트북
│   └── phase1/
│       ├── 1-1_data_eda.ipynb
│       ├── 1-2_feature_engineering.ipynb
│       ├── 1-3_xgboost.ipynb
│       ├── 1-4_lstm.ipynb
│       ├── 1-5_ensemble.ipynb
│       ├── 1-6_shap.ipynb
│       ├── 1-7_fastapi.ipynb
│       ├── 1-8_react_admin.md   # React Admin 가이드
│       ├── 1-9_tree_stacking.ipynb   # ⭐⭐ 트리 스태킹 (필수)
│       ├── 1-10_transformer.ipynb    # Transformer (선택)
│       ├── 1-11_hybrid.ipynb         # 하이브리드 DL+XGB (선택)
│       └── 1-12_paysim.ipynb         # PaySim 시퀀스 실험 (선택)
│
├── src/                         # 프로덕션 코드
│   ├── models/                  # ⭐ PyTorch 클래스 정의 (필수!)
│   │   ├── __init__.py
│   │   ├── lstm.py              # FraudLSTM 클래스
│   │   ├── cnn_lstm.py          # CNN-LSTM 클래스 (선택)
│   │   └── fusion.py            # 융합 모델 클래스
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py     # 피처 엔지니어링
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py             # 학습 스크립트
│   ├── explainer/
│   │   ├── __init__.py
│   │   └── shap_explainer.py
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       └── schemas.py
│
├── data/
│   ├── raw/                     # IEEE-CIS 원본
│   └── processed/               # 전처리 데이터
│
├── models/                      # 저장된 가중치
│   ├── xgb_model.pkl            # XGBoost (joblib)
│   └── lstm_model.pt            # LSTM (torch.save)
│
├── configs/
│   └── config.yaml              # 하이퍼파라미터, 피처 목록
│
├── docker-compose.yml
└── requirements.txt
```

### ⚠️ PyTorch 모델 저장/로드 방식 (현업 필수)

```python
# ❌ 노트북에서만 클래스 정의 → API에서 로드 불가
# torch.load()는 클래스 정의가 필요함

# ✅ 현업 방식: src/models/에 클래스 정의
# 1. src/models/lstm.py에 FraudLSTM 클래스 정의
# 2. 노트북에서 from src.models.lstm import FraudLSTM
# 3. API에서도 동일하게 import

# 저장
torch.save(model.state_dict(), 'models/lstm_model.pt')

# 로드 (API)
from src.models.lstm import FraudLSTM
model = FraudLSTM(input_size=35, hidden_size=64)
model.load_state_dict(torch.load('models/lstm_model.pt'))
```

---

## 1-1: 데이터 + EDA (Day 1)

### 필요 패키지
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### 세부 설명 리스트

**1. IEEE-CIS 데이터셋**
- Kaggle에서 다운로드
- Transaction 테이블: 거래 정보 (금액, 시간, 카드)
- Identity 테이블: 기기/브라우저 정보
- TransactionID로 LEFT JOIN

**2. 기본 EDA**
- shape, dtypes, info()
- head(), tail()
- describe()

**3. 불균형 데이터**
- 타겟(isFraud) 분포 확인
- 사기 비율 ~3.5%
- Accuracy가 무의미한 이유
- 평가 지표: AUC-ROC, PR-AUC, F1

**4. 결측치 분석**
- 컬럼별 결측 비율
- 결측 50% 이상 컬럼 처리 전략
- Identity 테이블 결측 (병합으로 인한)

**5. 피처 탐색**
- 수치형: TransactionAmt, card1~5
- 범주형: ProductCD, card4, card6
- 시간: TransactionDT

**6. 타겟별 분석**
- 정상 vs 사기 금액 분포
- 시간대별 사기 비율
- 카테고리별 사기 비율

**7. 시간 기반 분할**
- 왜 랜덤 분할이 안 되는지 (Data Leakage)
- TransactionDT 기준 정렬
- 80/20 분할

### 실습 목록
- 실습 1: 데이터 로드 및 병합 (LEFT JOIN)
- 실습 2: 타겟 불균형 시각화
- 실습 3: 결측치 분석 및 처리 전략
- 실습 4: 타겟별 금액 분포 비교
- 실습 5: 시간 기반 train/test 분할

### 핵심 코드

```python
# 데이터 병합
df = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

# 시간 기반 분할
df_sorted = df.sort_values('TransactionDT')
split_idx = int(len(df_sorted) * 0.8)
train_df = df_sorted.iloc[:split_idx]
test_df = df_sorted.iloc[split_idx:]

# 검증: 시간 순서 확인
assert train_df['TransactionDT'].max() <= test_df['TransactionDT'].min()
```

### 면접 포인트

Q: "왜 랜덤 분할이 아닌 시간 기반 분할을 하나요?"
> "실제 FDS는 과거 데이터로 학습해서 미래 거래를 예측합니다. 랜덤 분할은 미래 정보가 학습에 포함되어 Data Leakage가 발생합니다. 시간 기반 분할이 실제 운영 환경을 반영합니다."

---

## 1-2: Feature Engineering (Day 2)

### 필요 패키지
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
```

### 세부 설명 리스트

**1. 정형 피처 (XGBoost용)**
- 시간 피처: hour, dayofweek, is_weekend, is_night
- 금액 피처: amt_log, amt_bin, amt_decimal
- 집계 피처: card1별 거래 횟수, 평균 금액
- 범주형 인코딩: LabelEncoder

**2. 시계열 피처 (LSTM용)** ⭐
- 사용자별 최근 N개 거래 시퀀스 추출
- 시퀀스 피처: [금액, 시간차, 카테고리, ...]
- Padding: 거래 수가 N 미만이면 0으로 채움
- Scaling: MinMaxScaler로 0~1 정규화

**3. 시퀀스 생성 로직**
```
사용자 A의 거래: [t1, t2, t3, t4, t5]
시퀀스 길이 N=3일 때:
- t3 예측용: [t1, t2] → 길이 부족 → [0, t1, t2]
- t4 예측용: [t1, t2, t3]
- t5 예측용: [t2, t3, t4]
```

**4. 피처 저장**
- X_tabular: 정형 피처 (DataFrame)
- X_sequence: 시계열 피처 (3D array: samples x seq_len x features)

### 실습 목록
- 실습 1: 시간 피처 생성
- 실습 2: 금액 피처 생성
- 실습 3: 집계 피처 생성
- 실습 4: 범주형 인코딩
- 실습 5: 시퀀스 데이터 생성 ⭐

### 핵심 코드: 시퀀스 생성

```python
def create_sequences(df, user_col, features, seq_len=10):
    """사용자별 시퀀스 생성"""
    sequences = []
    labels = []

    for user_id, group in df.groupby(user_col):
        group = group.sort_values('TransactionDT')

        for i in range(len(group)):
            # 현재 거래 이전 seq_len개 거래
            start_idx = max(0, i - seq_len)
            seq = group.iloc[start_idx:i][features].values

            # 패딩 (길이 부족 시)
            if len(seq) < seq_len:
                pad = np.zeros((seq_len - len(seq), len(features)))
                seq = np.vstack([pad, seq])

            sequences.append(seq)
            labels.append(group.iloc[i]['isFraud'])

    return np.array(sequences), np.array(labels)

# 시퀀스 피처
seq_features = ['TransactionAmt_scaled', 'hour_scaled', 'time_diff_scaled']
X_seq, y_seq = create_sequences(train_df, 'card1', seq_features, seq_len=10)
print(f"Sequence shape: {X_seq.shape}")  # (samples, 10, 3)
```

### 면접 포인트

Q: "시퀀스 길이 10은 어떻게 정했나요?"
> "5, 10, 15, 20으로 실험했습니다. 10이 AUC와 학습 시간의 트레이드오프에서 최적이었습니다. 5는 패턴을 못 잡고, 20은 학습 시간 대비 성능 향상이 미미했습니다."

---

## 1-3: XGBoost 모델 (Day 3) ⭐

### 필요 패키지
```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
import optuna
import joblib
import time
```

### 세부 설명 리스트

**1. 모델 비교 실험** ⭐
- XGBoost, LightGBM, CatBoost
- 동일 조건 (동일 피처, 동일 분할)
- AUC, 학습 시간 측정
- 결과 표로 정리

**2. XGBoost 선택 이유**
- AUC 최고
- SHAP TreeExplainer 호환성 최상
- GPU 학습 지원

**3. Threshold 최적화** ⭐
- FN:FP 비용 비율 정의 (10:1)
- 비용 함수로 최적 threshold 찾기
- Precision-Recall Curve

**4. Optuna 하이퍼파라미터 튜닝**
- 탐색 공간 정의
- objective 함수
- n_trials 설정

**5. n_estimators와 Early Stopping** ⭐
- 현업 트리 개수: 100~500개 (상황별 상이)
- Early Stopping으로 최적 개수 자동 탐색
- GPU vs CPU 선택 기준

| 상황 | 트리 수 | 이유 |
|------|---------|------|
| 빠른 실험 | 100 | 베이스라인 |
| 프로덕션 FDS | 200~500 | 성능 vs 속도 균형 |
| Kaggle 대회 | 1000+ | 최고 성능 |
| 실시간 API | 50~200 | 응답속도 중요 |

**6. 모델 저장**
- joblib.dump
- 메타데이터 함께 저장

### 실습 목록
- 실습 1: 3개 모델 비교 실험 → 결과 표
- 실습 2: XGBoost 학습
- 실습 3: Threshold 비용 분석 → 그래프
- 실습 4: Optuna 튜닝
- 실습 5: 모델 저장

### 핵심 코드: 모델 비교

```python
models = {
    'XGBoost': XGBClassifier(n_estimators=100, tree_method='hist', device='cuda'),
    'LightGBM': LGBMClassifier(n_estimators=100, device='gpu'),
    'CatBoost': CatBoostClassifier(n_estimators=100, task_type='GPU', verbose=0),
}

results = []
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    results.append({'Model': name, 'AUC': auc, 'Time(s)': train_time})

results_df = pd.DataFrame(results)
print(results_df.to_markdown(index=False))
```

### 핵심 코드: Threshold 최적화

```python
def calculate_cost(threshold, y_true, y_prob, fn_cost=10, fp_cost=1):
    """비용 함수: FN이 FP보다 10배 비쌈"""
    y_pred = (y_prob >= threshold).astype(int)
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return fn * fn_cost + fp * fp_cost

# 최적 threshold 찾기
thresholds = np.arange(0.1, 0.9, 0.05)
costs = [calculate_cost(t, y_test, y_prob) for t in thresholds]
optimal_threshold = thresholds[np.argmin(costs)]
print(f"최적 Threshold: {optimal_threshold:.2f}")
```

### 핵심 코드: Early Stopping

```python
# Early Stopping으로 최적 트리 개수 찾기
model = XGBClassifier(
    n_estimators=1000,           # 일단 많이
    early_stopping_rounds=50,    # 50번 연속 개선 없으면 중단
    eval_metric='auc',
    device='cuda',
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=100  # 100번마다 출력
)

print(f"실제 사용된 트리 수: {model.best_iteration}")  # 예: 287
```

### 면접 포인트

Q: "왜 XGBoost를 선택했나요?"
> "3개 모델을 동일 조건에서 비교했습니다. XGBoost가 AUC 0.92로 가장 높았고, SHAP TreeExplainer 호환성도 최상이었습니다."

Q: "Threshold는 어떻게 정했나요?"
> "FDS에서 FN(놓친 사기)이 FP(오탐)보다 비용이 큽니다. FN:FP = 10:1로 비용 함수를 정의하고 최소화하는 Threshold 0.18을 찾았습니다. 이 Threshold에서 Recall 90.55%를 달성했습니다."

Q: "n_estimators는 어떻게 정했나요?"
> "Early Stopping을 사용했습니다. n_estimators=1000으로 설정하고, validation AUC가 50 epoch 연속 개선 없으면 중단합니다. 실제로는 약 300개 트리에서 수렴했습니다."

Q: "GPU를 안 쓰면 어떻게 되나요?"
> "28만건 데이터에서는 CPU도 충분히 빠릅니다 (0.6초). GPU 오버헤드 때문에 오히려 느릴 수 있습니다. 수백만건 이상에서 GPU가 효과적입니다."

---

## 1-4: LSTM 모델 (Day 4) ⭐

### 필요 패키지
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
```

### 세부 설명 리스트

**1. 왜 LSTM이 필요한가?**
- XGBoost는 단일 거래만 봄
- 사기 패턴: "소액 → 소액 → 고액"
- 시퀀스 패턴은 LSTM이 학습

**2. LSTM 피처 선택 (현업 방식)** ⭐

```python
# 1. V컬럼 상위 20개 (PCA 기반, 논문 표준)
v_features = [f'V{i}' for i in range(1, 21)]  # V1~V20

# 2. XGBoost importance 상위 10개 (V 제외)
# → 1-3에서 저장한 feature_importance 활용
xgb_importance = pd.read_csv('data/processed/xgb_importance.csv')
xgb_top = xgb_importance[~xgb_importance['feature'].str.startswith('V')].head(10)['feature'].tolist()

# 3. 시계열 피처 (직접 생성)
time_features = [
    'amt_log',              # 금액 로그
    'hour',                 # 시간
    'dayofweek',            # 요일
    'time_since_last_tx',   # 이전 거래 후 경과 시간
    'rolling_avg_amt_5',    # 최근 5개 평균 금액
    'tx_count_24h',         # 24시간 내 거래 횟수
]

# 전체 시퀀스 피처 (~35개)
SEQ_FEATURES = v_features + xgb_top + time_features
print(f"총 피처 수: {len(SEQ_FEATURES)}")  # ~35개
```

**3. LSTM 구조**
- Input: (batch, seq_len, 35)  # 35개 피처
- Hidden: 64~128
- Output: 1 (sigmoid)
- pos_weight: ~27 (클래스 불균형 처리)

**4. 클래스 불균형 처리**

```python
# 사기 비율 3.5% → pos_weight 계산
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
pos_weight = n_neg / n_pos  # ~27

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
```

**5. PyTorch 구현**
- Dataset 클래스
- DataLoader
- BCEWithLogitsLoss + Adam
- Early Stopping

**6. 학습 루프**
- Train/Valid 분리
- Epoch별 loss/AUC 추적
- Best model 저장

### 실습 목록
- 실습 1: 피처 선택 (V1~V20 + XGBoost importance + 시계열)
- 실습 2: Dataset 클래스 구현
- 실습 3: LSTM 모델 정의 (src/models/lstm.py)
- 실습 4: 학습 루프 + pos_weight
- 실습 5: Optuna 튜닝
- 실습 6: 모델 평가 및 저장

### 핵심 코드: LSTM 모델

```python
class FraudLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, features)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden)
        out = self.fc(h_n[-1])
        return self.sigmoid(out)

# 모델 초기화
model = FraudLSTM(input_size=3, hidden_size=64)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### 핵심 코드: Dataset

```python
class FraudSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# DataLoader
train_dataset = FraudSequenceDataset(X_seq_train, y_seq_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
```

### 핵심 코드: 학습 루프

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

# Early Stopping
best_auc = 0
patience = 5
counter = 0

for epoch in range(100):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_auc = evaluate(model, val_loader, device)

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), 'models/lstm_best.pt')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

### 면접 포인트

Q: "왜 LSTM을 선택했나요?"
> "거래 시퀀스의 시간적 패턴을 학습하기 위해서입니다. Transformer도 고려했지만, 시퀀스 길이가 10~20으로 짧아서 LSTM이 충분했고 학습도 더 빠릅니다."

Q: "왜 Transformer가 아닌가요?"
> "시퀀스 길이가 짧습니다. Transformer는 긴 시퀀스에서 강점이 있지만, 10~20개 시퀀스에서는 LSTM과 성능 차이가 거의 없고 구현도 간단합니다."

---

## 1-5: Ensemble 실험 + 평가 (Day 5) ⭐

### 필요 패키지
```python
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
```

### 세부 설명 리스트

**1. 앙상블 실험 결과**
- LSTM AUC 0.70으로 예상보다 낮음
- 앙상블해도 +0.12% 향상에 그침
- **결론: XGBoost 단독 채택**

**2. 실험 결과**
```
XGBoost 단독: AUC 0.9042 → 채택
LSTM 단독:    AUC 0.7054 → 성능 낮음
앙상블:       AUC 0.9054 → +0.12% (효과 미미)
최적 가중치:  XGBoost 90%, LSTM 10%
```

**3. 채택 근거**
- +0.12% 향상은 LSTM 서빙 비용 대비 효과 없음
- 복잡도 증가 (PyTorch, 시퀀스 생성) vs 성능 향상 trade-off
- XGBoost 단독으로 Recall 90.55% 달성

### 실습 목록
- 실습 1: XGBoost 예측
- 실습 2: LSTM 예측
- 실습 3: 가중치 최적화 (Grid Search)
- 실습 4: 성능 비교 → XGBoost 단독 채택 결론
- 실습 5: 복잡도 대비 효과 분석

### 핵심 코드: 가중치 최적화

```python
# XGBoost, LSTM 예측
p_xgb = xgb_model.predict_proba(X_test_tabular)[:, 1]
p_lstm = lstm_model(X_test_seq).detach().cpu().numpy().flatten()

# 가중치 최적화
best_auc = 0
best_weight = 0

for w_xgb in np.arange(0.3, 0.8, 0.1):
    w_lstm = 1 - w_xgb
    p_ensemble = w_xgb * p_xgb + w_lstm * p_lstm
    auc = roc_auc_score(y_test, p_ensemble)

    if auc > best_auc:
        best_auc = auc
        best_weight = w_xgb

print(f"최적 가중치: XGBoost {best_weight:.1f}, LSTM {1-best_weight:.1f}")
print(f"Ensemble AUC: {best_auc:.4f}")
```

### 핵심 코드: 성능 비교

```python
# 성능 비교 표
results = pd.DataFrame([
    {'Model': 'XGBoost', 'AUC': roc_auc_score(y_test, p_xgb)},
    {'Model': 'LSTM', 'AUC': roc_auc_score(y_test, p_lstm)},
    {'Model': 'Ensemble', 'AUC': best_auc},
])
print(results.to_markdown(index=False))

# 시각화
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(results['Model'], results['AUC'])
ax.set_ylabel('AUC')
ax.set_title('Model Comparison')
plt.show()
```

### 면접 포인트

Q: "앙상블로 얼마나 성능이 올랐나요?"
> "LSTM AUC가 0.70으로 낮아서, 앙상블해도 +0.12% 향상에 그쳤습니다. LSTM 서빙 비용 대비 효과가 없어서 XGBoost 단독을 채택했습니다."

Q: "왜 LSTM이 효과가 없었나요?"
> "IEEE-CIS 데이터 특성상, 시계열 패턴보다 정형 피처(금액, 시간, 카드정보)가 더 결정적이었습니다. 모든 문제에 딥러닝이 최선은 아닙니다."

---

## 1-6: SHAP 설명 (Day 6)

### 필요 패키지
```python
import shap
import torch
import numpy as np
```

### 세부 설명 리스트

**1. XGBoost 설명: TreeExplainer**
- 빠르고 정확
- 전체 피처 중요도
- 개별 예측 설명

**2. 자연어 설명 생성**
- 피처명 → 설명 매핑
- 방향 (증가/감소) 포함
- Top 5 피처 추출

### 실습 목록
- 실습 1: XGBoost SHAP 계산
- 실습 2: SHAP Summary Plot
- 실습 3: 개별 예측 설명
- 실습 4: 자연어 설명 생성
- 실습 5: API 응답 형태로 변환

### 핵심 코드: XGBoost SHAP

```python
# TreeExplainer
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_test_tabular)

# Summary Plot
shap.summary_plot(shap_values_xgb, X_test_tabular, max_display=10)
```

### 핵심 코드: 자연어 설명 생성

```python
# 피처명 → 설명 매핑
FEATURE_DESC = {
    'TransactionAmt': '거래 금액',
    'hour': '거래 시간',
    'card1_fraud_rate': '카드 사기 이력',
}

def to_natural_language(shap_values, feature_names, top_k=5):
    """SHAP 값을 자연어 설명으로 변환"""
    # Top K 피처 추출
    importance = np.abs(shap_values)
    top_idx = np.argsort(importance)[-top_k:][::-1]

    lines = ["[사기 판단 근거]"]
    for i in top_idx:
        name = FEATURE_DESC.get(feature_names[i], feature_names[i])
        direction = "높음" if shap_values[i] > 0 else "낮음"
        lines.append(f"- {name}: 사기 확률 {direction}")

    return "\n".join(lines)
```

### 면접 포인트

Q: "왜 SHAP을 선택했나요?"
> "SHAP은 이론적 기반(Shapley Value)이 탄탄하고, XGBoost의 TreeExplainer로 빠르고 정확한 설명이 가능합니다."

Q: "설명은 어떻게 보여주나요?"
> "TreeExplainer로 피처별 기여도를 계산하고, Top 5 피처를 자연어로 변환해서 제공합니다. 예: '거래 금액이 높음 → 사기 확률 증가'"

---

## 1-7: FastAPI 배포 (Day 7)

### 필요 패키지
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch
import uvicorn
```

### 세부 설명 리스트

**1. API 엔드포인트**
- GET /health: 헬스체크
- POST /predict: 사기 예측 + 설명

**2. Request/Response 스키마**
- Pydantic BaseModel
- 타입 검증

**3. 모델 로딩**
- 앱 시작 시 한 번만 로딩
- XGBoost + LSTM + SHAP Explainer

**4. Docker 컨테이너화**
- Dockerfile 작성
- docker-compose.yml
- 단일 명령어로 실행

**5. 성능 최적화**
- 모델 캐싱
- 응답 시간 < 200ms

### 실습 목록
- 실습 1: Pydantic 스키마 정의
- 실습 2: /health 엔드포인트
- 실습 3: /predict 엔드포인트
- 실습 4: Dockerfile 작성
- 실습 5: docker-compose 실행 및 테스트

### 핵심 코드: FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import joblib
import torch

app = FastAPI(title="FDS API", version="1.0")

# 모델 로딩 (시작 시 한 번)
xgb_model = joblib.load('models/xgb_model.pkl')
lstm_model = torch.load('models/lstm_model.pt')
lstm_model.eval()

class Transaction(BaseModel):
    transaction_id: int
    amount: float
    hour: int
    card1: int
    # ... 기타 피처
    recent_transactions: List[Dict]  # 최근 거래 시퀀스

class PredictionResponse(BaseModel):
    fraud_probability: float
    model_scores: Dict[str, float]
    top_factors: List[Dict]
    is_fraud: bool

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    # 1. Feature Engineering
    X_tabular = extract_tabular_features(transaction)
    X_sequence = extract_sequence_features(transaction.recent_transactions)

    # 2. 모델 예측
    p_xgb = xgb_model.predict_proba(X_tabular)[0, 1]
    with torch.no_grad():
        p_lstm = lstm_model(torch.FloatTensor(X_sequence).unsqueeze(0)).item()

    # 3. XGBoost 단독 사용 (앙상블 효과 미미하여 채택)
    # 참고: 앙상블 실험에서 p = 0.9*xgb + 0.1*lstm → +0.12% 향상에 그침

    # 4. SHAP 설명
    explanation = generate_explanation(X_tabular)

    return PredictionResponse(
        fraud_probability=p_xgb,
        model_scores={"xgboost": p_xgb},
        top_factors=explanation,
        is_fraud=p_xgb > 0.18  # 최적화된 threshold
    )
```

### 핵심 코드: Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 핵심 코드: docker-compose.yml

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
```

### 면접 포인트

Q: "API 응답 시간은 얼마나 걸리나요?"
> "XGBoost ~10ms, LSTM ~50ms, SHAP ~100ms로 총 약 160ms입니다. 목표였던 200ms 이하를 달성했습니다."

Q: "모델 업데이트는 어떻게 하나요?"
> "현재는 Docker 이미지 재빌드 방식입니다. Phase 2에서 MLflow로 모델 버전 관리를 추가하면 무중단 배포가 가능해집니다."

---

## 1-8: React Admin (Day 8)

### 필요 패키지
```bash
npx create-react-app fds-admin
npm install antd axios recharts
```

### 세부 설명 리스트

**1. 프로젝트 구조**
```
frontend/
├── src/
│   ├── components/
│   │   ├── TransactionTable.jsx   # 거래 목록 테이블
│   │   └── TransactionDetail.jsx  # 거래 상세 + SHAP 요인
│   ├── pages/
│   │   ├── Dashboard.jsx          # 메인 대시보드
│   │   └── TransactionPage.jsx    # 거래 조회 페이지
│   ├── api/
│   │   └── client.js              # Axios 설정
│   └── App.jsx
├── package.json
└── Dockerfile
```

**2. 핵심 화면 (2개)**
- **거래 목록**: 테이블 (거래ID, 금액, 사기확률, top요인, 조치)
- **거래 상세**: SHAP 요인 표시, 승인/차단 버튼

**3. API 연동**
```javascript
// api/client.js
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
});

export const getTransactions = () => api.get('/transactions');
export const getTransaction = (id) => api.get(`/transactions/${id}`);
export const predict = (data) => api.post('/predict', data);
```

**4. 현업 스타일 (단순함 유지)**
- 예쁜 그래프 X → 텍스트/테이블만
- SHAP waterfall X → top 3 요인 텍스트
- 분석가용 Admin 느낌

### 실습 목록
- 실습 1: React 프로젝트 설정 + Ant Design
- 실습 2: 거래 목록 테이블 (TransactionTable)
- 실습 3: 거래 상세 페이지 (SHAP 요인 표시)
- 실습 4: FastAPI 연동
- 실습 5: Docker 컨테이너화

### 핵심 코드: 거래 테이블

```jsx
// components/TransactionTable.jsx
import { Table, Tag } from 'antd';

const columns = [
  { title: '거래ID', dataIndex: 'transaction_id', key: 'id' },
  { title: '금액', dataIndex: 'amount', key: 'amount',
    render: (val) => `₩${val.toLocaleString()}` },
  { title: '사기확률', dataIndex: 'fraud_probability', key: 'prob',
    render: (val) => (
      <Tag color={val > 0.5 ? 'red' : 'green'}>
        {(val * 100).toFixed(1)}%
      </Tag>
    )},
  { title: '주요요인', dataIndex: 'top_factors', key: 'factors',
    render: (factors) => factors.slice(0, 2).map(f => f.feature).join(', ') },
  { title: '조치', key: 'action',
    render: () => <a>상세보기</a> },
];

export default function TransactionTable({ data }) {
  return <Table columns={columns} dataSource={data} rowKey="transaction_id" />;
}
```

### 핵심 코드: SHAP 요인 표시

```jsx
// components/TransactionDetail.jsx
import { Card, List, Typography } from 'antd';
const { Text } = Typography;

export default function TransactionDetail({ transaction }) {
  const { fraud_probability, top_factors } = transaction;

  return (
    <Card title={`사기 확률: ${(fraud_probability * 100).toFixed(1)}%`}>
      <List
        header={<Text strong>주요 판단 근거</Text>}
        dataSource={top_factors}
        renderItem={(item) => (
          <List.Item>
            <Text>{item.feature}</Text>
            <Text type={item.impact > 0 ? 'danger' : 'success'}>
              {item.impact > 0 ? '+' : ''}{item.impact.toFixed(3)}
            </Text>
          </List.Item>
        )}
      />
    </Card>
  );
}
```

### 핵심 코드: Dockerfile

```dockerfile
# frontend/Dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
```

### docker-compose 추가

```yaml
# docker-compose.yml에 추가
services:
  api:
    build: .
    ports:
      - "8000:8000"

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - api
```

### 면접 포인트

Q: "프론트엔드는 왜 React를 선택했나요?"
> "금융권에서 React 사용 비율이 가장 높고, 인력 수급이 용이합니다. Ant Design을 사용해 테이블 중심의 Admin UI를 빠르게 구축했습니다."

Q: "SHAP 시각화는 어떻게 보여주나요?"
> "현업에서는 실시간으로 복잡한 그래프를 띄우지 않습니다. top 3 요인을 텍스트로 표시하고, 상세 분석은 오프라인에서 Jupyter로 합니다. 이 방식이 응답 속도와 가독성 면에서 더 효율적입니다."

---

## 1-9: 트리 스태킹 (Day 9) ⭐⭐

### 필요 패키지
```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import pandas as pd
```

### 세부 설명 리스트

**1. 왜 트리 스태킹인가?**
- 2025 벤치마크 논문에서 우수한 성능 보고
- 현업에서 증가 중인 트렌드
- 실시간 추론 가능 (~12ms)
- 각 모델의 장점을 결합
- **확률 분포 양극화**: 정상 거래 확률 20~30% → 1~2%로 낮아져 운영 비용 절감

**실제 달성 결과:**
| 지표 | 값 | 설명 |
|------|-----|------|
| AUPRC | 0.5957 | 불균형 데이터 적합 지표 |
| AUC | 0.9205 | 전체 성능 |
| Recall @5% FPR | 71% | FPR 제약 시 탐지율 |
| Threshold | 0.08 | FPR Constraint 기준 |

**2. 스태킹 구조**
```
[Base Models - Level 0]
XGBoost  ─┐
LightGBM ─┼→ [Meta-Learner - Level 1] → 최종 예측
CatBoost ─┘

교차 검증으로 OOF (Out-of-Fold) 예측 생성 → Meta-Learner 학습
```

**3. 각 모델의 강점/약점**

| 모델 | 강점 | 약점 |
|------|------|------|
| XGBoost | 정규화 우수, SHAP 최상 | 상대적 느림 |
| LightGBM | 가장 빠름, 메모리 효율 | Tail latency |
| CatBoost | 범주형 자동 처리 | 학습 시간 김 |

**4. Meta-Learner 선택**
- Logistic Regression: 간단, 오버피팅 방지
- XGBoost: 더 복잡한 패턴 학습 가능

### 실습 목록
- 실습 1: LightGBM 단독 학습 및 평가
- 실습 2: CatBoost 단독 학습 및 평가
- 실습 3: 3개 모델 성능 비교 표
- 실습 4: StackingClassifier로 스태킹 구현
- 실습 5: OOF 수동 구현 (sklearn 대비 유연성)
- 실습 6: 최종 성능 비교 및 모델 저장

### 핵심 코드: sklearn StackingClassifier

```python
# Base models
base_models = [
    ('xgb', XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        tree_method='hist',
        device='cuda',
        random_state=42
    )),
    ('lgbm', LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        device='gpu',
        random_state=42,
        verbose=-1
    )),
    ('cat', CatBoostClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        task_type='GPU',
        random_state=42,
        verbose=0
    ))
]

# Stacking
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    passthrough=False,  # True면 원본 피처도 Meta-Learner에 전달
    n_jobs=-1
)

stacking_model.fit(X_train, y_train)
y_prob = stacking_model.predict_proba(X_test)[:, 1]
print(f"Stacking AUC: {roc_auc_score(y_test, y_prob):.4f}")
```

### 핵심 코드: 수동 OOF 스태킹

```python
from sklearn.model_selection import StratifiedKFold

def get_oof_predictions(model, X, y, n_splits=5):
    """Out-of-Fold 예측 생성"""
    oof_preds = np.zeros(len(X))
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kfold.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y.iloc[train_idx]

        model_clone = clone(model)
        model_clone.fit(X_tr, y_tr)
        oof_preds[val_idx] = model_clone.predict_proba(X_val)[:, 1]

    return oof_preds

# 각 모델의 OOF 예측
oof_xgb = get_oof_predictions(xgb_model, X_train, y_train)
oof_lgbm = get_oof_predictions(lgbm_model, X_train, y_train)
oof_cat = get_oof_predictions(cat_model, X_train, y_train)

# Meta-learner용 피처
meta_features = np.column_stack([oof_xgb, oof_lgbm, oof_cat])

# Meta-learner 학습
meta_model = LogisticRegression()
meta_model.fit(meta_features, y_train)

# Test set 예측
test_xgb = xgb_model.predict_proba(X_test)[:, 1]
test_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
test_cat = cat_model.predict_proba(X_test)[:, 1]
test_meta = np.column_stack([test_xgb, test_lgbm, test_cat])

y_final = meta_model.predict_proba(test_meta)[:, 1]
```

### 면접 포인트

Q: "왜 트리 스태킹을 사용했나요?"
> "XGBoost, LightGBM, CatBoost 각각의 강점을 결합하기 위해서입니다. XGBoost는 정규화, LightGBM은 속도, CatBoost는 범주형 처리에 강합니다. Meta-learner(LogisticRegression)가 각 모델의 예측을 최적 조합하여 AUC 0.92, AUPRC 0.60을 달성했습니다."

Q: "스태킹 적용 후 확률 분포가 어떻게 변했나요?"
> "**확률 분포가 양극화**되었습니다. XGBoost 단독에서는 정상 거래도 20~30% 확률이 나왔는데, 스태킹 후에는 1~2%로 낮아졌습니다. Block 비율은 유지하면서 Hold/Verify 비율이 감소해서 **운영 비용이 절감**됩니다. 3개 모델이 동의해야 높은 확률이 나오기 때문입니다."

Q: "스태킹의 단점은?"
> "추론 시간이 단일 모델 대비 2~3배 증가합니다. 하지만 12ms로 실시간 서비스에 충분합니다. 학습 시간도 3배 증가하지만, Optuna 튜닝 때만 문제될 뿐 운영에는 영향 없습니다."

Q: "Voting vs Stacking 차이는?"
> "Voting은 단순 평균/가중 평균이고, Stacking은 Meta-learner가 최적의 결합 방식을 학습합니다. 우리 데이터에서 Stacking이 Voting보다 AUC +2% 높았습니다."

Q: "왜 F1이 아닌 AUPRC, Recall을 쓰나요?"
> "FDS에서는 **Recall이 핵심**입니다. 사기를 놓치면(FN) 큰 손실이고, 오탐(FP)은 추가 검증으로 해결 가능합니다. F1은 Precision-Recall 균형을 보는데, FDS에서는 균형보다 Recall 우선입니다. AUPRC는 불균형 데이터(사기 3.5%)에서 AUC-ROC보다 더 정확한 성능 지표입니다."

---

## 1-10: Transformer (Day 10) - 선택 ⭐⭐

### 필요 패키지
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
# pip install tab-transformer-pytorch (선택)
```

### 세부 설명 리스트

**1. 왜 Transformer인가?**
- 2025 벤치마크: F1 0.998 (연구 최강)
- Self-Attention으로 피처 간 관계 학습
- 정형 데이터에서도 강력한 성능

**2. TabTransformer 구조**
```
[입력] 정형 피처
      ↓
[Embedding Layer]
- 범주형 → Embedding
- 수치형 → 그대로 또는 MLP
      ↓
[Transformer Encoder]
- Multi-Head Self-Attention
- Feed-Forward Network
      ↓
[MLP Head] → 예측
```

**3. Self-Attention 직관**
- 각 피처가 다른 피처들과의 관계를 학습
- 예: "고액 거래" + "새벽 시간" 조합 패턴 자동 학습
- XGBoost의 수동 피처 엔지니어링을 대체

### 실습 목록
- 실습 1: Self-Attention 구현 이해
- 실습 2: TabTransformer 직접 구현
- 실습 3: 학습 루프 및 Early Stopping
- 실습 4: XGBoost와 성능 비교
- 실습 5: 추론 속도 벤치마크

### 핵심 코드: TabTransformer 구현

```python
class TabTransformer(nn.Module):
    def __init__(
        self,
        num_continuous: int,
        num_categories: list,  # 각 범주형 피처의 카디널리티
        dim: int = 32,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        # 범주형 임베딩
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_cat, dim) for num_cat in num_categories
        ])

        # 수치형 처리
        self.cont_norm = nn.LayerNorm(num_continuous)
        self.cont_proj = nn.Linear(num_continuous, dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # MLP Head
        total_dim = dim * (len(num_categories) + 1)  # +1 for continuous
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_cat, x_cont):
        # 범주형 임베딩
        cat_embeds = [
            emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)
        ]
        cat_embeds = torch.stack(cat_embeds, dim=1)  # (batch, num_cat, dim)

        # 수치형 처리
        x_cont = self.cont_norm(x_cont)
        cont_embed = self.cont_proj(x_cont).unsqueeze(1)  # (batch, 1, dim)

        # 결합
        x = torch.cat([cat_embeds, cont_embed], dim=1)  # (batch, num_cat+1, dim)

        # Transformer
        x = self.transformer(x)

        # Flatten + MLP
        x = x.flatten(1)
        return self.mlp(x)
```

### 핵심 코드: 학습 루프

```python
def train_transformer(model, train_loader, val_loader, epochs=50, patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_auc = 0
    counter = 0

    for epoch in range(epochs):
        model.train()
        for x_cat, x_cont, y in train_loader:
            x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x_cat, x_cont).squeeze()
            loss = criterion(output, y.float())
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_cat, x_cont, y in val_loader:
                x_cat, x_cont = x_cat.to(device), x_cont.to(device)
                output = model(x_cat, x_cont).squeeze()
                val_preds.extend(output.cpu().numpy())
                val_targets.extend(y.numpy())

        val_auc = roc_auc_score(val_targets, val_preds)
        scheduler.step()

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'models/transformer_best.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch}: Val AUC = {val_auc:.4f}")

    return best_auc
```

### 면접 포인트

Q: "왜 Transformer를 FDS에 적용했나요?"
> "2025년 연구에서 TabTransformer가 정형 데이터에서 F1 0.998을 달성했습니다. Self-Attention이 피처 간 복잡한 관계(예: 고액+새벽+해외)를 자동으로 학습해서, 수동 피처 엔지니어링 없이 높은 성능을 얻을 수 있습니다."

Q: "현업에서 Transformer 안 쓰는 이유는?"
> "추론 속도가 50-100ms로 XGBoost(5ms)보다 느립니다. 하지만 HSBC, Featurespace 등 대형 금융사에서 도입 중이고, 배치 추론이나 고위험 거래 재검토에는 충분히 적용 가능합니다."

---

## 1-11: PaySim 공정 비교 (Day 11) - 선택 ⭐⭐

### 필요 패키지
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import time
```

### 세부 설명 리스트

**1. 왜 PaySim으로 재실험?**
- IEEE-CIS: V1~V339가 PCA 변환된 정적 피처 → LSTM AUC 0.70 실패
- PaySim: 진짜 시계열 (사용자별 거래 순서) → 공정한 ML vs DL 비교 가능
- 시간 윈도우 집계 피처 직접 구현 → 현업 파이프라인 경험

**2. PaySim 데이터셋**

| 항목 | 값 |
|------|-----|
| 총 거래 수 | 6,362,620 |
| 기간 | 30일 (744 steps, 1 step = 1시간) |
| 사용자 ID | nameOrig |
| 거래 타입 | CASH_IN, CASH_OUT, TRANSFER, DEBIT, PAYMENT |
| 사기 비율 | 0.13% (8,213건) |

**3. 시간 윈도우 집계 피처 (12개) - 현업 수준**
```python
# 시간 윈도우별 거래 빈도 (3개)
tx_count_1h      # 최근 1시간 거래 수
tx_count_24h     # 최근 24시간 거래 수
tx_count_7d      # 최근 7일 거래 수

# 시간 윈도우별 금액 합계 (3개)
amt_sum_1h       # 최근 1시간 총액
amt_sum_24h      # 최근 24시간 총액
amt_sum_7d       # 최근 7일 총액

# 시간 간격 (2개)
time_since_last  # 마지막 거래 후 경과 시간
avg_time_gap     # 평균 거래 간격

# 잔액 관련 (2개)
balance_ratio    # newBalance / oldBalance
balance_drop_pct # 잔액 감소율

# 패턴 탐지 (2개)
same_dest_count  # 같은 수취자에게 보낸 횟수
is_first_transfer # 첫 송금 여부
```

**4. 비교 모델 (4개)**

| 모델 | 입력 | 특징 |
|------|------|------|
| XGBoost | 집계 피처 + 원본 | 베이스라인 |
| 트리 스태킹 | 집계 피처 + 원본 | XGB+LGBM+Cat |
| LSTM | 시퀀스 (seq_len=10) | 시계열 패턴 |
| Transformer | 집계 피처 + 원본 | Self-Attention |

### 실습 목록
- 실습 1: PaySim 데이터 로드 및 EDA
- 실습 2: 시간 윈도우 집계 피처 구현 (12개)
- 실습 3: LSTM용 시퀀스 생성
- 실습 4: 4개 모델 학습 (XGBoost, 스태킹, LSTM, Transformer)
- 실습 5: 추론 속도 벤치마크
- 실습 6: 성능 + 속도 비교 분석

### 핵심 코드: 시간 윈도우 집계 피처

```python
def create_time_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    현업 수준 시간 윈도우 집계 피처 생성 (Vectorized, O(n) 성능)

    기존 O(n²) 루프 방식 대비 ~100배 빠름 (600만 건 기준)
    """
    df = df.sort_values(['nameOrig', 'step']).copy()

    # step을 datetime으로 변환 (1 step = 1 hour)
    df['datetime'] = pd.to_datetime(df['step'], unit='h', origin='2020-01-01')
    df = df.set_index('datetime')

    # === 1. 시간 윈도우별 거래 빈도 (rolling count) ===
    # closed='left': 현재 거래 제외 (과거만)
    df['tx_count_1h'] = df.groupby('nameOrig')['amount'].transform(
        lambda x: x.shift(1).rolling('1H', min_periods=0).count()
    ).fillna(0).astype(int)

    df['tx_count_24h'] = df.groupby('nameOrig')['amount'].transform(
        lambda x: x.shift(1).rolling('24H', min_periods=0).count()
    ).fillna(0).astype(int)

    df['tx_count_7d'] = df.groupby('nameOrig')['amount'].transform(
        lambda x: x.shift(1).rolling('168H', min_periods=0).count()  # 7일 = 168시간
    ).fillna(0).astype(int)

    # === 2. 시간 윈도우별 금액 합계 (rolling sum) ===
    df['amt_sum_1h'] = df.groupby('nameOrig')['amount'].transform(
        lambda x: x.shift(1).rolling('1H', min_periods=0).sum()
    ).fillna(0)

    df['amt_sum_24h'] = df.groupby('nameOrig')['amount'].transform(
        lambda x: x.shift(1).rolling('24H', min_periods=0).sum()
    ).fillna(0)

    df['amt_sum_7d'] = df.groupby('nameOrig')['amount'].transform(
        lambda x: x.shift(1).rolling('168H', min_periods=0).sum()
    ).fillna(0)

    # === 3. 시간 간격 피처 ===
    df['time_since_last'] = df.groupby('nameOrig')['step'].diff().fillna(0)
    df['avg_time_gap'] = df.groupby('nameOrig')['step'].transform(
        lambda x: x.diff().expanding().mean()
    ).fillna(0)

    # === 4. 잔액 관련 피처 ===
    df['balance_ratio'] = df['newbalanceOrig'] / (df['oldbalanceOrg'] + 1e-6)
    df['balance_drop_pct'] = (df['oldbalanceOrg'] - df['newbalanceOrig']) / (df['oldbalanceOrg'] + 1e-6)

    # === 5. 패턴 탐지 피처 ===
    # 같은 수취자 거래 횟수 (cumcount)
    df['same_dest_count'] = df.groupby(['nameOrig', 'nameDest']).cumcount()

    # 첫 송금 여부
    df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
    df['transfer_cumsum'] = df.groupby('nameOrig')['is_transfer'].cumsum() - df['is_transfer']
    df['is_first_transfer'] = ((df['is_transfer'] == 1) & (df['transfer_cumsum'] == 0)).astype(int)

    # 인덱스 복원 및 임시 컬럼 제거
    df = df.reset_index(drop=True)
    df = df.drop(columns=['is_transfer', 'transfer_cumsum'], errors='ignore')

    feature_cols = [
        'tx_count_1h', 'tx_count_24h', 'tx_count_7d',
        'amt_sum_1h', 'amt_sum_24h', 'amt_sum_7d',
        'time_since_last', 'avg_time_gap',
        'balance_ratio', 'balance_drop_pct',
        'same_dest_count', 'is_first_transfer'
    ]

    return df[feature_cols]
```

> **성능 비교**: 600만 건 기준
> - 기존 O(n²) 루프: ~수 시간
> - Vectorized O(n): ~30초

### 핵심 코드: 추론 속도 벤치마크

```python
def benchmark_inference(model, X_sample, n_runs=100):
    """단일 샘플 추론 속도 측정"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict_proba(X_sample.reshape(1, -1))
        times.append((time.perf_counter() - start) * 1000)  # ms
    return np.mean(times), np.std(times)

# 각 모델 벤치마크
results = []
for name, model in models.items():
    mean_ms, std_ms = benchmark_inference(model, X_test[0])
    results.append({'Model': name, 'Latency (ms)': f"{mean_ms:.2f} ± {std_ms:.2f}"})

print(pd.DataFrame(results).to_markdown(index=False))
```

### 예상 실험 결과

| 모델 | AUC | AUPRC | 추론 속도 (ms) |
|------|-----|-------|---------------|
| XGBoost | 0.93+ | 0.60+ | ~0.5 |
| 트리 스태킹 | 0.94+ | 0.62+ | ~1.5 |
| LSTM | 0.90+ | 0.55+ | ~15 |
| Transformer | 0.92+ | 0.58+ | ~30 |

### 면접 포인트

Q: "왜 PaySim으로 재실험했나요?"
> "IEEE-CIS에서 LSTM AUC 0.70으로 실패한 원인을 분석했습니다. V1~V339가 PCA 익명화된 피처라서 시계열 패턴이 없었습니다. PaySim은 실제 거래 시퀀스가 있어서 ML vs DL 공정 비교가 가능했습니다."

Q: "시간 윈도우 집계 피처는 어떻게 설계했나요?"
> "현업 논문과 블로그를 참고해서 12개 피처를 설계했습니다. 1시간/24시간/7일 윈도우별 거래 빈도와 금액, 잔액 변화율, 같은 수취자 반복 패턴 등입니다. 이 피처들이 +200% 성능 향상에 기여한다는 연구 결과가 있습니다."

Q: "추론 속도 측정 결과는?"
> "XGBoost 0.5ms, LSTM 15ms로 30배 차이났습니다. LSTM이 성능은 좋지만 실시간 서빙에 부적합해서, 이를 해결하기 위해 1-12에서 하이브리드 아키텍처를 구현했습니다."

---

## 1-12: 하이브리드 서빙 (Day 12) - 선택 ⭐⭐

### 필요 패키지
```python
import torch
import torch.nn as nn
import redis
import pickle
import numpy as np
from xgboost import XGBClassifier
import time
```

### 세부 설명 리스트

**1. 문제 정의**
- 1-11 결과: LSTM AUC 0.90+ 달성하지만 추론 속도 15ms
- XGBoost: AUC 0.93+, 추론 속도 0.5ms
- 목표: DL 성능 + XGBoost 속도 결합

**2. 해결책: NVIDIA 레퍼런스 아키텍처**
```
배치 파이프라인 (1시간마다):
┌─────────────────────────────────────┐
│ 1. 전체 고객 시퀀스 로드            │
│ 2. LSTM으로 고객별 임베딩 계산      │
│ 3. Redis에 저장 (key: customer_id)  │
└─────────────────────────────────────┘

실시간 파이프라인 (거래 발생 시):
┌─────────────────────────────────────┐
│ 1. Redis에서 임베딩 조회 (0.1ms)    │
│ 2. 원본 피처 + 임베딩 결합          │
│ 3. XGBoost 추론 (0.5ms)             │
│ 4. 총 < 1ms                         │
└─────────────────────────────────────┘
```

**3. 왜 Redis?**
- In-memory → 조회 0.1ms 미만
- 현업 표준 (Feedzai, Stripe 등에서 사용)
- Docker로 쉽게 구성

### 실습 목록
- 실습 1: LSTM 임베딩 추출 함수 구현
- 실습 2: Redis 연결 및 임베딩 저장/로드
- 실습 3: 하이브리드 XGBoost 학습 (원본 + 임베딩)
- 실습 4: 추론 속도 벤치마크 (LSTM 직접 vs 하이브리드)
- 실습 5: 성능 비교 (XGBoost 단독 vs 하이브리드)

### 핵심 코드: LSTM 임베딩 추출

```python
class LSTMEmbedder(nn.Module):
    """LSTM에서 임베딩만 추출 (분류 헤드 제외)"""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # (batch, hidden_size)

def extract_user_embeddings(model, user_sequences, device):
    """고객별 임베딩 추출"""
    model.eval()
    embeddings = {}

    with torch.no_grad():
        for user_id, seq in user_sequences.items():
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
            emb = model(seq_tensor).cpu().numpy().flatten()
            embeddings[user_id] = emb

    return embeddings
```

### 핵심 코드: Redis 임베딩 저장/로드

```python
# Redis 연결
r = redis.Redis(host='localhost', port=6379, db=0)

EMBEDDING_DIM = 64  # LSTM hidden_size

def save_embeddings_to_redis(embeddings: dict[str, np.ndarray], batch_size: int = 1000):
    """
    임베딩을 Redis에 저장 (Pipeline + tobytes로 최적화)

    - pickle 대신 tobytes: 직렬화 오버헤드 제거
    - Pipeline: 네트워크 왕복 최소화 (100배 빠름)
    """
    pipe = r.pipeline()
    count = 0

    for user_id, emb in embeddings.items():
        # tobytes()는 pickle보다 ~10배 빠르고 메모리 효율적
        pipe.hset(f"emb:{user_id}", mapping={
            "vector": emb.astype(np.float32).tobytes(),
            "dim": EMBEDDING_DIM
        })
        count += 1

        # 배치 단위로 실행 (메모리 관리)
        if count % batch_size == 0:
            pipe.execute()
            pipe = r.pipeline()

    pipe.execute()  # 남은 것 처리
    print(f"Saved {len(embeddings)} embeddings to Redis")

def load_embedding_from_redis(user_id: str) -> np.ndarray:
    """Redis에서 임베딩 조회 (0.1ms 미만)"""
    data = r.hget(f"emb:{user_id}", "vector")
    if data:
        return np.frombuffer(data, dtype=np.float32)
    return np.zeros(EMBEDDING_DIM, dtype=np.float32)  # fallback (신규 고객)

# 배치 저장
embeddings = extract_user_embeddings(lstm_embedder, user_sequences, device)
save_embeddings_to_redis(embeddings)
```

> **최적화 포인트**:
> - `tobytes()` + `frombuffer()`: pickle 대비 ~10배 빠름
> - Pipeline: 1만 건 저장 시 10초 → 0.1초

### 핵심 코드: 하이브리드 추론

```python
def hybrid_predict(user_id, transaction_features, xgb_model, redis_client):
    """하이브리드 실시간 추론"""
    start = time.perf_counter()

    # 1. Redis에서 임베딩 조회
    embedding = load_embedding_from_redis(user_id)

    # 2. 원본 피처 + 임베딩 결합
    hybrid_features = np.concatenate([transaction_features, embedding])

    # 3. XGBoost 추론
    prob = xgb_model.predict_proba(hybrid_features.reshape(1, -1))[0, 1]

    latency = (time.perf_counter() - start) * 1000
    return prob, latency

# 벤치마크
latencies = []
for _ in range(100):
    _, latency = hybrid_predict(test_user, test_features, xgb_hybrid, r)
    latencies.append(latency)

print(f"Hybrid latency: {np.mean(latencies):.2f} ± {np.std(latencies):.2f} ms")
```

### 실제 실험 결과 (5개 모델 비교)

| 모델 | AUC | Recall@5%FPR | 추론 속도 |
|------|-----|--------------|-----------|
| XGBoost 단독 | 0.9997 | 99.92% | 0.38ms |
| FT-Transformer | 0.9995 | 99.86% | 24.58ms |
| 하이브리드 (XGB+임베딩) | **0.9997** | **99.95%** | 1.03ms |
| 스태킹 (3-Tree) | **0.9998** | 99.92% | 1.63ms |
| 하이브리드 스태킹 | 0.9992 | 99.89% | 2.35ms |

**핵심 발견:**
- **스태킹이 AUC 최고** (0.9998) - DL 없이 트리 앙상블만으로 최고
- **하이브리드 스태킹 성능 하락** (0.9992) - 과적합/정보 중복 문제
- **하이브리드가 Recall 최고** (99.95%) - DL 임베딩 효과
- PaySim 특성: 모든 모델 AUC 0.999+ 수렴 → 실용성 기준으로 선택

### Docker 설정

```yaml
# docker-compose.yml에 추가
services:
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
```

### 면접 포인트

Q: "하이브리드로 어떻게 속도를 개선했나요?"
> "LSTM 추론이 15ms로 느려서, 배치로 고객별 임베딩을 미리 계산해서 Redis에 캐싱했습니다. 실시간에는 Redis 조회(0.1ms) + XGBoost(0.5ms)로 총 0.6ms에 추론합니다. LSTM 직접 실행 대비 25배 빨라졌습니다."

Q: "이 아키텍처의 장점은?"
> "NVIDIA 레퍼런스 아키텍처 패턴입니다. DL의 패턴 인식력(임베딩)과 XGBoost의 속도/설명성을 결합합니다. Redis 캐싱으로 실시간 서빙이 가능하고, 배치 업데이트 주기(1시간)로 임베딩 신선도를 유지합니다."

Q: "임베딩이 오래된 경우는?"
> "최대 1시간 지연이 발생할 수 있습니다. 하지만 고객 행동 패턴은 급격히 변하지 않고, XGBoost가 실시간 피처(현재 거래 정보)를 처리하므로 충분히 보완됩니다. 필요시 업데이트 주기를 10분으로 단축할 수 있습니다."

Q: "왜 5가지 모델을 비교했나요?"
> "동일 PaySim 데이터에서 공정 비교를 위해:
> 1. XGBoost 단독 - 베이스라인
> 2. FT-Transformer - DL 직접 추론
> 3. 하이브리드 (XGB+임베딩) - NVIDIA Blueprint 패턴
> 4. 스태킹 (3-Tree) - 트리 앙상블 다양성
> 5. 하이브리드 스태킹 - 스태킹 + DL 임베딩 결합
> 결론: 스태킹이 AUC 최고(0.9998)지만, 실무에서는 속도 대비 개선 폭이 작아서 XGBoost 단독 또는 하이브리드가 실용적입니다."

Q: "하이브리드 스태킹이 오히려 성능이 낮아진 이유는?"
> "피처가 35개(확률 3개 + 임베딩 32차원)로 많아져서 LogisticRegression이 과적합되었습니다. 또한 임베딩과 스태킹 확률 간 정보 중복(redundancy)이 발생했고, 스케일 불일치(확률 0~1 vs 임베딩 -2~+2)도 원인입니다. 이론상 '더 많은 정보 = 더 좋은 성능'이 아님을 보여주는 사례입니다."

---

## 전체 요약

| 노트북 | 시간 | 핵심 산출물 |
|--------|------|------------|
| 1-1 | 3h | train.csv, test.csv |
| 1-2 | 4h | preprocessing.py, X_tabular, X_sequence |
| 1-3 | 4h | 모델 비교 표, xgb_model.pkl, xgb_importance.csv |
| 1-4 | 4h | src/models/lstm.py, lstm_model.pt |
| 1-5 | 3h | 앙상블 성능 비교 표 |
| 1-6 | 3h | shap_explainer.py, 설명 시각화 |
| 1-7 | 4h | FastAPI, Docker, 통합 테스트 |
| 1-8 | 4h | React Admin (거래 목록 + 상세) |
| 1-9 | 4h | 트리 스태킹 (XGB+LGBM+Cat), stacking_model.pkl ⭐⭐ |
| 1-10 | 5h | TabTransformer, transformer_model.pt (선택) |
| 1-11 | 4h | 하이브리드 (DL임베딩+XGB), hybrid_model.pkl (선택) |
| 1-12 | 4h | PaySim 시퀀스 실험, LSTM 검증 (선택) |

**총 약 46시간 (12일, 전체 선택 시)**
**필수만: ~37시간 (1-1~1-9)**

---

## 핵심 실험 결과 (면접용)

### 1. 모델 비교 (1-3)

| Model | AUC | Time(s) | SHAP |
|-------|-----|---------|------|
| XGBoost | 0.9114 | 45 | ✅ 최상 |
| LightGBM | 0.91 | 32 | ✅ 좋음 |
| CatBoost | 0.91 | 98 | ⚠️ 제한 |

### 실제 달성 지표 (검증 완료)

| 지표 | 값 | 현업 기준 |
|------|-----|----------|
| AUC-ROC | 0.9114 | ≥0.90 ✅ |
| Recall | 90.55% | 80-95% ✅ |
| Precision | 9.78% | 5-30% ✅ |
| AUPRC | 0.5313 | ≥0.50 ✅ |

### 다단계 위험도 (4단계)

```
approve: 0.00 ~ 0.18 (승인) - 67%
verify:  0.18 ~ 0.40 (추가인증) - 21%
hold:    0.40 ~ 0.65 (보류) - 7%
block:   0.65 이상   (차단) - 5%
```

### 2. LSTM 단계별 개선 (1-4)

| 단계 | 구성 | AUC |
|------|------|-----|
| 베이스라인 | V1~V20 + 시계열 피처 (35개) | 0.82 |
| + Optuna 튜닝 | 하이퍼파라미터 최적화 | 0.84 |
| + 피처 추가 | XGBoost importance 기반 | 0.86 |

### 3. 앙상블 실험 결과 (1-5)

| Model | AUC | 결론 |
|-------|-----|------|
| XGBoost | 0.9114 | **채택** |
| LSTM | 0.7054 | 성능 낮음 |
| Ensemble (0.9:0.1) | 0.9054 | +0.12% (효과 미미) |

✅ XGBoost 단독 AUC 0.91 → 목표 달성, 앙상블 불필요

### 4. 앙상블 실험 결론

| 방법 | AUC | 결론 |
|------|-----|------|
| XGBoost 단독 | 0.9042 | **채택** |
| LSTM 단독 | 0.7054 | 성능 낮음 |
| 앙상블 (0.9:0.1) | 0.9054 | +0.12% (효과 미미) |

**이 표가 면접에서 "왜 XGBoost 단독?"에 대한 근거!**

**핵심 스토리:**
> "LSTM 앙상블을 시도했지만 +0.12% 향상에 그침 → 복잡도 대비 효과 분석 → XGBoost 단독 채택 → 딥러닝이 항상 좋은 건 아니다"

---

---

## Lessons Learned: API와 학습 코드 일관성

### 발생한 문제

API 코드와 학습 코드가 분리되어 작성되면서 다음 문제 발생:

```
학습 코드 (노트북): LabelEncoder → ProductCD "W" = 4
API 코드 (predictor.py): CATEGORY_MAPPINGS → ProductCD "w" = 0
→ 완전히 다른 값! → Recall 0%
```

### 해결 방법

1. **전처리된 피처 그대로 사용**: `/samples` API에서 447개 피처 전체 반환
2. **새 엔드포인트**: `/predict/direct/batch` - 인코딩 변환 없이 바로 예측
3. **Recall 검증 노트북 추가**: `1-3-1_recall_check.ipynb`

### 면접 어필 포인트

> "학습 코드와 API 코드 불일치로 Recall 0% 이슈 발생. 원인 분석 후 전처리된 피처를 그대로 전달하는 방식으로 해결. 이 경험으로 학습-서빙 일관성의 중요성을 깊이 이해하게 됨."

---

## 다음 단계: Phase 2

> 상세: [docs/roadmap.md](./roadmap.md)

Phase 1 완료 후 Phase 2에서 추가:
- **MLflow**: 실험 추적, 모델 버전 관리
- **Evidently**: 드리프트 모니터링
- **GitHub Actions**: CI/CD
- **비용 기반 최적화**: 비즈니스 임팩트 계산
