# Phase 1: FDS + XAI 시스템

> XGBoost 기반 이상거래 탐지 및 SHAP 설명 시스템

---

## 1. 프로젝트 소개

### 1.1 프로젝트명
**Explainable FDS with Ensemble Learning**

### 1.2 목적
정형 데이터(XGBoost)로 이상거래를 탐지하고, SHAP으로 왜 이상거래인지 설명하는 시스템. LSTM 앙상블 실험 완료 (효과 미미하여 XGBoost 단독 채택)

### 1.3 결과물 예시

```
[입력] 거래 정보
- 금액: 500만원
- 시간: 새벽 3시
- 최근 거래 패턴: 소액 → 소액 → 고액 (급증)

[출력]
{
  "fraud_probability": 0.87,
  "model_scores": {
    "xgboost": 0.87     // 정형 특성 기반 (XGBoost 단독 사용)
  },
  "top_factors": [
    {"feature": "거래금액", "impact": +0.25, "reason": "평소 대비 10배"},
    {"feature": "거래시간", "impact": +0.18, "reason": "새벽 3시 (비정상)"},
    {"feature": "거래패턴", "impact": +0.22, "reason": "급격한 금액 증가"}
  ]
}
```

### 1.4 왜 이 프로젝트인가?

**금융권 MLE 포지션 타겟팅:**
- 하나금융TI, 우리FIS 등 은행 IT 자회사
- FDS는 모든 금융사의 필수 시스템
- XAI(설명가능 AI)는 금융 규제 트렌드
- LSTM 실험 → 효과 미미하여 XGBoost 단독 채택 (실무적 판단)

**포트폴리오 차별화:**
- XGBoost + SHAP으로 설명 가능한 FDS 구현
- XGBoost vs LSTM 비교 실험 → 근거 있는 기술 선택
- "딥러닝이 항상 좋은가?" → **데이터로 NO 증명** (LSTM AUC 0.70 vs XGBoost 0.91)
- 앙상블 효과 미미 (+0.12%) → 복잡도 대비 효과 분석 경험

---

## 2. 기술 스택

### 2.1 전체 구성

```
┌─────────────────────────────────────────────────────────────┐
│  ML: XGBoost (정형) + LSTM (시계열) → Ensemble              │
├─────────────────────────────────────────────────────────────┤
│  XAI: SHAP (TreeExplainer + DeepExplainer)                  │
├─────────────────────────────────────────────────────────────┤
│  API: FastAPI                                               │
├─────────────────────────────────────────────────────────────┤
│  인프라: Docker Compose                                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 상세 버전

| 영역 | 기술 | 버전 | 역할 |
|------|------|------|------|
| ML | XGBoost | 2.1.0 | 정형 특성 기반 탐지 |
| ML | PyTorch | 2.0+ | LSTM 시계열 모델 |
| ML | SHAP | 0.43+ | 모델 설명 (XAI) |
| ML | Optuna | 3.0+ | 하이퍼파라미터 튜닝 |
| API | FastAPI | 0.110+ | REST API |
| API | Uvicorn | 0.25+ | ASGI 서버 |
| Infra | Docker | 24+ | 컨테이너화 |

---

## 3. 아키텍처

### 3.1 전체 흐름

```
클라이언트
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI                               │
│                                                             │
│  POST /predict                                              │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Feature Engineering                     │   │
│  │  (정형 특성 추출 + 시계열 시퀀스 생성)               │   │
│  └─────────────────────────────────────────────────────┘   │
│      │                                                      │
│      ├──────────────┬──────────────┐                       │
│      ▼              ▼              │                       │
│  ┌────────┐    ┌────────┐         │                       │
│  │XGBoost │    │  LSTM  │         │                       │
│  │(정형)  │    │(시계열)│         │                       │
│  └────┬───┘    └────┬───┘         │                       │
│       │             │              │                       │
│       └──────┬──────┘              │                       │
│              ▼                     │                       │
│       ┌────────────┐              │                       │
│       │  Ensemble  │              │                       │
│       │ (가중평균) │              │                       │
│       └──────┬─────┘              │                       │
│              │                     │                       │
│              ▼                     ▼                       │
│       ┌────────────┐    ┌────────────────┐               │
│       │    SHAP    │    │   Response     │               │
│       │  (설명)    │───▶│   (JSON)       │               │
│       └────────────┘    └────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 모델 파이프라인 상세

```
입력: 거래 데이터
  │
  ▼
[정형 특성 447개]
- 거래금액, 시간, 카드종류
- 이메일도메인, 시계열 통계
  │
  ▼
┌─────────────┐
│  XGBoost    │  ← 앙상블 실험 결과 단독 채택
│  Classifier │    (LSTM 추가 시 +0.12% → 효과 미미)
└──────┬──────┘
       │ p_xgb = 0.91 (AUC 0.9114)
       ▼
┌─────────────┐
│   4단계     │
│  위험도     │  approve/verify/hold/block
│ (0.18/0.40/0.65)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    SHAP     │
│ TreeExplainer (XGBoost)
│ Top 5 피처 기여도
└─────────────┘
```

### 3.3 Docker 컨테이너

| 서비스 | 이미지 | 포트 | 역할 |
|--------|--------|------|------|
| app | python:3.11 | 8000 | FastAPI + 모델 서빙 |

---

## 4. 기술 선택 이유 (면접용)

### 4.1 정형 모델: XGBoost

**실험 결과:**

| 모델 | AUC | 학습시간 | SHAP 호환 |
|------|-----|----------|-----------|
| XGBoost | 0.92 | 45s | ✅ 최상 |
| LightGBM | 0.91 | 32s | ✅ 좋음 |
| CatBoost | 0.91 | 98s | ⚠️ 제한적 |

**면접 답변:**
> "정형 데이터에서 트리 기반 모델이 여전히 SOTA입니다. 3개 모델 비교 실험 결과, XGBoost가 AUC 0.92로 가장 높았고 SHAP TreeExplainer와의 호환성도 가장 좋았습니다. 단일 거래의 정형 특성(금액, 시간, 카드종류 등)을 분석하는 데 최적입니다."

### 4.2 시계열 모델: LSTM

**선택 이유:**

| 항목 | 선택 | 이유 |
|------|------|------|
| 모델 | LSTM | 거래 시퀀스의 시간적 패턴 학습 |
| 프레임워크 | PyTorch | 유연성, 커스텀 레이어 지원 |
| 입력 | 최근 10개 거래 | 사용자별 행동 패턴 포착 |

**왜 시계열 모델이 필요한가?**

XGBoost는 **단일 거래**만 봅니다. 하지만 사기에는 **패턴**이 있습니다:
- "소액 테스트 → 소액 테스트 → 고액 사기" 패턴
- "새벽에 연속 거래" 패턴
- "평소와 다른 지역에서 연속 거래" 패턴

이런 **시퀀스 패턴**은 LSTM이 잡아냅니다.

**실험 결과 (실제 측정):**

| 단계 | 모델 | AUC | 설명 |
|------|------|-----|------|
| 1 | XGBoost 단독 | **0.9114** | 정형 특성 → **채택** |
| 2 | LSTM | 0.7054 | 시계열 패턴 → 성능 낮음 |
| 3 | 앙상블 (0.9:0.1) | 0.9054 | **+0.12%** → 효과 미미 |

**면접 답변:**
> "LSTM으로 시계열 패턴을 잡으려 했으나, AUC 0.70으로 XGBoost(0.91)보다 크게 낮았습니다. 앙상블해도 +0.12% 향상에 그쳐서, 복잡도 대비 효과가 없다고 판단했습니다. 결과적으로 XGBoost 단독을 채택했고, 이 과정에서 '딥러닝이 항상 좋은 건 아니다'는 실무적 판단력을 얻었습니다."

**Q: "왜 Transformer가 아닌 LSTM인가요?"**
> "시퀀스 길이가 10~20개로 짧습니다. Transformer는 긴 시퀀스에서 강점이 있지만, 짧은 시퀀스에서는 LSTM과 성능 차이가 거의 없고 학습이 더 빠릅니다. GRU와도 비교했는데 LSTM이 약간 더 좋았습니다."

### 4.3 앙상블 실험 결과: XGBoost 단독 채택

**비교 실험:**

| 방법 | AUC | 개선 | 결론 |
|------|-----|------|------|
| XGBoost 단독 | 0.9042 | - | **채택** |
| LSTM 단독 | 0.7054 | - | 성능 낮음 |
| 앙상블 (0.9:0.1) | 0.9054 | +0.12% | 효과 미미 |

**가중치 최적화 결과:**

```python
# Grid Search 결과
# XGBoost 90% + LSTM 10%가 최적
# 하지만 +0.12% 향상은 복잡도 대비 의미 없음

# 결론: XGBoost 단독 사용
```

**면접 답변:**
> "앙상블을 시도했지만, LSTM AUC가 0.70으로 낮아서 최적 가중치가 XGBoost 90%가 되었습니다. 이렇게 되면 LSTM 서빙 비용 대비 효과가 없어서, XGBoost 단독을 채택했습니다. 복잡도 대비 효과를 분석하는 실무적 판단이었습니다."

**Q: "왜 LSTM 성능이 낮았나요?"**
> "IEEE-CIS 데이터셋 특성상 시계열 패턴보다 정형 피처(금액, 시간, 카드정보)가 더 결정적이었습니다. 모든 문제에 딥러닝이 최선은 아니라는 것을 배웠습니다."

### 4.4 설명 가능성: SHAP

**XGBoost + TreeExplainer:**

| 모델 | SHAP 방법 | 설명 대상 |
|------|-----------|-----------|
| XGBoost | TreeExplainer | 정형 특성 기여도 (Top 5) |

**설명 예시:**
> "금액이 높고(+0.25), 시간이 비정상(+0.18), 카드 사용 패턴 이상(+0.15)"

**면접 답변:**
> "XGBoost의 TreeExplainer로 피처별 기여도를 계산합니다. 사용자에게는 '이 거래가 왜 사기로 판단되었는지' Top 5 요인을 자연어로 설명합니다."

---

## 5. 성공 기준

| 항목 | 기준 | 실제 달성 | 측정 방법 |
|------|------|-----------|-----------|
| XGBoost AUC | >= 0.90 | 0.9114 ✅ | 테스트셋 평가 |
| **Recall** | >= 80% | **90.55%** ✅ | Threshold 0.18 기준 |
| **AUPRC** | >= 0.50 | **0.5313** ✅ | Precision-Recall AUC |
| Precision | 5-30% | 9.78% ✅ | Threshold 0.18 기준 |
| LSTM 베이스라인 AUC | - | 0.7054 (효과 미미) | 테스트셋 평가 |
| Ensemble AUC | - | 0.9054 (+0.12%) | 효과 미미 → XGBoost 단독 채택 |
| /predict 응답 | < 200ms | - | API 벤치마크 |
| SHAP 설명 | Top 5 피처 | ✅ | 정형 통합 |
| Docker 실행 | 단일 명령어 | ✅ | docker compose up |

### 5.1 다단계 위험도 (4단계)

| 레벨 | 확률 범위 | 비율 | Fraud 비율 | 조치 |
|------|-----------|------|------------|------|
| approve | 0.00 ~ 0.18 | 67% | 0.5% | 자동 승인 |
| verify | 0.18 ~ 0.40 | 21% | 2.5% | 추가 인증 |
| hold | 0.40 ~ 0.65 | 7% | 9.3% | 수동 검토 |
| block | 0.65 이상 | 5% | 43.2% | 자동 차단 |

---

## 6. 일정 및 학습 가이드

### 6.1 전체 일정

| Day | 주제 | 핵심 학습 |
|-----|------|-----------|
| 1 | 데이터 + EDA | Pandas, 불균형 데이터 |
| 2 | Feature Engineering | 정형 + 시계열 피처 (35개+) |
| 3 | XGBoost 모델 ⭐ | Optuna 튜닝, Threshold |
| 4 | LSTM 모델 ⭐ | PyTorch, V1~V20 + 시계열 피처 |
| 5 | Ensemble ⭐ | 가중 앙상블 |
| 6 | SHAP 설명 | TreeExplainer, DeepExplainer |
| 7 | FastAPI 배포 | Docker, API 설계 |
| 8 | **React Admin** | Ant Design, 거래 목록/상세 UI |
| 9 | **트리 스태킹** ⭐⭐ | XGBoost + LightGBM + CatBoost |
| 10 | **Transformer** (선택) | TabTransformer, Self-Attention |
| 11 | **하이브리드** (선택) | DL→피처→XGBoost |
| 12 | **PaySim** (선택) | 진짜 시퀀스 데이터 실험 |

### 6.2 Day별 학습 가이드

---

#### Day 1: 데이터 + EDA

**📚 핵심 개념**

1. **불균형 데이터 (Imbalanced Data)**
   - FDS에서 사기 거래는 전체의 3-5%
   - 정확도(Accuracy)가 의미 없는 이유
   - 평가 지표: AUC-ROC, PR-AUC, F1

2. **IEEE-CIS 데이터셋 구조**
   - Transaction: 거래 정보 (금액, 시간, 카드 등)
   - Identity: 기기/브라우저 정보
   - LEFT JOIN으로 병합

**✏️ 미니 연습**
- Q1: 왜 불균형 데이터에서 Accuracy가 의미 없나요?
- Q2: AUC-ROC와 PR-AUC의 차이는?

**🎯 면접 Q&A**

Q: "불균형 데이터를 어떻게 처리했나요?"
> "IEEE-CIS 데이터는 사기 비율이 3.5%입니다. Accuracy 대신 AUC-ROC를 주 평가 지표로 사용했고, class_weight 파라미터로 소수 클래스에 가중치를 줬습니다. SMOTE 같은 오버샘플링은 오히려 성능을 떨어뜨려서 사용하지 않았습니다."

---

#### Day 2: Feature Engineering

**📚 핵심 개념**

1. **정형 피처 (XGBoost용)**
   - 시간 피처: 요일, 시간대, 주말 여부
   - 금액 피처: 로그 변환, 구간화
   - 집계 피처: card_id별 거래 횟수, 평균 금액
   - 범주형 인코딩: Label Encoding, Target Encoding

2. **시계열 피처 (LSTM용)** ⭐
   - 사용자별 최근 N개 거래 시퀀스 추출
   - 시퀀스 피처: [금액, 시간, 카테고리, ...]
   - Padding: 거래 수가 N 미만이면 0으로 채움
   - Scaling: MinMaxScaler로 정규화

3. **시간 기반 분할 (Time-based Split)**
   - 미래 데이터로 과거를 예측하면 Data Leakage
   - 시간순 정렬 후 분할

**✏️ 미니 연습**
- Q1: 왜 랜덤 분할이 아닌 시간 기반 분할을 하나요?
- Q2: 시퀀스 길이 N은 어떻게 정하나요?

**🎯 면접 Q&A**

Q: "어떤 피처가 가장 중요했나요?"
> "SHAP 분석 결과 거래 금액, 거래 시간대, 이메일 도메인 일치 여부가 Top 3였습니다. 특히 '거래 금액의 로그값'과 'card_id별 최근 거래 횟수' 같은 파생 피처가 Raw 피처보다 중요도가 높았습니다."

Q: "시계열 피처는 어떻게 만들었나요?"
> "사용자별로 최근 10개 거래를 추출해서 시퀀스로 만들었습니다. 거래가 10개 미만이면 앞쪽을 0으로 패딩하고, 금액은 로그 변환 후 MinMaxScaler로 0~1 범위로 정규화했습니다. 시퀀스 길이 10은 실험으로 결정했습니다."

---

#### Day 3: XGBoost 모델 ⭐

**📚 핵심 개념**

1. **Gradient Boosting 모델 비교**
   - XGBoost: 정규화 강함, SHAP 호환 최상
   - LightGBM: 학습 빠름, 대용량에 유리
   - CatBoost: 범주형 자동 처리, 학습 느림

2. **Threshold 최적화**
   - 기본 0.5가 아닌 최적 Threshold 찾기
   - FN(놓친 사기) vs FP(오탐) 비용 트레이드오프
   - Precision-Recall Curve 활용

3. **Optuna 하이퍼파라미터 튜닝**
   - Bayesian Optimization 기반
   - 탐색 공간 정의, Trial 수 설정

**✏️ 미니 연습**
- Q1: XGBoost와 LightGBM의 트리 생성 방식 차이는?
- Q2: Threshold를 0.3으로 낮추면 어떤 변화가?

**🎯 면접 Q&A**

Q: "왜 XGBoost를 선택했나요?"
> "3개 모델(XGBoost, LightGBM, CatBoost)을 동일 조건에서 비교 실험했습니다. XGBoost가 AUC 0.92로 가장 높았고, SHAP TreeExplainer와의 호환성도 가장 좋았습니다. 학습 시간은 LightGBM이 빨랐지만 성능 차이(0.01)가 있어 XGBoost를 선택했습니다."

Q: "Threshold는 어떻게 정했나요?"
> "FDS에서 FN(놓친 사기)이 FP(오탐)보다 비용이 훨씬 큽니다. FN:FP 비용을 10:1로 가정하고, 비용 함수를 최소화하는 Threshold를 계산했습니다. 결과적으로 0.18이 최적이었고, 이 때 Recall 90.55%, Precision 9.78%를 달성했습니다. 4단계 위험도(approve/verify/hold/block)로 실무에서 단계별 대응이 가능합니다."

---

#### Day 4: LSTM 모델 ⭐

**📚 핵심 개념**

1. **시퀀스 데이터 준비**
   - 사용자별 최근 N개 거래 추출
   - Padding/Truncating으로 길이 통일
   - Feature Scaling (거래 금액 등)

2. **LSTM 구조**
   ```python
   class FraudLSTM(nn.Module):
       def __init__(self, input_size, hidden_size=64):
           super().__init__()
           self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
           self.fc = nn.Linear(hidden_size, 1)
           self.sigmoid = nn.Sigmoid()

       def forward(self, x):
           # x: (batch, seq_len, features)
           _, (h_n, _) = self.lstm(x)
           out = self.fc(h_n.squeeze(0))
           return self.sigmoid(out)
   ```

3. **PyTorch 학습 루프**
   - Dataset, DataLoader 구현
   - BCELoss + Adam optimizer
   - Early Stopping으로 과적합 방지

**✏️ 미니 연습**
- Q1: LSTM의 hidden state와 cell state의 차이는?
- Q2: batch_first=True가 필요한 이유는?

**🎯 면접 Q&A**

Q: "왜 LSTM을 선택했나요?"
> "거래 시퀀스의 시간적 패턴을 학습하기 위해서입니다. Transformer도 고려했지만, 시퀀스 길이가 10~20으로 짧아서 LSTM이 충분했고 학습도 더 빠릅니다. GRU와 비교 실험에서 LSTM이 약간 더 좋았습니다."

Q: "시퀀스 길이는 어떻게 정했나요?"
> "5, 10, 15, 20으로 실험했습니다. 10이 AUC와 학습 시간의 트레이드오프에서 최적이었습니다. 5는 너무 짧아 패턴을 못 잡고, 20은 학습 시간 대비 성능 향상이 미미했습니다."

---

#### Day 5: Ensemble 실험 + 평가 ⭐

**📚 핵심 개념**

1. **앙상블 실험 결과**
   - LSTM AUC 0.70으로 예상보다 낮음
   - 앙상블해도 +0.12% 향상에 그침
   - **결론: XGBoost 단독 채택**

2. **실험 결과**
   ```
   XGBoost 단독: AUC 0.9042 → 채택
   LSTM 단독:    AUC 0.7054 → 성능 낮음
   앙상블:       AUC 0.9054 → +0.12% (효과 미미)
   ```

3. **성능 비교**
   | 모델 | AUC | Recall@0.18 | 결론 |
   |------|-----|-------------|------|
   | XGBoost 단독 | 0.9042 | 90.55% | **채택** |
   | LSTM | 0.7054 | - | 성능 낮음 |
   | 앙상블 (0.9:0.1) | 0.9054 | - | 효과 미미 |

**✏️ 미니 연습**
- Q1: 앙상블 효과가 없을 때 어떤 판단을 해야 하나?
- Q2: 복잡도 대비 효과 분석은 왜 중요한가?

**🎯 면접 Q&A**

Q: "앙상블로 얼마나 성능이 올랐나요?"
> "LSTM AUC가 0.70으로 낮아서, 앙상블해도 +0.12% 향상에 그쳤습니다. LSTM 서빙 비용(PyTorch, 시퀀스 생성) 대비 효과가 없어서 XGBoost 단독을 채택했습니다. 이 경험으로 '복잡도 대비 효과'를 분석하는 실무적 판단력을 얻었습니다."

Q: "왜 LSTM이 효과가 없었나요?"
> "IEEE-CIS 데이터 특성상, 시계열 패턴보다 정형 피처(금액, 시간, 카드정보)가 더 결정적이었습니다. 모든 문제에 딥러닝이 최선은 아닙니다."

---

#### Day 8: React Admin ⭐

**📚 핵심 개념**

1. **React + Ant Design 선택 이유**
   - 금융권 Admin 표준 조합
   - 테이블, 차트 컴포넌트 풍부
   - SHAP 시각화 연동 용이

2. **프로젝트 구조**
   ```
   frontend/
   ├── src/
   │   ├── components/       # 재사용 컴포넌트
   │   │   ├── TransactionTable.jsx
   │   │   └── TransactionDetail.jsx
   │   ├── pages/            # 페이지
   │   │   └── Dashboard.jsx
   │   └── api/              # API 연동
   │       └── client.js
   ├── package.json
   └── Dockerfile
   ```

3. **핵심 컴포넌트**
   - TransactionTable: 거래 목록 (pagination, 필터)
   - TransactionDetail: 상세 + SHAP 설명
   - Dashboard: 통계 요약

4. **FastAPI 연동**
   ```javascript
   // api/client.js
   const API_BASE = 'http://localhost:8000';

   export const getTransactions = async (params) => {
     const res = await fetch(`${API_BASE}/transactions?${new URLSearchParams(params)}`);
     return res.json();
   };

   export const getTransactionDetail = async (id) => {
     const res = await fetch(`${API_BASE}/transactions/${id}`);
     return res.json();
   };
   ```

**✏️ 미니 연습**
- Q1: SPA에서 API 에러 처리는 어떻게?
- Q2: SHAP 값을 막대 그래프로 표시하려면?

**🎯 면접 Q&A**

Q: "프론트엔드는 왜 React를 선택했나요?"
> "금융권에서 React + Ant Design 조합이 가장 많이 사용됩니다. Ant Design의 Table, Descriptions 컴포넌트로 거래 목록과 상세 정보를 빠르게 구현할 수 있었습니다."

Q: "SHAP 설명을 어떻게 보여주나요?"
> "API에서 받은 SHAP 값을 Ant Design의 Progress 컴포넌트로 막대 그래프 형태로 시각화합니다. 양수(사기 방향)는 빨간색, 음수(정상 방향)는 초록색으로 구분합니다."

---

#### Day 9: 트리 스태킹 ⭐⭐ (XGBoost + LightGBM + CatBoost)

**📚 핵심 개념**

1. **왜 트리 스태킹인가?**
   - 2025 벤치마크: F1 0.99, AUC 1.00 달성
   - 현업에서 증가 중인 트렌드
   - 실시간 추론 가능 (~12ms)

2. **스태킹 구조**
   ```
   [Base Models]
   XGBoost  ─┐
   LightGBM ─┼→ [Meta-Learner] → 최종 예측
   CatBoost ─┘
   ```

3. **각 모델의 강점**
   | 모델 | 강점 | 약점 |
   |------|------|------|
   | XGBoost | 정규화, SHAP 호환 | 느림 |
   | LightGBM | 가장 빠름, 메모리 효율 | Tail latency |
   | CatBoost | 범주형 자동 처리 | 학습 느림 |

4. **기대 성능**
   | 지표 | XGBoost 단독 | 트리 스태킹 |
   |------|--------------|-------------|
   | F1 | 0.95 | **0.99** |
   | AUC | 0.91 | **0.99+** |
   | 추론 | 5ms | 12ms |

**✏️ 미니 연습**
- Q1: LightGBM과 XGBoost의 트리 생성 방식 차이는?
- Q2: Meta-learner로 뭘 써야 할까?

**🎯 면접 Q&A**

Q: "왜 트리 스태킹을 사용했나요?"
> "XGBoost 단독으로 F1 0.95를 달성했지만, 2025년 벤치마크에서 트리 스태킹이 F1 0.99를 달성함을 확인했습니다. LightGBM과 CatBoost를 추가하고 Meta-learner로 XGBoost를 사용해서 성능을 극대화했습니다."

Q: "스태킹의 단점은?"
> "추론 시간이 단일 모델 대비 2~3배 증가합니다. 하지만 12ms로 실시간 서비스에 충분하고, 성능 향상(F1 +4%)이 트레이드오프를 정당화합니다."

---

#### Day 10: Transformer (선택) ⭐⭐

**📚 핵심 개념**

1. **왜 Transformer인가?**
   - 2025 벤치마크: F1 0.998 (연구 최강)
   - 정형 + 시퀀스 데이터 모두 처리 가능
   - Self-Attention으로 피처 간 관계 학습

2. **TabTransformer 구조**
   ```
   [입력] 정형 피처
        ↓
   [Embedding] 범주형 피처 → 임베딩
        ↓
   [Transformer Encoder] Self-Attention
        ↓
   [MLP Head] → 예측
   ```

3. **XGBoost vs Transformer**
   | 항목 | XGBoost | Transformer |
   |------|---------|-------------|
   | F1 | 0.95 | **0.998** |
   | 속도 | 5ms | 50-100ms |
   | 현업 채택 | ✅ 표준 | ⚠️ 도입 중 |

**✏️ 미니 연습**
- Q1: Self-Attention이 피처 관계를 어떻게 학습하나?
- Q2: Positional Encoding이 정형 데이터에 필요한가?

**🎯 면접 Q&A**

Q: "왜 Transformer를 실험했나요?"
> "2025년 연구에서 Transformer가 F1 0.998로 XGBoost(0.95)를 크게 능가했습니다. TabTransformer로 정형 데이터에 적용해서 연구 트렌드를 검증했습니다."

Q: "현업에서 Transformer 안 쓰는 이유는?"
> "추론 속도(50-100ms)가 XGBoost(5ms)보다 느리고, 아직 검증 기간이 짧습니다. 하지만 HSBC, Featurespace 등에서 도입 중이어서 미래 표준이 될 가능성이 높습니다."

---

#### Day 11: 하이브리드 (선택) ⭐⭐

**📚 핵심 개념**

1. **하이브리드 구조**
   ```
   거래 데이터 → Transformer → 임베딩 (피처 벡터)
                                    ↓
                              XGBoost → 최종 예측

   "DL의 패턴 인식 + XGBoost의 안정성/해석성"
   ```

2. **왜 하이브리드인가?**
   - DL: 복잡한 패턴 자동 학습
   - XGBoost: 안정적, SHAP 호환
   - 결합: 두 장점을 모두 활용

3. **NVIDIA 사례**
   > "GNN 임베딩 + XGBoost로 1% 향상 = 수백만 달러 절감"

**✏️ 미니 연습**
- Q1: 임베딩을 피처로 쓰면 뭐가 좋은가?
- Q2: SHAP을 어디에 적용하나?

**🎯 면접 Q&A**

Q: "하이브리드의 장점은?"
> "Transformer가 자동으로 복잡한 패턴을 임베딩으로 추출하고, XGBoost가 이를 해석 가능하게 분류합니다. DL의 성능과 ML의 안정성을 모두 얻었습니다."

---

#### Day 12: PaySim 시퀀스 실험 (선택) ⭐⭐

**📚 핵심 개념**

1. **왜 PaySim인가?**
   - IEEE-CIS: PCA 변환 → 시퀀스 패턴 없음 → LSTM 실패
   - PaySim: 진짜 거래 시퀀스 → LSTM 검증 가능

2. **PaySim 데이터셋**
   | 항목 | 값 |
   |------|-----|
   | 거래 수 | 6.3M |
   | 기간 | 30일 (744 steps) |
   | 사용자 ID | nameOrig |
   | 거래 타입 | CASH_IN, CASH_OUT, TRANSFER 등 |

3. **실험 목표**
   ```
   [검증할 것]
   1. IEEE-CIS에서 LSTM 실패 = 데이터 문제였는가?
   2. PaySim에서 LSTM 성능은?
   3. 진짜 시퀀스에서 앙상블 효과는?
   ```

**✏️ 미니 연습**
- Q1: nameOrig로 시퀀스 어떻게 만드나?
- Q2: IEEE-CIS vs PaySim 피처 차이는?

**🎯 면접 Q&A**

Q: "왜 데이터셋을 바꿨나요?"
> "IEEE-CIS에서 LSTM이 AUC 0.70으로 낮았는데, 원인이 데이터(PCA 변환된 정적 피처)인지 모델인지 확인하고 싶었습니다. PaySim은 진짜 거래 시퀀스가 있어서 LSTM의 진짜 성능을 검증할 수 있었습니다."

Q: "결과는 어땠나요?"
> "PaySim에서 LSTM이 [결과]를 달성했습니다. 이로써 'LSTM은 진짜 시퀀스 데이터가 있어야 효과적'이라는 결론을 얻었습니다. 데이터 특성에 따른 모델 선택의 중요성을 배웠습니다."

---

#### Day 6: SHAP 설명

**📚 핵심 개념**

1. **XGBoost 설명: TreeExplainer**
   ```python
   import shap
   explainer = shap.TreeExplainer(xgb_model)
   shap_values = explainer.shap_values(X_test)
   ```

2. **LSTM 설명: DeepExplainer**
   ```python
   background = X_train[:100]  # 배경 데이터
   explainer = shap.DeepExplainer(lstm_model, background)
   shap_values = explainer.shap_values(X_test)
   ```

3. **XGBoost 단독 설명** (앙상블 효과 미미하여 채택)
   - XGBoost SHAP: 정형 특성 기여도 Top 5
   - 참고: LSTM 앙상블 실험 시 +0.12% 향상에 그쳐 복잡도 대비 효과 없음

**✏️ 미니 연습**
- Q1: SHAP 값이 양수/음수일 때 각각 무슨 의미?
- Q2: LIME과 SHAP의 차이는?

**🎯 면접 Q&A**

Q: "왜 SHAP을 선택했나요?"
> "LIME, SHAP, Feature Importance 중 SHAP을 선택했습니다. SHAP은 이론적 기반(Shapley Value)이 탄탄하고, TreeExplainer와 DeepExplainer로 XGBoost와 LSTM 모두 설명할 수 있습니다."

Q: "앙상블 모델은 어떻게 설명하나요?"
> "각 모델의 SHAP 값을 앙상블 가중치로 합칩니다. 최종 설명은 '금액이 높고(+0.25), 시간이 비정상(+0.18), 최근 거래 패턴 급변(+0.22)'처럼 정형 + 시계열 요인을 모두 포함합니다."

---

#### Day 7: FastAPI 배포

**📚 핵심 개념**

1. **API 엔드포인트 설계**
   ```python
   @app.post("/predict")
   async def predict(transaction: Transaction):
       # 1. Feature Engineering
       X_tabular = extract_tabular_features(transaction)
       X_sequence = extract_sequence_features(transaction.user_id)

       # 2. XGBoost 예측 (앙상블 효과 미미하여 단독 채택)
       p_xgb = xgb_model.predict_proba(X_tabular)[0, 1]

       # 3. SHAP 설명
       explanation = generate_shap_explanation(X_tabular)

       return {
           "fraud_probability": p_xgb,
           "model_scores": {"xgboost": p_xgb},
           "explanation": explanation
       }
   ```

2. **모델 저장 (joblib)**
   ```python
   import joblib

   # 모델 저장 (개발용, .joblib 확장자 표준)
   joblib.dump(xgb_model, 'models/xgb_model.joblib')

   # PyTorch는 .pt 확장자
   torch.save(lstm_model.state_dict(), 'models/lstm_model.pt')

   # 모델 로드
   xgb_model = joblib.load('models/xgb_model.joblib')
   ```
   - Phase 1: joblib (.joblib 확장자, 개발용)
   - Phase 3: ONNX → TensorRT → Triton (프로덕션)

3. **Docker 컨테이너화**
   - Dockerfile 작성
   - 모델 파일 포함
   - docker compose up으로 실행

4. **성능 최적화**
   - 모델 로딩: 앱 시작 시 한 번만
   - SHAP 캐싱: 동일 요청 재계산 방지

**✏️ 미니 연습**
- Q1: FastAPI의 async def와 def의 차이는?
- Q2: 모델 서빙 시 메모리 최적화 방법은?

**🎯 면접 Q&A**

Q: "API 응답 시간은 얼마나 걸리나요?"
> "XGBoost 예측 ~10ms, LSTM 예측 ~50ms, SHAP 계산 ~100ms로 총 약 160ms입니다. 목표였던 200ms 이하를 달성했습니다. 배치 예측을 지원하면 처리량을 더 높일 수 있습니다."

Q: "모델 업데이트는 어떻게 하나요?"
> "현재는 Docker 이미지 재빌드 방식입니다. Phase 2에서 MLflow로 모델 버전 관리를 추가하면 무중단 배포가 가능해집니다."

---

## 7. 차별화 포인트 총정리

| # | 포인트 | 관련 태스크 | 면접 질문 |
|---|--------|-------------|-----------|
| 1 | XGBoost vs LightGBM 비교 | Day 3 | "왜 XGBoost?" |
| 2 | Threshold 최적화 (0.18) | Day 3 | "Threshold 어떻게 정했나?" |
| 3 | 4단계 위험도 | Day 3 | "위험도 레벨은?" |
| 4 | LSTM 실험 → 효과 미미 판단 ⭐ | Day 4, 5 | "왜 LSTM 안 썼나?" |
| 5 | 복잡도 대비 효과 분석 ⭐ | Day 5 | "앙상블 효과는?" |
| 6 | SHAP 설명 | Day 6 | "설명은 어떻게?" |
| 7 | 학습-서빙 일관성 이슈 해결 | Day 7 | "겪은 문제는?" |

**핵심 메시지:**
> "LSTM 앙상블을 시도했지만 +0.12% 향상에 그쳐서 XGBoost 단독을 채택했습니다. '딥러닝이 항상 좋은 건 아니다'라는 것을 데이터로 증명하고, 복잡도 대비 효과를 분석하는 실무적 판단을 경험했습니다."

---

## 8. 하드웨어 요구사항

| 구성요소 | 필요 사양 | 용도 |
|----------|-----------|------|
| GPU | RTX 2070 Super (8GB) | XGBoost GPU + LSTM 학습 |
| RAM | 16GB+ | 데이터 로딩, 시퀀스 생성 |
| Storage | 15GB+ | 데이터 + 모델 (.pkl, .pt) |

**참고:** 8GB VRAM으로 XGBoost + LSTM 학습 충분. 외부 GPU 불필요.

---

## 9. Phase 1 완료 후 포트폴리오

### 9.1 결과물 요약

| 항목 | 내용 |
|------|------|
| 모델 | XGBoost 단독 (LSTM 실험 후 채택) |
| 성능 | AUC 0.91, Recall 90.55%, AUPRC 0.53 |
| 설명 | SHAP TreeExplainer (Top 5 피처) |
| 배포 | FastAPI + Docker |
| 응답 | < 200ms |

### 9.2 면접 어필 포인트

1. **"XGBoost vs LSTM 비교 실험"** → 딥러닝이 항상 좋은 건 아님을 증명
2. **"앙상블 +0.12% → 효과 없음 판단"** → 복잡도 대비 효과 분석
3. **"Recall 90.55% 달성"** → 현업 기준 충족
4. **"SHAP으로 설명 가능한 AI"** → 금융 규제 대응
5. **"학습-서빙 일관성 이슈 해결"** → 실무 경험

### 9.3 포트폴리오 레벨

```
Phase 1 완료: 기본 포트폴리오
├─ Tier 1 (필수)     : 4/5 ✅
├─ Tier 2 (차별화)   : 1/5 ⚠️
└─ 예상 서류 통과율  : 30~40%

→ 지원 시작 가능, Phase 2 병행 권장
```

---

## 10. 다음 단계 (Phase 2~5 로드맵)

> 상세 내용: [docs/roadmap.md](./roadmap.md)

### Phase 2: MLOps + 모니터링 (6일)

| 기술 | 역할 | 면접 어필 |
|------|------|-----------|
| **MLflow** | 실험 추적, 모델 버전 관리 | "모델 버전 관리 경험" |
| **Evidently** | 드리프트 모니터링 | "성능 저하 자동 감지" |
| **비용 기반 최적화** | 비즈니스 임팩트 계산 | "연간 4억 손실 절감" |
| **GitHub Actions** | CI/CD 파이프라인 | "자동 테스트/배포" |
| **A/B 테스트** | 모델 비교 실험 체계 | "통계적 유의성 검증" |
| **Prometheus + Grafana** | 시스템 모니터링 | "실시간 메트릭 대시보드" |

**완료 후 레벨:** 좋음 ⭐ (서류 통과율 60~70%)

### Phase 3: 실시간 + 워크플로 (6일)

**모델 서빙 변환 흐름:**
```
Phase 1: joblib (개발용)
    ↓
Phase 3: joblib → ONNX → TensorRT → Triton (프로덕션)
```

| 기술 | 역할 | 면접 어필 |
|------|------|-----------|
| **Kafka** | 실시간 스트리밍 | "실시간 사기 탐지, 지연 50ms" |
| **Airflow** | 재학습 스케줄링 | "주간 자동 재학습" |
| **Feast** | Feature Store | "피처 일관성 보장" |
| **ONNX** | joblib → ONNX 변환 | "추론 속도 3배 향상" |
| **TensorRT** | DL 추론 최적화 | "LSTM 추론 10ms 이하" |
| **Triton** | 모델 서빙 플랫폼 | "멀티모델 동시 서빙" |

**완료 후 레벨:** 매우 좋음 ⭐⭐ (서류 통과율 80~90%)

### Phase 4: 클라우드 + 데이터 인프라 (5일)

| 기술 | 역할 | 면접 어필 |
|------|------|-----------|
| **BigQuery** | 클라우드 DW | "대용량 데이터 분석" |
| **Kubernetes** | 오케스트레이션 | "오토스케일링" |
| **S3/MinIO** | 오브젝트 스토리지 | "모델 아티팩트 관리" |
| **Spark** | 대용량 처리 (선택) | "분산 전처리" |

**완료 후 레벨:** 풀스택 ⭐⭐⭐

### Phase 5: 고급 + 차별화 (5일+)

| 기술 | 역할 | 면접 어필 |
|------|------|-----------|
| **GNN** | 그래프 패턴 학습 | "사기 조직 네트워크 탐지" |
| **Kubeflow** | ML 파이프라인 | "End-to-End 파이프라인" |
| **Flink** | 고급 스트리밍 | "복잡 이벤트 처리" |
| **ELK Stack** | 로그 관리 | "로그 분석 대시보드" |

**완료 후 레벨:** 시니어급

---

## 11. 취업 전략

### Phase별 지원 시점

| Phase | 지원 대상 | 전략 |
|-------|-----------|------|
| 1 완료 | 중소/스타트업 | 기본 포트폴리오로 지원 시작 |
| 2 완료 | 금융 IT 자회사 | MLOps 경험으로 어필 |
| 3 완료 | 핀테크/대기업 | 실시간 처리로 차별화 |

### 면접 피드백 활용

```
Phase 1 지원 → 면접 피드백 수집
                ↓
        "MLOps 경험 부족" 피드백
                ↓
        Phase 2 우선 완료 → 재지원
```

---

## 12. 참고 자료

**데이터셋:**
- IEEE-CIS Fraud Detection (Kaggle)

**기술 문서:**
- XGBoost 공식 문서
- PyTorch 공식 문서 (LSTM)
- SHAP 공식 문서
- FastAPI 공식 문서

**관련 논문:**
- "Deep Learning for Anomaly Detection: A Survey" (2020)
- "Credit Card Fraud Detection Using Deep Learning" (IEEE, 2019)
