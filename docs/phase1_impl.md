# Phase 1: FDS + XAI + RAG - 구현 상세 (AI용)

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

### 참고 예시

상세 구현 예시는 별도 파일 참조:
- `1-1_impl_example.md`: 노트북 전체 셀 구조 + 코드
- `1-2_impl_example.md`: 노트북 + src 모듈화 예시

**1-3 ~ 1-7도 위 예시와 동일한 수준으로 구현해야 함.**

---

## LLM 선택: Qwen 2.5 3B

### 선택 이유

| 조건 | Qwen 2.5 3B | 비고 |
|------|-------------|------|
| VRAM | Q4 양자화 시 ~2-3GB | RTX 2070 Super 8GB OK |
| 한국어 | 다국어 모델 중 상위 | 128K 컨텍스트 |
| 라이센스 | Apache 2.0 | 제약 없음 |
| Ollama | ✅ 지원 | 설치 간편 |
| 커뮤니티 | 활발 | 문제 해결 용이 |

### 면접 답변

> "8GB VRAM 제약으로 3B급 모델이 필수였습니다. 한국어 금융 문서 RAG용으로 다국어 지원 + 128K 컨텍스트가 필요했고, Qwen 2.5가 이 조건을 충족했습니다. Apache 2.0 라이센스와 활발한 Ollama 커뮤니티 지원도 선택 이유입니다. 한국어 특화가 아닌 점은 QLoRA 파인튜닝으로 보완했습니다."

### 대안 모델 비교 (참고)

| 모델 | 크기 | 한국어 | 비고 |
|------|------|--------|------|
| Qwen 2.5 | 3B | ⭐⭐ | 선택 ✅ |
| EXAONE 3.5 | 7.8B | ⭐⭐⭐ | 한국어 최강, but 더 큼 |
| Llama 3.2 | 3B | ⭐ | 영어 강함 |
| Phi-3 | 3.8B | ⭐ | 추론 강함 |

---

## 파일 구조

```
fds-system/
├── notebooks/
│   └── phase1/
│       ├── 1-1_data_eda.ipynb
│       ├── 1-2_feature_baseline.ipynb
│       ├── 1-3_model_optimization.ipynb
│       ├── 1-4_shap_explanation.ipynb
│       ├── 1-5_rag_setup.ipynb
│       ├── 1-6_rag_advanced.ipynb
│       └── 1-7_agent_api.ipynb
├── src/
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   ├── model.py
│   │   └── explainer.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunking.py
│   │   ├── embedding.py
│   │   ├── retriever.py
│   │   └── generator.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── graph.py
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       ├── schemas.py
│       └── tasks.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── docker-compose.yml
└── requirements.txt
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
- 왜 랜덤 분할이 안 되는지
- TransactionDT 기준 정렬
- 80/20 분할

### 실습 목록
- 실습 1: 데이터 로드 및 병합 (LEFT JOIN)
- 실습 2: 타겟 불균형 시각화
- 실습 3: 결측치 분석 및 처리 전략
- 실습 4: 타겟별 금액 분포 비교
- 실습 5: 시간 기반 train/test 분할

### 노트북 구조

```
[마크다운] # 1-1: 데이터 + EDA
[마크다운] ## 학습 목표
[코드] import
[마크다운] ## 1. 데이터 로드
[코드] pd.read_csv
[마크다운] ### 💻 실습 1: 데이터 병합
[코드] 실습 1 (TODO)
[코드] 체크포인트 1
[마크다운] ## 2. 불균형 데이터
[코드] 타겟 분포
[마크다운] ### 💻 실습 2: 불균형 시각화
[코드] 실습 2 (TODO)
[코드] 체크포인트 2
[마크다운] ## 3. 결측치
[코드] 결측 비율
[마크다운] ### 💻 실습 3: 결측치 전략
[코드] 실습 3 (TODO)
[코드] 체크포인트 3
[마크다운] ## 4. 피처 EDA
[코드] 분포 시각화
[마크다운] ### 💻 실습 4: 타겟별 비교
[코드] 실습 4 (TODO)
[코드] 체크포인트 4
[마크다운] ## 5. 시간 분할
[마크다운] ### 💻 실습 5: train/test 분할
[코드] 실습 5 (TODO)
[코드] 체크포인트 5
[마크다운] ## ✅ 최종 체크포인트
[코드] 데이터 저장, 요약
```

### 상세 코드

#### 실습 1: 데이터 병합

```python
# 데이터 로드
train_transaction = pd.read_csv('data/raw/train_transaction.csv')
train_identity = pd.read_csv('data/raw/train_identity.csv')

print(f"Transaction: {train_transaction.shape}")
print(f"Identity: {train_identity.shape}")
```

```python
# 💻 실습 1: LEFT JOIN
# TODO: TransactionID 기준으로 병합
df = None  # pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
```

```python
# 체크포인트 1
assert df is not None, "병합하세요"
assert df.shape[0] == train_transaction.shape[0], "LEFT JOIN이므로 행 수 동일"
assert 'DeviceType' in df.columns, "Identity 컬럼 포함"
print("✅ 체크포인트 1 통과!")
```

#### 실습 2: 불균형 시각화

```python
# 타겟 분포
fraud_rate = df['isFraud'].mean()
print(f"사기 비율: {fraud_rate:.2%}")
```

```python
# 💻 실습 2: 막대 그래프
# TODO: isFraud 분포 시각화
fig, ax = plt.subplots(figsize=(6, 4))
# df['isFraud'].value_counts().plot(kind='bar', ax=ax)
# ax.set_title('Target Distribution')
plt.show()
```

```python
# 체크포인트 2
assert fraud_rate < 0.05, "불균형 데이터 확인"
print("✅ 체크포인트 2 통과!")
```

#### 실습 3: 결측치 분석

```python
# 결측 비율
missing = df.isnull().sum() / len(df) * 100
missing = missing[missing > 0].sort_values(ascending=False)
print(missing.head(20))
```

```python
# 💻 실습 3: 50% 이상 결측 컬럼
# TODO: 50% 이상 결측 컬럼 리스트
high_missing = None  # missing[missing > 50].index.tolist()

# TODO: 해당 컬럼 제거
df_clean = None  # df.drop(columns=high_missing)
```

```python
# 체크포인트 3
assert high_missing is not None, "리스트 생성"
assert df_clean.shape[1] < df.shape[1], "컬럼 제거됨"
print("✅ 체크포인트 3 통과!")
```

#### 실습 4: 타겟별 비교

```python
# 💻 실습 4: 정상 vs 사기 금액 분포
# TODO: 두 히스토그램 겹쳐 그리기
fig, ax = plt.subplots(figsize=(10, 4))

# normal = df[df['isFraud']==0]['TransactionAmt']
# fraud = df[df['isFraud']==1]['TransactionAmt']
# ax.hist(normal, bins=50, alpha=0.5, label='Normal')
# ax.hist(fraud, bins=50, alpha=0.5, label='Fraud')
# ax.legend()

plt.show()
```

```python
# 체크포인트 4
fraud_mean = df[df['isFraud']==1]['TransactionAmt'].mean()
normal_mean = df[df['isFraud']==0]['TransactionAmt'].mean()
print(f"정상 평균: ${normal_mean:,.0f}, 사기 평균: ${fraud_mean:,.0f}")
print("✅ 체크포인트 4 통과!")
```

#### 실습 5: 시간 분할

```python
# 💻 실습 5: 시간 기반 분할
# TODO: TransactionDT 기준 정렬
df_sorted = None  # df.sort_values('TransactionDT')

# TODO: 80/20 분할
split_idx = int(len(df_sorted) * 0.8)
train_df = None  # df_sorted.iloc[:split_idx]
test_df = None   # df_sorted.iloc[split_idx:]
```

```python
# 체크포인트 5
assert train_df['TransactionDT'].max() <= test_df['TransactionDT'].min(), "시간순"
print("✅ 체크포인트 5 통과!")
```

#### 최종

```python
# 저장
train_df.to_csv('data/processed/train.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)

print("="*50)
print("🎉 1-1 완료!")
print("="*50)
```

---

## 1-2: Feature Engineering + Baseline (Day 2)

### 필요 패키지
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
```

### 세부 설명 리스트

**1. 시간 피처**
- hour: 시간 (0-23)
- dayofweek: 요일 (0-6)
- is_weekend: 주말 여부
- is_night: 야간 여부 (22-6시)

**2. 금액 피처**
- amt_log: 로그 변환
- amt_bin: 구간화
- amt_decimal: 소수점 유무

**3. 집계 피처**
- card1별 거래 횟수
- card1별 평균 금액
- card1별 사기율 (train만!)

**4. 범주형 인코딩**
- LabelEncoder
- NaN → 'unknown' 처리

**5. Baseline 모델**
- RandomForest
- AUC 확인

### 실습 목록
- 실습 1: 시간 피처 생성
- 실습 2: 금액 피처 생성
- 실습 3: 집계 피처 생성
- 실습 4: 범주형 인코딩
- 실습 5: Baseline 모델

---

## 1-3: 모델 고도화 (Day 3) ⭐

### 필요 패키지
```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
import optuna
import time
```

### 세부 설명 리스트

**1. 모델 비교 실험** ⭐
- XGBoost, LightGBM, CatBoost
- 동일 조건 (동일 피처, 동일 분할)
- AUC, 학습 시간 측정
- 결과 표로 정리

**2. Threshold 최적화** ⭐
- FN:FP 비용 비율 정의 (10:1)
- 비용 함수 계산
- 최적 threshold 찾기
- Precision-Recall Curve

**3. Optuna 튜닝**
- 탐색 공간 정의
- objective 함수
- n_trials 설정

**4. 최종 모델 저장**
- joblib.dump
- 메타데이터 함께 저장

### 실습 목록
- 실습 1: 3개 모델 비교 실험 → 결과 표
- 실습 2: Threshold 비용 분석 → 그래프
- 실습 3: Optuna 튜닝
- 실습 4: 모델 저장

### 핵심 코드: 모델 비교 표

```python
# 실험 결과 표
results_df = pd.DataFrame([
    {'Model': 'XGBoost', 'AUC': 0.91, 'Time(s)': 45, 'SHAP호환': '최상'},
    {'Model': 'LightGBM', 'AUC': 0.90, 'Time(s)': 32, 'SHAP호환': '좋음'},
    {'Model': 'CatBoost', 'AUC': 0.90, 'Time(s)': 98, 'SHAP호환': '제한적'},
])
print(results_df.to_markdown(index=False))
```

### 핵심 코드: Threshold 분석

```python
def calculate_cost(threshold, y_true, y_prob, fn_cost=10, fp_cost=1):
    y_pred = (y_prob >= threshold).astype(int)
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return fn * fn_cost + fp * fp_cost

# 최적 threshold 찾기
thresholds = np.arange(0.1, 0.9, 0.05)
costs = [calculate_cost(t, y_test, y_prob) for t in thresholds]
optimal_threshold = thresholds[np.argmin(costs)]
```

---

## 1-4: SHAP 설명 (Day 4)

### 필요 패키지
```python
import shap
import joblib
```

### 세부 설명 리스트

**1. SHAP 기초**
- TreeExplainer (트리 모델용)
- shap_values 계산
- expected_value

**2. 시각화**
- Summary Plot: 전체 피처 중요도
- Waterfall Plot: 개별 예측 설명
- Force Plot: 기여도 시각화

**3. Top 피처 추출**
- 절대값 기준 정렬
- Top K 선택

**4. 자연어 변환**
- 피처명 → 설명 매핑
- 방향 (증가/감소) 포함

### 실습 목록
- 실습 1: SHAP 값 계산
- 실습 2: Summary Plot
- 실습 3: 개별 예측 Waterfall
- 실습 4: 자연어 설명 생성

### 핵심 코드: 자연어 설명

```python
FEATURE_EXPLANATIONS = {
    'TransactionAmt': '거래 금액',
    'hour': '거래 시간',
    'is_night': '야간 거래',
    'amt_log': '거래 금액 (로그)',
    'card1_count': '카드 거래 횟수',
}

def get_top_features(shap_values, feature_names, top_k=3):
    abs_shap = np.abs(shap_values)
    top_idx = np.argsort(abs_shap)[-top_k:][::-1]
    
    result = []
    for idx in top_idx:
        result.append({
            'feature': feature_names[idx],
            'value': shap_values[idx],
            'direction': '증가' if shap_values[idx] > 0 else '감소'
        })
    return result

def to_natural_language(top_features):
    lines = []
    for f in top_features:
        name = FEATURE_EXPLANATIONS.get(f['feature'], f['feature'])
        lines.append(f"- {name}: 사기 확률 {f['direction']}")
    return "\n".join(lines)
```

---

## 1-5: RAG 환경 + 청킹 (Day 5) ⭐

### 필요 패키지
```python
# Docker: postgres (pgvector), ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
```

### 세부 설명 리스트

**1. Docker 환경**
- docker-compose.yml
- PostgreSQL + pgvector 확장
- Ollama + qwen2.5:3b

**2. 금융 규정 문서**
- 전자금융거래법
- FDS 가이드라인
- 텍스트 파일로 준비

**3. 청킹 전략 비교** ⭐
- Fixed (500자)
- Semantic (의미 단위)
- Sentence (문장 단위)
- 실험 결과 표

**4. 임베딩**
- BGE-M3 (다국어)
- 차원: 1024

**5. 벡터 저장**
- PGVector 연결
- 문서 저장

### 실습 목록
- 실습 1: Docker 환경 구성
- 실습 2: 문서 로드 및 청킹
- 실습 3: 청킹 전략 비교 → 결과 표
- 실습 4: 임베딩 및 PGVector 저장

### 핵심 코드: 청킹 비교

```python
# 3가지 청킹 전략
chunkers = {
    'Fixed_500': RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
    'Fixed_1000': RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
    'Sentence': # 문장 단위 분할기
}

# 각 전략으로 청킹 후 검색 테스트
# Hit Rate 비교
```

---

## 1-6: RAG 고도화 + QLoRA (Day 6) ⭐

### 세부 설명 리스트

**1. 검색 전략 비교** ⭐
- Dense: 벡터 유사도
- Sparse: BM25 키워드
- Hybrid: 두 점수 결합
- 실험 결과 표

**2. RAG 평가 지표**
- Hit Rate@K: Top K에 정답 포함
- MRR: 정답 순위 역수 평균
- 테스트 Q&A 20개

**3. QLoRA 파인튜닝**
- 왜 필요한지 (도메인 용어)
- 데이터 준비 (100개 Q&A)
- Kaggle T4에서 학습

### 실습 목록
- 실습 1: 검색 전략 비교 → 결과 표
- 실습 2: RAG 평가 (Hit Rate, MRR)
- 실습 3: QLoRA 학습 (Kaggle 노트북)

### 핵심 코드: 검색 비교

```python
# 테스트 쿼리
test_queries = [
    "이상금융거래 보고 의무",
    "FDS 시스템 구축 요건",
    # ...
]

# 각 전략으로 검색
results = {
    'Dense': [],
    'Sparse': [],
    'Hybrid': []
}

# Hit Rate 계산
for strategy, retriever in retrievers.items():
    hits = 0
    for q in test_queries:
        docs = retriever.get_relevant_documents(q, k=3)
        if is_relevant(docs):
            hits += 1
    results[strategy] = hits / len(test_queries)
```

---

## 1-7: Agent + API (Day 7)

### 세부 설명 리스트

**1. LangGraph Agent**
- State 정의 (TypedDict)
- Node 함수 5개
- Edge 연결

**2. FastAPI**
- /health (GET)
- /predict (POST, 동기)
- /analyze (POST, 비동기)
- /result/{task_id} (GET)

**3. Celery**
- Redis 브로커
- Worker 구성
- analyze_task

**4. 통합 테스트**
- E2E 테스트
- 응답 시간 측정

### 실습 목록
- 실습 1: State 정의
- 실습 2: Node 함수 구현
- 실습 3: Graph 연결
- 실습 4: FastAPI 엔드포인트
- 실습 5: Celery 태스크
- 실습 6: 통합 테스트

### 핵심 코드: LangGraph State

```python
from typing import TypedDict, Optional, List

class FDSAgentState(TypedDict):
    transaction: dict
    is_fraud: Optional[bool]
    probability: Optional[float]
    top_features: Optional[List[dict]]
    query: Optional[str]
    regulations: Optional[List[str]]
    explanation: Optional[str]
```

### 핵심 코드: Node 함수

```python
def detect_fraud(state: FDSAgentState) -> FDSAgentState:
    """XGBoost 예측"""
    prob = model.predict_proba([state['transaction']])[0, 1]
    state['probability'] = prob
    state['is_fraud'] = prob > threshold
    return state

def explain_shap(state: FDSAgentState) -> FDSAgentState:
    """SHAP 설명"""
    shap_values = explainer.shap_values([state['transaction']])[0]
    state['top_features'] = get_top_features(shap_values)
    return state

def search_regulations(state: FDSAgentState) -> FDSAgentState:
    """벡터DB 검색"""
    docs = retriever.get_relevant_documents(state['query'], k=3)
    state['regulations'] = [d.page_content for d in docs]
    return state

def generate_report(state: FDSAgentState) -> FDSAgentState:
    """LLM 리포트 생성"""
    prompt = build_prompt(state)
    state['explanation'] = llm.invoke(prompt)
    return state
```

### 핵심 코드: FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictRequest(BaseModel):
    transaction_id: int
    amount: float
    hour: int
    # ...

class AnalyzeResponse(BaseModel):
    task_id: str

@app.post("/analyze")
async def analyze(request: PredictRequest):
    task = analyze_task.delay(request.dict())
    return AnalyzeResponse(task_id=task.id)

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    result = AsyncResult(task_id)
    if result.ready():
        return {"status": "done", "result": result.get()}
    return {"status": "pending"}
```

---

## 전체 요약

| 노트북 | 시간 | Docker | 핵심 산출물 |
|--------|------|--------|------------|
| 1-1 | 3h | ❌ | train.csv, test.csv |
| 1-2 | 3h | ❌ | feature_engineering.py |
| 1-3 | 4h | ❌ | 모델 비교 표, xgb_model.pkl |
| 1-4 | 3h | ❌ | explainer.py, SHAP 시각화 |
| 1-5 | 4h | ✅ | 청킹 비교 표, 벡터 저장 |
| 1-6 | 4h | ✅ | 검색 비교 표, QLoRA 모델 |
| 1-7 | 4h | ✅ | Agent, API, 통합 테스트 |

**총 약 25시간**

---

## 핵심 실험 결과 (면접용)

### 1. 모델 비교 (1-3)

| Model | AUC | Time(s) | SHAP |
|-------|-----|---------|------|
| XGBoost | 0.91 | 45 | ✅ 최상 |
| LightGBM | 0.90 | 32 | ✅ 좋음 |
| CatBoost | 0.90 | 98 | ⚠️ 제한 |

### 2. 청킹 비교 (1-5)

| Strategy | Chunk Size | Hit Rate@3 |
|----------|------------|------------|
| Fixed | 500자 | 70% |
| Fixed | 1000자 | 75% |
| Semantic | 가변 | 85% |

### 3. 검색 비교 (1-6)

| Strategy | Hit Rate@3 | MRR |
|----------|------------|-----|
| Dense | 75% | 0.6 |
| Sparse | 70% | 0.5 |
| Hybrid | 85% | 0.7 |

이 표들이 면접에서 "왜 이걸 선택했나요?"에 대한 근거!
