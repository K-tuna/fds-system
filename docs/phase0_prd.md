# Phase 0: 기초 학습

> Phase 1 시작 전에 필요한 사전 지식

---

## 개요

### 목적
Phase 1 (FDS+RAG 구현) 코드를 이해하기 위한 기초 학습

### 대상
- Python 기본 문법은 아는 상태
- 클래스, numpy, pandas, ML은 처음

### 총 학습 시간
약 14.5시간

---

## 학습 목차

| 번호 | 주제 | 시간 | Phase 1 연관 |
|------|------|------|-------------|
| 0-1 | 클래스 + 타입 힌트 | 1h | 전체 코드 구조 |
| 0-2 | Numpy | 1.5h | 데이터 연산 |
| 0-3 | Pandas | 2.5h | 데이터 전처리 |
| 0-4 | Matplotlib | 1h | EDA 시각화 |
| 0-5 | ML 개념 + Sklearn | 2.5h | 모델 학습/평가 |
| 0-6 | 모델 저장/튜닝/설명 | 1h | XGBoost, SHAP |
| 0-7 | LLM/RAG 개념 | 2h | RAG 파이프라인 |
| 0-8 | LangChain/LangGraph | 1.5h | Agent 구현 |
| 0-9 | FastAPI + 비동기 | 1.5h | API 서빙 |

---

## 0-1: 클래스 + 타입 힌트 (1시간)

### 📚 학습 내용

**클래스 기초**
- `__init__`: 객체 생성 시 초기화
- `self`: 객체 자신을 가리킴
- 메서드: 클래스 안의 함수

**왜 배우나?**
sklearn, XGBoost 등 ML 라이브러리가 전부 클래스 기반:
```python
model = XGBClassifier()  # 객체 생성 (__init__ 호출)
model.fit(X, y)          # 메서드 호출
model.predict(X)         # 메서드 호출
```

**타입 힌트**
- `List`, `Dict`, `Optional`: 타입 명시
- `TypedDict`: 딕셔너리 구조 정의
- 함수 파라미터/리턴 타입 표기

### ✅ 체크포인트
- [ ] 클래스 정의하고 객체 생성할 수 있다
- [ ] `__init__`과 `self` 역할을 설명할 수 있다
- [ ] `List[str]`, `Optional[int]` 의미를 안다

---

## 0-2: Numpy (1.5시간)

### 📚 학습 내용

**배열 기초**
- `np.array()`: 배열 생성
- 인덱싱, 슬라이싱
- `shape`, `dtype`

**연산**
- `mean()`, `sum()`, `std()`
- 배열 간 연산 (브로드캐스팅)
- 조건 필터링: `arr[arr > 0]`

**왜 배우나?**
ML에서 데이터는 전부 numpy 배열로 처리됨:
```python
y_true = np.array([0, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1])
accuracy = (y_true == y_pred).mean()  # 0.75
```

### ✅ 체크포인트
- [ ] 배열 생성하고 인덱싱할 수 있다
- [ ] `mean()`, `sum()` 등 집계 연산을 쓸 수 있다
- [ ] 조건 필터링 (`arr[arr > 0]`)을 이해한다

---

## 0-3: Pandas (2.5시간)

### 📚 학습 내용

**DataFrame 기초**
- `pd.read_csv()`: CSV 읽기
- 컬럼 선택: `df['col']`, `df[['a', 'b']]`
- 행 필터링: `df[df['col'] > 0]`

**데이터 조작**
- `groupby()`: 그룹별 집계
- `merge()`: 테이블 조인
- `fillna()`, `dropna()`: 결측치 처리

**왜 배우나?**
데이터 전처리의 90%가 pandas:
```python
df = pd.read_csv('transactions.csv')
fraud_rate = df.groupby('card_type')['isFraud'].mean()
```

### ✅ 체크포인트
- [ ] CSV 읽고 컬럼 선택할 수 있다
- [ ] 조건으로 행 필터링할 수 있다
- [ ] `groupby()`로 집계할 수 있다
- [ ] 두 테이블을 `merge()`로 합칠 수 있다

---

## 0-4: Matplotlib (1시간)

### 📚 학습 내용

**기본 그래프**
- `plt.hist()`: 히스토그램
- `plt.bar()`: 막대 그래프
- `plt.plot()`: 선 그래프

**꾸미기**
- `plt.title()`, `plt.xlabel()`, `plt.ylabel()`
- `plt.subplot()`: 여러 그래프 배치

**왜 배우나?**
EDA에서 데이터 분포 확인:
```python
plt.hist(df['TransactionAmt'], bins=50)
plt.title('거래 금액 분포')
```

### ✅ 체크포인트
- [ ] 히스토그램, 막대 그래프를 그릴 수 있다
- [ ] 제목, 축 레이블을 추가할 수 있다
- [ ] subplot으로 여러 그래프를 배치할 수 있다

---

## 0-5: ML 개념 + Sklearn (2.5시간)

### 📚 학습 내용

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

### ✅ 체크포인트
- [ ] 분류와 회귀의 차이를 안다
- [ ] 왜 train/test를 나누는지 설명할 수 있다
- [ ] `fit()` → `predict()` 흐름을 안다
- [ ] Recall, Precision, AUC 의미를 안다
- [ ] 결정트리 → 랜덤포레스트 → XGBoost 관계를 안다

---

## 0-6: 모델 저장/튜닝/설명 (1시간)

### 📚 학습 내용

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

### ✅ 체크포인트
- [ ] joblib로 모델 저장/로드할 수 있다
- [ ] 하이퍼파라미터가 뭔지 안다
- [ ] SHAP이 왜 필요한지 설명할 수 있다

---

## 0-7: LLM/RAG 개념 (2시간)

### 📚 학습 내용

**LLM 기초**
- LLM: 대규모 언어 모델 (GPT, Claude, Qwen 등)
- 토큰: 텍스트를 쪼갠 단위
- 프롬프트: LLM에 주는 입력

**임베딩**
- 텍스트를 숫자 벡터로 변환
- 유사한 의미 → 가까운 벡터
- 벡터 검색의 기반

**RAG (Retrieval-Augmented Generation)**
```
질문 → 검색 (관련 문서 찾기) → 생성 (LLM이 답변)
```
- 왜 필요? LLM이 모르는 정보 (최신, 내부 문서) 활용
- 할루시네이션 감소

**벡터DB**
- 벡터를 저장하고 유사도 검색
- PGVector, Pinecone, Chroma 등

### ✅ 체크포인트
- [ ] LLM이 뭔지 설명할 수 있다
- [ ] 임베딩이 뭔지 안다
- [ ] RAG 흐름 (검색 → 생성)을 이해한다
- [ ] 벡터DB 역할을 안다

---

## 0-8: LangChain/LangGraph (1.5시간)

### 📚 학습 내용

**LangChain**
- LLM 애플리케이션 프레임워크
- 주요 구성: Chain, Prompt, LLM, Retriever

```python
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)
result = chain.invoke("질문")
```

**LangGraph**
- 복잡한 워크플로우를 그래프로 정의
- State: 전체 상태 관리
- Node: 개별 작업
- Edge: 노드 간 연결

```
START → detect → explain → search → generate → END
```

**왜 LangGraph?**
- 단순 Chain: 선형 흐름만 가능
- LangGraph: 조건 분기, 반복 가능

### ✅ 체크포인트
- [ ] LangChain이 뭔지 안다
- [ ] Chain, Prompt, Retriever 역할을 안다
- [ ] LangGraph의 State, Node, Edge 개념을 안다
- [ ] 왜 LangGraph를 쓰는지 설명할 수 있다

---

## 0-9: FastAPI + 비동기 (1.5시간)

### 📚 학습 내용

**REST API 개념**
- API: 프로그램 간 통신 인터페이스
- REST: HTTP 기반 API 설계 방식
- GET, POST, PUT, DELETE

**FastAPI**
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(data: InputSchema):
    return {"result": ...}
```

**Pydantic**
- 데이터 검증 라이브러리
- FastAPI에서 요청/응답 스키마 정의

**비동기 (async/await)**
- 오래 걸리는 작업 중 다른 요청 처리 가능
- `async def`, `await`

**Celery + Redis**
- Celery: 백그라운드 태스크 큐
- Redis: 브로커 (작업 전달) + 결과 저장
- LLM처럼 오래 걸리는 작업을 비동기로 처리

```
POST /analyze → task_id 반환 (즉시)
GET /result/{task_id} → 결과 조회 (완료 후)
```

### ✅ 체크포인트
- [ ] REST API가 뭔지 안다
- [ ] FastAPI로 간단한 엔드포인트를 만들 수 있다
- [ ] Pydantic으로 스키마를 정의할 수 있다
- [ ] 왜 Celery를 쓰는지 설명할 수 있다

---

## 학습 완료 후

### 전체 체크리스트
- [ ] 0-1: 클래스 + 타입 힌트
- [ ] 0-2: Numpy
- [ ] 0-3: Pandas
- [ ] 0-4: Matplotlib
- [ ] 0-5: ML 개념 + Sklearn
- [ ] 0-6: 모델 저장/튜닝/설명
- [ ] 0-7: LLM/RAG 개념
- [ ] 0-8: LangChain/LangGraph
- [ ] 0-9: FastAPI + 비동기

### 다음 단계
Phase 0 완료 → Phase 1 시작 (FDS+RAG 구현)
