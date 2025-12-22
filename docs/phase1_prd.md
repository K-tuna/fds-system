# Phase 1: FDS + XAI + RAG 시스템

> 금융 이상거래 탐지 및 규정 기반 설명 시스템

---

## 1. 프로젝트 소개

### 1.1 프로젝트명
**Explainable FDS with Regulatory RAG**

### 1.2 목적
이상거래를 탐지하고, 왜 이상거래인지 SHAP으로 설명하며, 관련 금융 규정을 RAG로 검색하여 근거를 제시하는 통합 시스템

### 1.3 결과물 예시

```
[입력] 거래 정보
- 금액: 500만원
- 시간: 새벽 3시
- 이메일: tempmail.com → gmail.com

[출력]
"이 거래는 87% 확률로 이상거래입니다.

 주요 위험 요인:
 - 야간 거래 (새벽 3시)
 - 고액 거래 (500만원)
 - 이메일 도메인 불일치

 관련 규정:
 전자금융거래법 제9조에 따르면 금융회사는
 이상금융거래 탐지시스템을 구축해야 하며..."
```

### 1.4 왜 이 프로젝트인가?

**금융권 MLE 포지션 타겟팅:**
- 하나금융TI, 우리FIS 등 은행 IT 자회사
- FDS는 모든 금융사의 필수 시스템
- XAI(설명가능 AI)는 금융 규제 트렌드
- RAG는 현재 가장 핫한 LLM 활용 패턴

**포트폴리오 차별화:**
- 단순 구현이 아닌 실험 기반 의사결정
- 모든 기술 선택에 "왜?"에 대한 답변 준비
- 면접에서 깊이 있는 대화 가능

---

## 2. 기술 스택

### 2.1 전체 구성

```
┌─────────────────────────────────────────────────────────────┐
│  ML/XAI: XGBoost + SHAP                                     │
├─────────────────────────────────────────────────────────────┤
│  RAG: LangChain + LangGraph + PGVector + Ollama (Qwen 3B)   │
├─────────────────────────────────────────────────────────────┤
│  API: FastAPI + Celery + Redis                              │
├─────────────────────────────────────────────────────────────┤
│  인프라: Docker Compose + PostgreSQL                        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 상세 버전

| 영역 | 기술 | 버전 | 역할 |
|------|------|------|------|
| ML | XGBoost | 2.1.0 | 이상거래 탐지 모델 |
| ML | SHAP | 0.43+ | 모델 설명 |
| RAG | LangChain | 0.2+ | RAG 파이프라인 |
| RAG | LangGraph | 0.1+ | Agent 오케스트레이션 |
| RAG | PGVector | 0.5+ | 벡터 저장소 |
| LLM | Qwen 2.5 | 3B | 응답 생성 |
| API | FastAPI | 0.110+ | REST API |
| API | Celery | 5.3+ | 비동기 작업 |
| DB | PostgreSQL | 16 | 메타데이터 저장 |
| DB | Redis | 7+ | 브로커 + 캐시 |

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
│  /predict ──────────────────────────────▶ XGBoost (즉시)    │
│                                                             │
│  /analyze ──▶ Celery ──▶ Redis Queue                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Celery Worker                           │
│                                                             │
│  XGBoost ──▶ SHAP ──▶ RAG 검색 ──▶ LLM 생성                 │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        PostgreSQL       PGVector         Ollama
```

### 3.2 LangGraph Agent 흐름

```
START
  │
  ▼
detect_fraud (XGBoost 예측)
  │
  ▼
explain_shap (SHAP 설명)
  │
  ▼
build_query (검색 쿼리 생성)
  │
  ▼
search_regulations (PGVector 검색)
  │
  ▼
generate_report (LLM 최종 리포트)
  │
  ▼
END
```

### 3.3 Docker 컨테이너 (Just-in-Time)

| 서비스 | 이미지 | 포트 | 시작 시점 |
|--------|--------|------|-----------|
| app | 커스텀 | 8000 | Day 1 |
| postgres | pgvector:pg16 | 5432 | Day 5 |
| ollama | ollama/ollama | 11434 | Day 5 |
| redis | redis:7-alpine | 6379 | Day 7 |
| worker | 커스텀 | - | Day 7 |

---

## 4. 기술 선택 이유 (면접용)

### 4.1 FDS 모델: XGBoost

**실험 결과:**

| 모델 | AUC | 학습시간 | SHAP 호환 |
|------|-----|----------|-----------|
| XGBoost | 0.91 | 45s | ✅ 최상 |
| LightGBM | 0.90 | 32s | ✅ 좋음 |
| CatBoost | 0.90 | 98s | ⚠️ 제한적 |

**면접 답변:**
> "정형 데이터에서 트리 기반 모델이 여전히 SOTA입니다. 3개 모델 비교 실험 결과, XGBoost가 AUC 0.91로 가장 높았고 SHAP TreeExplainer와의 호환성도 가장 좋았습니다. LightGBM이 학습은 빨랐지만 성능 차이가 있어 XGBoost를 선택했습니다."

### 4.2 LLM: Qwen 2.5 3B + QLoRA

**선택 이유:**

| 항목 | 선택 | 이유 |
|------|------|------|
| 모델 크기 | 3B | RTX 2070 Super 8GB VRAM 제약 |
| 양자화 | Q4 | ~2-3GB VRAM으로 여유 확보 |
| 파인튜닝 | QLoRA | Kaggle T4에서 학습 가능 |

**면접 답변:**
> "RAG만으로는 금융 도메인 용어 이해가 부족했습니다. '이상금융거래', 'FDS', '전자금융거래법' 같은 용어를 정확히 이해하도록 QLoRA로 도메인 특화 파인튜닝을 했습니다. 로컬 GPU(8GB VRAM) 제약으로 3B 모델을 선택했고, 파인튜닝은 Kaggle T4에서 진행했습니다."

### 4.3 벡터 DB: PGVector

**비교:**

| 옵션 | 비용 | 적합 규모 | 인프라 |
|------|------|-----------|--------|
| PGVector | 무료 | 수천~수만 | PostgreSQL 통합 |
| Pinecone | $70+/월 | 수천만+ | 별도 서비스 |
| Chroma | 무료 | 수천 | 임베디드 |

**면접 답변:**
> "현재 규모(수천 벡터)에서 Pinecone은 오버엔지니어링입니다. PGVector는 PostgreSQL과 통합되어 메타데이터와 벡터를 한 DB에서 관리할 수 있고, 인프라가 단순해집니다. 규모가 커지면 Pinecone 마이그레이션을 고려할 수 있습니다."

### 4.4 브로커: Redis

**비교:**

| 옵션 | 역할 | 장점 |
|------|------|------|
| Redis | 브로커 + 캐시 + 결과저장 | 일석삼조 |
| RabbitMQ | 브로커 전용 | 메시지 보장 강함 |

**면접 답변:**
> "Redis 하나로 Celery 브로커, API 응답 캐시, 작업 결과 저장까지 해결합니다. RabbitMQ가 메시지 보장은 더 강하지만, 현재 규모에서는 Redis의 다목적 활용이 더 효율적입니다. 컨테이너 하나로 세 가지 역할을 하는 일석삼조 선택입니다."

### 4.5 비동기: Celery

**면접 답변:**
> "LLM 응답이 2-3초 걸리는데, 동기 처리하면 API 스레드가 블로킹됩니다. Celery로 비동기 처리해서 POST /analyze는 즉시 task_id를 반환하고, 클라이언트는 GET /result/{task_id}로 결과를 polling합니다. 이게 현업에서 LLM 서비스를 처리하는 일반적인 패턴입니다."

---

## 5. 성공 기준

| 항목 | 기준 | 측정 방법 |
|------|------|-----------|
| FDS AUC | >= 0.90 | 테스트셋 평가 |
| /predict 응답 | < 100ms | API 벤치마크 |
| /analyze 응답 | < 3초 | E2E 테스트 |
| RAG Hit Rate | Top 3에 관련 규정 | 테스트 Q&A 20개 |
| Docker 실행 | 단일 명령어 | docker compose up |

---

## 6. 일정 및 학습 가이드

### 6.1 전체 일정

| Day | 주제 | Docker | 핵심 학습 |
|-----|------|--------|-----------|
| 1 | 데이터 + EDA | ❌ | Pandas, 불균형 데이터 |
| 2 | Feature + Baseline | ❌ | Feature Engineering |
| 3 | 모델 고도화 ⭐ | ❌ | 모델 비교, Threshold |
| 4 | SHAP 설명 | ❌ | XAI, SHAP |
| 5 | RAG 환경 ⭐ | ✅ | 청킹, 임베딩 |
| 6 | RAG 고도화 ⭐ | ✅ | 검색 전략, QLoRA |
| 7 | Agent + API | ✅ | LangGraph, Celery |

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

#### Day 2: Feature Engineering + Baseline

**📚 핵심 개념**

1. **Feature Engineering 패턴**
   - 시간 피처: 요일, 시간대, 주말 여부
   - 금액 피처: 로그 변환, 구간화
   - 집계 피처: card_id별 거래 횟수, 평균 금액
   - 범주형 인코딩: Label Encoding, Target Encoding

2. **시간 기반 분할 (Time-based Split)**
   - 미래 데이터로 과거를 예측하면 Data Leakage
   - 시간순 정렬 후 분할

**✏️ 미니 연습**
- Q1: 왜 랜덤 분할이 아닌 시간 기반 분할을 하나요?
- Q2: Target Encoding의 장단점은?

**🎯 면접 Q&A**

Q: "어떤 피처가 가장 중요했나요?"
> "SHAP 분석 결과 거래 금액, 거래 시간대, 이메일 도메인 일치 여부가 Top 3였습니다. 특히 '거래 금액의 로그값'과 'card_id별 최근 거래 횟수' 같은 파생 피처가 Raw 피처보다 중요도가 높았습니다."

Q: "Data Leakage를 어떻게 방지했나요?"
> "시간 기반 분할을 사용했습니다. 전체 데이터를 시간순 정렬 후 앞 80%를 학습, 뒤 20%를 검증에 사용했습니다. 집계 피처도 해당 시점까지의 데이터만 사용해서 미래 정보가 누출되지 않도록 했습니다."

---

#### Day 3: 모델 고도화 + 평가 ⭐

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
> "3개 모델(XGBoost, LightGBM, CatBoost)을 동일 조건에서 비교 실험했습니다. XGBoost가 AUC 0.91로 가장 높았고, SHAP TreeExplainer와의 호환성도 가장 좋았습니다. 학습 시간은 LightGBM이 빨랐지만 성능 차이(0.01)가 있어 XGBoost를 선택했습니다."

Q: "Threshold는 어떻게 정했나요?"
> "FDS에서 FN(놓친 사기)이 FP(오탐)보다 비용이 훨씬 큽니다. FN:FP 비용을 10:1로 가정하고, 비용 함수를 최소화하는 Threshold를 계산했습니다. 결과적으로 0.35가 최적이었고, 이 때 Recall 0.85, Precision 0.45를 달성했습니다."

---

#### Day 4: SHAP + 설명

**📚 핵심 개념**

1. **SHAP (SHapley Additive exPlanations)**
   - 게임 이론의 Shapley Value 기반
   - 각 피처가 예측에 기여한 정도를 계산
   - TreeExplainer: 트리 모델 특화, 빠름

2. **SHAP 시각화**
   - Waterfall Plot: 단일 예측 설명
   - Force Plot: 피처 기여도 시각화
   - Summary Plot: 전체 피처 중요도

3. **설명 템플릿**
   - SHAP 값을 자연어로 변환
   - 피처별 설명 매핑 정의

**✏️ 미니 연습**
- Q1: SHAP 값이 양수/음수일 때 각각 무슨 의미?
- Q2: LIME과 SHAP의 차이는?

**🎯 면접 Q&A**

Q: "왜 SHAP을 선택했나요?"
> "LIME, SHAP, Feature Importance 중 SHAP을 선택했습니다. SHAP은 이론적 기반(Shapley Value)이 탄탄하고, 로컬(개별 예측)과 글로벌(전체 모델) 설명을 모두 제공합니다. TreeExplainer로 XGBoost와 호환이 좋고 속도도 빠릅니다."

Q: "XAI가 금융에서 왜 중요한가요?"
> "금융 규제에서 '설명 가능한 AI'를 점점 요구하고 있습니다. EU AI Act, 한국 금융위 가이드라인 모두 고위험 AI 시스템에 설명 의무를 부과합니다. 고객에게 '왜 거래가 차단됐는지' 설명해야 하고, 감사 시 모델 판단 근거를 제시해야 합니다."

---

#### Day 5: RAG 환경 + 청킹 실험 ⭐

**📚 핵심 개념**

1. **RAG (Retrieval-Augmented Generation)**
   - 검색(Retrieval) + 생성(Generation) 결합
   - LLM의 할루시네이션 감소
   - 도메인 지식을 동적으로 주입

2. **청킹 (Chunking) 전략**
   - Fixed Chunking: 고정 크기 (500자)
   - Semantic Chunking: 의미 단위
   - Sentence Chunking: 문장 단위

3. **임베딩 (Embedding)**
   - 텍스트를 벡터로 변환
   - BGE-M3: 다국어 지원, 한국어 성능 좋음
   - 유사도 검색의 기반

**✏️ 미니 연습**
- Q1: 청크 크기가 너무 크거나 작으면 어떤 문제?
- Q2: 코사인 유사도 vs 유클리드 거리?

**🎯 면접 Q&A**

Q: "왜 이 청킹 전략을 선택했나요?"
> "3가지 전략(Fixed, Semantic, Sentence)을 비교 실험했습니다. 법률 문서는 조항 단위로 의미가 완결되어 Semantic Chunking이 검색 정확도가 가장 높았습니다. Fixed 500자는 문맥이 잘리는 경우가 많았고, Sentence는 너무 짧아 정보가 부족했습니다."

Q: "임베딩 모델은 왜 BGE-M3를 선택했나요?"
> "한국어 금융 문서를 처리해야 해서 다국어 모델이 필수였습니다. BGE-M3는 한국어 성능이 좋고, Dense + Sparse 임베딩을 모두 지원해서 Hybrid Search에 유리합니다. OpenAI 임베딩 대비 비용도 무료입니다."

---

#### Day 6: RAG 고도화 + QLoRA ⭐

**📚 핵심 개념**

1. **검색 전략**
   - Dense Search: 벡터 유사도 (의미 검색)
   - Sparse Search: BM25 (키워드 매칭)
   - Hybrid Search: Dense + Sparse 결합

2. **RAG 평가 지표**
   - Hit Rate@K: Top K에 정답 포함 비율
   - MRR: 정답 순위의 역수 평균
   - 할루시네이션 체크: Citation 검증

3. **QLoRA 파인튜닝**
   - LoRA: Low-Rank Adaptation (일부 파라미터만 학습)
   - QLoRA: 4-bit 양자화 + LoRA
   - 적은 VRAM으로 파인튜닝 가능

**✏️ 미니 연습**
- Q1: Hybrid Search에서 Dense와 Sparse 점수를 어떻게 결합?
- Q2: LoRA의 rank(r)가 크면 어떻게 되나?

**🎯 면접 Q&A**

Q: "왜 Hybrid Search를 선택했나요?"
> "Dense, Sparse, Hybrid 3가지를 비교 실험했습니다. Dense는 '이상금융거래'를 '비정상 거래'로도 검색하지만 '제9조' 같은 정확한 키워드를 놓칩니다. BM25는 키워드에 강하지만 유의어를 못 찾습니다. Hybrid로 두 장점을 결합해서 Hit Rate가 가장 높았습니다."

Q: "RAG 성능은 어떻게 측정했나요?"
> "20개 테스트 Q&A 세트를 만들어 평가했습니다. Hit Rate@3 85%(상위 3개에 정답 포함), MRR 0.7을 달성했습니다. 할루시네이션은 응답에 포함된 Citation이 실제 문서에 있는지 검증하는 방식으로 체크했습니다."

Q: "왜 파인튜닝을 했나요?"
> "RAG만으로는 금융 도메인 용어 이해가 부족했습니다. '이상금융거래탐지시스템', 'FDS', '전자금융거래법' 같은 용어를 Base 모델이 정확히 이해하지 못했습니다. 100개 금융 Q&A 데이터로 QLoRA 파인튜닝해서 도메인 용어 정확도를 높였습니다."

---

#### Day 7: Agent + API + Celery

**📚 핵심 개념**

1. **LangGraph Agent**
   - State: 전체 상태 관리
   - Node: 개별 작업 (예측, 설명, 검색, 생성)
   - Edge: 노드 간 연결 및 조건 분기

2. **FastAPI 비동기 처리**
   - /predict: 동기, XGBoost만 (빠름)
   - /analyze: 비동기, 전체 파이프라인 (느림)

3. **Celery 태스크 큐**
   - Broker: 작업 큐 (Redis)
   - Worker: 실제 작업 실행
   - Result Backend: 결과 저장 (Redis)

**✏️ 미니 연습**
- Q1: LangGraph vs LangChain Agent의 차이?
- Q2: Celery Worker가 죽으면 작업은 어떻게 되나?

**🎯 면접 Q&A**

Q: "왜 Celery를 사용했나요?"
> "LLM 응답이 2-3초 걸리는데, 동기 처리하면 FastAPI 워커가 블로킹됩니다. 동시 요청 10개만 와도 응답 지연이 심해집니다. Celery로 비동기 처리해서 API는 즉시 task_id를 반환하고, 결과는 polling 방식으로 조회합니다."

Q: "LangGraph를 왜 선택했나요?"
> "단순 RAG Chain보다 복잡한 워크플로우가 필요했습니다. 예측 → 설명 → 검색 → 생성의 순차 흐름에 조건 분기(사기 확률 낮으면 RAG 스킵)도 추가해야 했습니다. LangGraph는 State 기반으로 복잡한 흐름을 명확하게 정의할 수 있습니다."

---

## 7. 차별화 포인트 총정리

| # | 포인트 | 관련 태스크 | 면접 질문 |
|---|--------|-------------|-----------|
| 1 | 모델 비교 실험 | Day 3 | "왜 XGBoost?" |
| 2 | Threshold 분석 | Day 3 | "Threshold 어떻게 정했나?" |
| 3 | 청킹 전략 비교 | Day 5 | "왜 이 청킹 전략?" |
| 4 | 검색 전략 비교 | Day 6 | "왜 Hybrid Search?" |
| 5 | RAG 품질 평가 | Day 6 | "RAG 성능 어떻게 측정?" |
| 6 | QLoRA 파인튜닝 | Day 6 | "왜 파인튜닝?" |
| 7 | Celery 비동기 | Day 7 | "왜 Celery?" |

**핵심 메시지:** 
> "단순히 기술을 갖다 쓴 게 아니라, 실험으로 검증하고 근거를 가지고 선택했습니다."

---

## 8. 하드웨어 요구사항

| 구성요소 | 필요 사양 | 용도 |
|----------|-----------|------|
| GPU | RTX 2070 Super (8GB) | XGBoost GPU, Qwen 3B |
| RAM | 16GB+ | 데이터 로딩 |
| Storage | 20GB+ | 데이터 + 모델 |
| Kaggle | T4 GPU (16GB) | QLoRA 파인튜닝 |

---

## 9. 다음 단계 (Phase 2 미리보기)

Phase 1 완료 후 Phase 2에서 추가할 MLOps 요소:

- **Airflow**: 데이터 파이프라인 스케줄링
- **MLflow**: 모델 버전 관리, 실험 추적
- **Evidently**: 데이터/모델 드리프트 모니터링
- **Kafka**: 실시간 스트리밍 (배치 → 실시간)

---

## 10. 참고 자료

**데이터셋:**
- IEEE-CIS Fraud Detection (Kaggle)

**금융 규정:**
- 전자금융거래법
- 금융위 FDS 가이드라인
- 금융보안원 이상금융거래탐지 지침

**기술 문서:**
- XGBoost 공식 문서
- SHAP 공식 문서
- LangChain / LangGraph 공식 문서
- Ollama 공식 문서
