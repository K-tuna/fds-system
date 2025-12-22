# FDS System - Explainable FDS with Regulatory RAG

## ⚠️ 필독 사항

### 개발 환경
- **OS**: Windows
- **GPU**: RTX 2070 Super (8GB VRAM)
- Windows 명령어 사용 주의 (bash 명령어 호환 안될 수 있음)

### 문서 위치 (반드시 읽을 것)
작업 전 반드시 해당 Phase 문서를 읽고 진행:
- `docs/phase0_prd.md` - Phase 0 기획
- `docs/phase0_impl.md` - Phase 0 구현 상세 ⭐
- `docs/phase1_prd.md` - Phase 1 기획
- `docs/phase1_impl.md` - Phase 1 구현 상세 ⭐
- `docs/1-2_impl_example.md` - 노트북 구현 예시

---

## 📍 현재 진행 상황

**마지막 업데이트**: Phase 0-1 클래스 + 타입 힌트 노트북 완료

| Phase | 섹션 | 상태 |
|-------|------|------|
| 세팅 | 폴더 구조, CLAUDE.md, Serena | ✅ 완료 |
| Phase 0 | 0-0 환경 세팅 | ✅ 완료 |
| Phase 0 | 0-1 클래스 + 타입 힌트 | ✅ 완료 |
| Phase 0 | 0-2 Numpy | ⏳ 시작 전 |
| Phase 1 | - | ⏳ 시작 전 |

**다음 작업**: Phase 0-2 Numpy 노트북 생성

---

## 프로젝트 개요

금융 이상거래 탐지(FDS) 시스템. XGBoost + SHAP + RAG 결합.
- 이상거래 탐지 (XGBoost)
- SHAP 기반 설명 (XAI)
- 금융 규정 검색 및 근거 제시 (RAG)

## 기술 스택

| 영역 | 기술 |
|------|------|
| ML | XGBoost, SHAP |
| RAG | LangChain, LangGraph, PGVector |
| LLM | Qwen 2.5 3B (Ollama) - 8GB VRAM 제약 |
| API | FastAPI, Celery, Redis |
| DB | PostgreSQL (PGVector) |
| Infra | Docker Compose |

## 프로젝트 구조

```
fds-system/
├── docs/                    # PRD 및 구현 가이드 ⭐ 필독
├── notebooks/
│   ├── phase0/             # 기초 학습 (0-0 ~ 0-9)
│   └── phase1/             # FDS 구현 (1-1 ~ 1-7)
├── src/
│   ├── ml/                 # feature_engineering, model, explainer
│   ├── rag/                # chunking, embedding, retriever, generator
│   ├── agent/              # state, nodes, graph
│   └── api/                # main, schemas, tasks
├── data/
│   ├── raw/                # IEEE-CIS 원본
│   └── processed/          # 전처리 데이터
├── models/                 # 학습된 모델 (.pkl)
├── docker-compose.yml
└── requirements.txt
```

## Phase 구성

### Phase 0: 기초 학습 (~14.5시간)
0-0 환경세팅 → 0-1 클래스/타입힌트 → 0-2 Numpy → 0-3 Pandas →
0-4 Matplotlib → 0-5 ML/Sklearn → 0-6 모델튜닝 → 0-7 LLM/RAG →
0-8 LangChain → 0-9 FastAPI

### Phase 1: FDS 구현 (~25시간)
1-1 EDA → 1-2 Feature Engineering → 1-3 모델 고도화 → 1-4 SHAP →
1-5 RAG 환경 → 1-6 RAG 고도화 → 1-7 Agent/API

## 개발 규칙

### 노트북 패턴
```
[마크다운] 개념 설명
[코드] 예제 (완성본)
[코드] 실습 TODO (빈칸)
[코드] 실습 정답
[코드] 체크포인트 (assert)
```

### 코드 스타일
- Python 3.11
- 타입 힌트 필수
- 검증된 코드는 src/로 모듈화

## 환경 설정

```bash
conda create -n fds python=3.11 -y
conda activate fds

# Phase 0 기본
pip install numpy pandas matplotlib scikit-learn

# Phase 0 후반
pip install xgboost optuna shap

# Phase 1 RAG/API
pip install langchain langchain-community langgraph
pip install fastapi uvicorn celery redis
pip install pgvector psycopg2-binary
```

## 핵심 면접 포인트

1. **XGBoost 선택**: AUC 최고 + SHAP 호환성
2. **Threshold 최적화**: FN:FP = 10:1 비용 기반
3. **청킹**: Semantic Chunking (법률 문서 특성)
4. **검색**: Hybrid Search (Dense + Sparse)
5. **비동기**: Celery (LLM 2-3초 지연 처리)

## 데이터

- IEEE-CIS Fraud Detection (Kaggle)
- 전자금융거래법, 금융위 FDS 가이드라인
