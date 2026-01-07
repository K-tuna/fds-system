# FDS System - Explainable FDS with Ensemble Learning

## ⚠️ 필독 사항

### 개발 환경
- **OS**: Windows
- **GPU**: RTX 2070 Super (8GB VRAM)
- Windows 명령어 사용 주의 (bash 명령어 호환 안될 수 있음)

### 문서 위치 (반드시 읽을 것)
작업 전 반드시 해당 Phase 문서를 읽고 진행:
- `docs/roadmap.md` - **전체 로드맵** ⭐⭐ (Phase 1~5)
- `docs/phase0_study.md` - Phase 0 학습 개요
- `docs/phase0_impl.md` - Phase 0 구현 상세
- `docs/phase1_study.md` - Phase 1 사전학습 (ML 개념)
- `docs/phase1_prd.md` - Phase 1 기획 ⭐
- `docs/phase1_impl.md` - Phase 1 구현 상세

---

## 📍 현재 진행 상황

**마지막 업데이트**: 1-12 하이브리드 서빙 + 5개 모델 비교 완료

### Phase 0 (완료)
| 섹션 | 상태 |
|------|------|
| 0-0 환경 세팅 | ✅ |
| 0-1 클래스 + 타입 힌트 | ✅ |
| 0-2 Numpy | ✅ |
| 0-3 Pandas | ✅ |
| 0-4 Matplotlib | ✅ |

### Phase 1 구현 진행 (완료)

| 노트북 | 주제 | 상태 |
|--------|------|------|
| 1-1 | EDA | ✅ |
| 1-2 | Feature Engineering | ✅ |
| 1-3 | XGBoost (Optuna GPU) | ✅ |
| 1-4 | LSTM | ✅ |
| 1-5 | Ensemble | ✅ |
| 1-6 | SHAP | ✅ |
| 1-7 | FastAPI + Docker | ✅ |
| 1-8 | React Admin | ✅ |
| 1-9 | 트리 스태킹 ⭐⭐ | ✅ |
| 1-S10 | Transformer 학습 | ✅ |
| 1-10 | FT-Transformer 구현 | ✅ |
| 1-11 | PaySim 공정 비교 | ✅ |
| 1-12 | 하이브리드 서빙 ⭐⭐ | ✅ |

**다음 작업**: Phase 2 MLOps + 모니터링 진행

---

## 프로젝트 개요

금융 이상거래 탐지(FDS) 시스템. XGBoost 기반 + SHAP 설명.
- 정형 특성 탐지 (XGBoost) - AUC 0.91, Recall 90.55%
- LSTM 실험 완료 (AUC 0.70 → 앙상블 효과 미미, API 미적용)
- SHAP 기반 설명 (XAI)

## 기술 스택

| 영역 | 기술 |
|------|------|
| ML | XGBoost, PyTorch (LSTM), SHAP |
| API | FastAPI |
| Infra | Docker Compose |

## 프로젝트 구조

```
fds-system/
├── docs/                    # PRD 및 구현 가이드 ⭐ 필독
├── notebooks/
│   ├── phase0/             # 기초 학습 (0-0 ~ 0-4)
│   └── phase1/
│       ├── study/          # Phase 1 사전학습 (1-S1 ~ 1-S5)
│       └── (구현 노트북)    # 1-1 ~ 1-7
├── src/
│   ├── ml/                 # feature_engineering, xgboost, lstm, ensemble
│   ├── explainer/          # shap 설명 모듈
│   └── api/                # FastAPI main, schemas
├── data/
│   ├── raw/                # IEEE-CIS 원본
│   └── processed/          # 전처리 데이터
├── models/                 # 학습된 모델 (.pkl, .pt)
├── docker-compose.yml
└── requirements.txt
```

## Phase 구성 (상세: docs/roadmap.md)

### Phase 0: 공통 기초 (6시간)
0-0 환경세팅 → 0-1 클래스/타입힌트 → 0-2 Numpy → 0-3 Pandas → 0-4 Matplotlib

### Phase 1: 학습 ↔ 구현 사이클 ⭐

**원칙**: 학습 → 바로 구현 → 다시 학습 → 구현 (사이클)

| 학습 (Study) | 구현 (Impl) |
|--------------|-------------|
| 1-S1 ML/Sklearn | → 1-1 EDA |
| 1-S2 모델튜닝/SHAP | → 1-2 Feature Eng |
| 1-S3 XGBoost | → 1-3 XGBoost |
| 1-S4 LSTM | → 1-4 LSTM |
| 1-S5 앙상블 | → 1-5 Ensemble |
| (이미 배움) | → 1-6 SHAP, 1-7 FastAPI |
| 1-S8 React | → 1-8 React Admin |
| **1-S9 스태킹** | → **1-9 트리 스태킹** ⭐⭐ |
| 1-S10 Transformer | → 1-10 Transformer (선택) |
| (기존 지식 활용) | → 1-11 PaySim 공정 비교 (선택) |
| (기존 지식 활용) | → 1-12 하이브리드 서빙 (선택) |

### Phase 2: MLOps + 모니터링 (6일)
MLflow → Evidently → 비용최적화 → GitHub Actions → A/B테스트 → Prometheus/Grafana

### Phase 3: 실시간 + 워크플로 (5일)
Kafka → Airflow → Feast → ONNX

### Phase 4: 클라우드 + 인프라 (5일)
BigQuery → Kubernetes → S3/MinIO → Spark

### Phase 5: 고급 + 차별화 (5일+)
GNN → Kubeflow → Flink → ELK

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

# Phase 0 후반 + Phase 1
pip install xgboost optuna shap
pip install torch  # LSTM용

# Phase 1 API
pip install fastapi uvicorn
```

## 핵심 면접 포인트

1. **XGBoost 선택**: 정형 데이터에서 AUC 0.91 + SHAP 호환성
2. **4단계 위험도**: approve/verify/hold/block (Threshold 0.18 기준)
3. **Recall 90.55%**: 현업 기준(80-95%) 충족
4. **Threshold 최적화**: FN:FP = 10:1 비용 기반 → 0.18
5. **SHAP 설명**: TreeExplainer로 Top 5 피처 설명
6. **학습-서빙 일관성**: API 설계 결함 발견 및 해결 경험
7. **앙상블 실험**: LSTM 추가 시 +0.12% → 복잡도 대비 효과 없어 XGBoost 단독 채택
8. **트리 스태킹**: XGBoost + LightGBM + CatBoost → AUC 0.92, Recall 71% @5%FPR + 확률 분포 양극화
9. **LSTM 실패 분석**: IEEE-CIS 데이터 특성 (PCA 정적 피처) → PaySim으로 재검증
10. **PaySim 공정 비교** (선택): 4모델 비교 + 12개 시간 윈도우 집계 피처 + 추론 속도 벤치마크
11. **하이브리드 서빙**: NVIDIA AI Blueprint 2024 패턴 (FT-Transformer 임베딩 → Redis 캐싱 → XGBoost) - 추론 24배 개선 (24ms → 1ms)
12. **5개 모델 공정 비교**: PaySim에서 XGBoost/Transformer/하이브리드/스태킹/하이브리드스태킹 비교 → 스태킹 AUC 최고(0.9998), 하이브리드 Recall 최고(99.95%)
13. **하이브리드 스태킹 실패 분석**: LogisticRegression 과적합 + 정보 중복 → 오히려 성능 하락 (0.9992) 원인 분석

## 데이터

- IEEE-CIS Fraud Detection (Kaggle)
