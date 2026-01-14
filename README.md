# 설명 가능한 금융 이상거래 탐지 시스템 (FDS)

XGBoost + SHAP 기반 이상거래 탐지 시스템 with MLOps

## 핵심 성과

| 지표 | 값 | 설명 |
|------|-----|------|
| AUC | **0.91** | IEEE-CIS 데이터셋 기준 |
| Recall | **90.55%** | 현업 기준(80-95%) 충족 |
| Threshold | **0.18** | FN:FP = 20:1 비용 기반 최적화 |

## 기술 스택

| 영역 | 기술 |
|------|------|
| ML | XGBoost, PyTorch (LSTM/Transformer), SHAP |
| MLOps | MLflow (Tracking, Registry), Evidently |
| API | FastAPI, Docker Compose |
| 테스트 | pytest |

## 아키텍처

```
IEEE-CIS 데이터
       ↓
┌──────────────────────────────────────────┐
│  Feature Engineering (12개 시간 윈도우)   │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  XGBoost (Optuna GPU 튜닝)               │
│  → Threshold 0.18 (비용 최적화)          │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  SHAP TreeExplainer (Top 5 피처 설명)    │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  FastAPI + Docker Compose                │
│  → 4단계 위험도: approve/verify/hold/block │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  MLOps: MLflow Tracking + Evidently      │
└──────────────────────────────────────────┘
```

## 프로젝트 구조

```
fds-system/
├── notebooks/
│   ├── phase0/         # 기초 학습 (Python, Numpy, Pandas)
│   ├── phase1/         # ML 구현 (EDA → XGBoost → SHAP → API)
│   └── phase2/         # MLOps (MLflow, Evidently, 비용 최적화)
├── src/
│   ├── ml/             # 피처 엔지니어링, 모델
│   ├── explainer/      # SHAP 설명 모듈
│   └── api/            # FastAPI
├── models/             # 학습된 모델 (.joblib)
├── tests/              # pytest 테스트
└── docs/               # PRD 및 구현 가이드
```

## 주요 구현 사항

### Phase 1: ML 파이프라인
- **1-1~1-2**: EDA + Feature Engineering
- **1-3**: XGBoost (Optuna GPU 튜닝)
- **1-6**: SHAP 기반 설명 (XAI)
- **1-7~1-8**: FastAPI + React Admin
- **1-9**: 트리 스태킹 (XGBoost + LightGBM + CatBoost)
- **1-10~1-12**: FT-Transformer + 하이브리드 서빙

### Phase 2: MLOps
- **2-1~2-2**: MLflow Tracking + Registry (Champion/Challenger)
- **2-3**: Evidently 드리프트 모니터링
- **2-4**: 비용 함수 + pytest 테스트

## 면접 포인트

### 1. 비용 기반 Threshold 최적화
- FN(놓친 사기): 100만원 손실
- FP(오탐): 5만원 비용
- 비용 비율 FN:FP = 20:1 → 최적 Threshold 0.18

### 2. 4단계 위험도 분류
```
[approve] < 0.18: 자동 승인
[verify]  0.18~0.50: 추가 인증
[hold]    0.50~0.80: 수동 검토
[block]   > 0.80: 자동 차단
```

### 3. SHAP 설명 (XAI)
- TreeExplainer로 실시간 Top 5 피처 설명
- "이 거래가 사기로 판단된 이유" 제공

### 4. 스태킹 앙상블 (1-9)
- XGBoost + LightGBM + CatBoost → LogisticRegression 메타 모델
- AUC 0.92, 확률 분포 양극화 (결정 명확성 향상)

### 5. 하이브리드 서빙 (1-12)
- NVIDIA AI Blueprint 2024 패턴 적용
- FT-Transformer 임베딩 → Redis 캐싱 → XGBoost
- 추론 속도 24배 개선 (24ms → 1ms)

### 6. Model Registry
- MLflow Alias 기반 Champion/Challenger 패턴
- A/B 테스트 및 롤백 지원

## 실행 방법

```bash
# 환경 설정
conda create -n fds python=3.11 -y
conda activate fds
pip install -r requirements.txt

# 테스트 실행
pytest tests/ -v
```

## 데이터

- [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle)
- 590,540 거래, 3.5% 사기 비율

## 라이선스

MIT License
