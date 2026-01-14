# FDS 프로젝트 로드맵

> Phase별 점진적 포트폴리오 확장 전략

---

## 전략 개요

```
[목표]
한 번에 다 하지 않고, Phase별로 완성된 포트폴리오를 만들면서 점진적 확장

[장점]
- Phase마다 바로 지원 가능한 포트폴리오
- 취업 기회 놓치지 않음
- 면접에서 "다음에 이거 추가 예정" 어필 가능
- 동기 부여 유지
```

---

## 기술 우선순위 (JD 분석 기반)

### Tier S: 필수 (JD 60%+)

| 순위 | 기술 | JD 빈도 | Phase |
|------|------|---------|-------|
| S-1 | MLflow | 70% | 2 |
| S-2 | Evidently | 55% | 2 |
| S-3 | 비용 기반 최적화 | - | 2 |

### Tier A: 강력 추천 (JD 40~60%)

| 순위 | 기술 | JD 빈도 | Phase |
|------|------|---------|-------|
| A-4 | Kafka | 45% | 3 |
| A-5 | GitHub Actions | 50% | 2 |
| A-6 | Prometheus + Grafana | 40% | 2 |
| A-7 | A/B 테스트 | 40% | 2 |

### Tier B: 있으면 좋음 (JD 25~40%)

| 순위 | 기술 | JD 빈도 | Phase |
|------|------|---------|-------|
| B-8 | Airflow | 35% | 3 |
| B-9 | K8s 기초 | 35% | 4 |
| B-10 | Feast (Feature Store) | 25% | 3 |
| B-11 | BigQuery/Redshift | 40% | 4 |

### Tier C: 선택적 (JD 15~25%)

| 순위 | 기술 | JD 빈도 | Phase |
|------|------|---------|-------|
| C-12 | Spark | 25% | 4 |
| C-13 | ONNX 최적화 | 20% | 3 |
| C-14 | TensorRT | 20% | 3 |
| C-15 | Triton Inference Server | 20% | 3 |
| C-16 | S3/MinIO | 20% | 4 |
| C-17 | ELK Stack | 20% | 5 |

### Tier D: 후순위 (JD 15% 미만)

| 순위 | 기술 | JD 빈도 | Phase |
|------|------|---------|-------|
| D-16 | GNN | 15% | 5 |
| D-17 | Kubeflow | 15% | 5 |
| D-18 | Flink | 10% | 5 |
| D-19 | Snowflake | 15% | 5 |

---

## Phase별 상세

### Phase 1: 모델링 + 배포 (12일)

**목표:** 동작하는 FDS 시스템 + Admin UI + 2025 최신 모델

| Day | 주제 | 핵심 기술 |
|-----|------|-----------|
| 1 | 데이터 + EDA | Pandas, 불균형 데이터 |
| 2 | Feature Engineering | 정형 + 시계열 피처 |
| 3 | XGBoost | Optuna 튜닝, Threshold 최적화 |
| 4 | LSTM | PyTorch, 시퀀스 데이터 |
| 5 | Ensemble | 가중 앙상블, 성능 비교 |
| 6 | SHAP | TreeExplainer, DeepExplainer |
| 7 | FastAPI + Docker | joblib 저장, API 서빙, 컨테이너화 |
| 8 | **React Admin** | Ant Design, 거래 목록/상세 UI |
| 9 | **트리 스태킹** ⭐⭐ | XGBoost + LightGBM + CatBoost, AUC 0.92 |
| 10 | **Transformer** (선택) | TabTransformer, Self-Attention |
| 11 | **PaySim 공정 비교** (선택) | 4모델 비교 + 추론 속도 측정 |
| 12 | **하이브리드 서빙** (선택) | DL 임베딩 + Redis + XGBoost |

**결과물:**
- XGBoost + LSTM 앙상블 실험 (LSTM 효과 미미 → XGBoost 단독 채택)
- **트리 스태킹 (XGB+LGBM+Cat)** → AUC 0.92, AUPRC 0.60, 확률 분포 양극화 ⭐⭐
- SHAP 기반 설명
- FastAPI REST API (스태킹 모델 지원)
- React Admin UI (거래 목록/상세/SHAP)
- Docker 컨테이너
- (선택) Transformer, PaySim 공정 비교, 하이브리드 서빙 실험
- **(완료) 1-12 5개 모델 비교**: 스태킹 AUC 0.9998 최고, 하이브리드 Recall 99.95% 최고

**포트폴리오 레벨:** 기본 ~ 좋음

**면접 어필 포인트:**
- "XGBoost vs LSTM 비교 실험 → LSTM AUC 0.70으로 효과 없음 판단"
- "앙상블 실험: +0.12% → 복잡도 대비 효과 분석 후 XGBoost 단독 채택"
- "SHAP으로 설명 가능한 AI"
- "React Admin으로 SHAP 시각화"
- **"트리 스태킹으로 확률 분포 양극화 → 운영 비용 절감"** ⭐⭐
- **"F1 대신 AUPRC/Recall 사용 이유 설명 가능"** (FDS 특성 이해)
- **"PaySim 시계열에서 4모델 공정 비교 + 추론 속도 벤치마크"** (1-11 선택 시)
- **"NVIDIA 레퍼런스 아키텍처: 배치 DL 임베딩 + Redis + 실시간 XGBoost"** (1-12 선택 시)

---

### Phase 2: MLOps + 모니터링 (6일)

**목표:** 모델 운영 + 모니터링 체계

| Day | 주제 | 핵심 기술 |
|-----|------|-----------|
| 1 | 실험 추적 | MLflow Tracking |
| 2 | 모델 레지스트리 | MLflow Model Registry |
| 3 | 드리프트 모니터링 | Evidently |
| 4 | 비용 최적화 + CI/CD | 비용 함수, GitHub Actions |
| 5 | A/B 테스트 | 모델 비교 실험 체계 |
| 6 | 시스템 모니터링 | Prometheus + Grafana |

**결과물:**
- MLflow 실험 대시보드 ✅
- 모델 버전 관리 체계 (Alias 패턴) ✅
- 데이터/모델 드리프트 감지 (Evidently) ✅
- 비용 함수 + pytest 테스트 ✅
- A/B 테스트 ⏳
- 시스템 메트릭 대시보드 (Prometheus/Grafana) ⏳

**포트폴리오 레벨:** 좋음 ⭐

**면접 어필 포인트:**
- "모델 버전 관리 및 실험 추적"
- "드리프트 모니터링으로 성능 저하 감지"
- "비용 함수로 연간 4억 손실 절감 효과 계산"
- "GitHub Actions로 자동 테스트/배포"

---

### Phase 3: 실시간 + 워크플로 (6일)

**목표:** 실시간 파이프라인 + 자동화

**모델 서빙 변환 흐름:**
```
Phase 1: joblib (개발용)
    ↓
Phase 3: joblib → ONNX → TensorRT → Triton (프로덕션)
```

| Day | 주제 | 핵심 기술 |
|-----|------|-----------|
| 1-2 | 실시간 스트리밍 | Kafka |
| 3 | 워크플로 오케스트레이션 | Airflow |
| 4 | 피처 스토어 | Feast |
| 5 | 모델 최적화 | joblib → ONNX 변환 |
| 6 | DL 최적화 + 서빙 | TensorRT + Triton |

**결과물:**
- Kafka 기반 실시간 추론 파이프라인
- Airflow 재학습 스케줄링
- Feast 피처 관리
- ONNX 추론 최적화
- TensorRT LSTM 최적화
- Triton 멀티모델 서빙

**포트폴리오 레벨:** 매우 좋음 ⭐⭐

**면접 어필 포인트:**
- "Kafka로 실시간 사기 탐지, 지연 50ms 이하"
- "Airflow로 주간 자동 재학습"
- "Feature Store로 피처 일관성 보장"
- "ONNX로 추론 속도 3배 향상"
- "TensorRT로 LSTM 추론 5배 최적화"
- "Triton으로 XGBoost + LSTM 동시 서빙"

---

### Phase 4: 클라우드 + 데이터 인프라 (5일)

**목표:** 클라우드 네이티브 + 확장성

| Day | 주제 | 핵심 기술 |
|-----|------|-----------|
| 1 | 클라우드 DW | BigQuery |
| 2-3 | 오케스트레이션 | Kubernetes 기초 |
| 4 | 오브젝트 스토리지 | S3/MinIO |
| 5 | 대용량 처리 | Spark (선택) |

**결과물:**
- BigQuery 데이터 파이프라인
- K8s 배포 매니페스트
- S3/MinIO 모델 아티팩트 저장소
- (선택) Spark 대용량 전처리

**포트폴리오 레벨:** 풀스택 ⭐⭐⭐

**면접 어필 포인트:**
- "BigQuery로 대용량 데이터 분석"
- "Kubernetes로 오토스케일링"
- "클라우드 네이티브 아키텍처"

---

### Phase 5: 고급 + 차별화 (5일+)

**목표:** 최신 기술 + 시니어급 차별화

| Day | 주제 | 핵심 기술 |
|-----|------|-----------|
| 1-3 | 그래프 신경망 | GNN (PyTorch Geometric) |
| 4-5 | ML 파이프라인 | Kubeflow |
| + | 고급 스트리밍 | Flink |
| + | 로그 관리 | ELK Stack |
| + | (택1) 클라우드 DW | Snowflake |

**결과물:**
- GNN 기반 관계 패턴 탐지
- Kubeflow ML 파이프라인
- (선택) Flink 복잡 이벤트 처리
- (선택) ELK 로그 분석

**포트폴리오 레벨:** 시니어급

**면접 어필 포인트:**
- "GNN으로 사기 조직 네트워크 탐지"
- "3-way 앙상블 (XGBoost + LSTM + GNN)"
- "Kubeflow로 End-to-End ML 파이프라인"

---

## 누적 일정

| Phase | 기간 | 누적 | 포트폴리오 레벨 |
|-------|------|------|-----------------|
| 1 | 12일 | 12일 | 기본~좋음 |
| 2 | 6일 | 18일 | 좋음 ⭐ |
| 3 | 6일 | 24일 | 매우 좋음 ⭐⭐ |
| 4 | 5일 | 29일 | 풀스택 ⭐⭐⭐ |
| 5 | 5일+ | 34일+ | 시니어급 |

> **참고**: Phase 1의 Day 10-12는 선택사항. 필수만(Day 1-9)하면 9일

---

## JD 커버리지 변화

```
Phase 1 완료 후:
├─ Tier 1 (필수)     : 4/5 ✅
├─ Tier 2 (차별화)   : 1/5 ⚠️
├─ Tier 3 (강력)     : 0/4 ❌
└─ 예상 서류 통과율  : 30~40%

Phase 2 완료 후:
├─ Tier 1 (필수)     : 5/5 ✅
├─ Tier 2 (차별화)   : 5/5 ✅
├─ Tier 3 (강력)     : 1/4 ⚠️
└─ 예상 서류 통과율  : 60~70%

Phase 3 완료 후:
├─ Tier 1 (필수)     : 5/5 ✅
├─ Tier 2 (차별화)   : 5/5 ✅
├─ Tier 3 (강력)     : 4/4 ✅
└─ 예상 서류 통과율  : 80~90%
```

---

## 취업 전략

### Phase 1 완료 시점
- 기본적인 포트폴리오로 **지원 시작**
- 면접 피드백 받으면서 Phase 2 진행

### Phase 2 완료 시점
- MLOps 경험 추가로 **경쟁력 상승**
- 금융권 IT 자회사 적극 지원

### Phase 3 완료 시점
- 실시간 처리 경험으로 **차별화 완성**
- 핀테크, 대기업 도전

---

## 참고: MLE vs DS vs DE 포지션별 Phase 우선순위

| Phase | MLE | DS | DE |
|-------|-----|----|----|
| 1 (모델링) | 필수 | 필수 | 선택 |
| 2 (MLOps) | **핵심** | 선택 | 선택 |
| 3 (실시간) | **핵심** | 선택 | **핵심** |
| 4 (인프라) | 좋음 | 선택 | **핵심** |
| 5 (고급) | 차별화 | 차별화 | 선택 |

**MLE 타겟이면:** Phase 1 → 2 → 3 순서 필수
**DS 타겟이면:** Phase 1 집중, 2~3 선택적
**DE 타겟이면:** Phase 1 → 3 → 4 순서

---

## 업데이트 이력

| 날짜 | 변경 내용 |
|------|-----------|
| 2024-12-24 | 초안 작성 |
| 2024-12-31 | Phase 1에 React Admin(Day 8), Fusion(Day 9) 추가 |
| 2025-01-02 | 1-7 FastAPI 완료, API 설계 결함 수정, Recall 90.55% 달성 |
| 2026-01-03 | Phase 1 확장: 1-9~1-12 추가 (트리 스태킹, Transformer, 하이브리드, PaySim) |
| 2026-01-05 | 1-8 React Admin 완료, 1-9 트리 스태킹 완료 (AUC 0.92, AUPRC 0.60), API 스태킹 지원, 1-S10 Transformer 학습 완료, F1 → AUPRC/Recall 지표 변경 |
| 2026-01-05 | 1-10 FT-Transformer 구현 노트북 생성 (PyTorch Tabular 활용) |
| 2026-01-08 | 1-11 PaySim 공정 비교 완료, 1-12 하이브리드 서빙 + 5개 모델 비교 완료 (스태킹 AUC 0.9998 최고, 하이브리드 스태킹 성능 하락 원인 분석) |
| 2026-01-15 | Phase 2 진행: 2-1~2-4 완료 (MLflow, Registry, Evidently, 비용함수+pytest), 2-S3 PSI 기준 수정 (금융감독원→업계 표준) |
