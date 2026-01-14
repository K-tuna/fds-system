# Phase 2 학습 가이드 - MLOps 기초

> **목표**: Phase 1에서 만든 모델을 "프로덕션에서 운영"하기 위한 MLOps 핵심 개념 습득
> **예상 시간**: 5시간
> **선수 조건**: Phase 1 완료 (모델 학습, API 배포 경험)

---

## 목차
1. [Phase 2 학습 방향](#1-phase-2-학습-방향)
2. [2-S1: MLflow 기초](#2-2-s1-mlflow-기초-2시간)
3. [2-S2: 모니터링/드리프트](#3-2-s2-모니터링드리프트-15시간)
4. [2-S3: CI/CD + A/B 테스트](#4-2-s3-cicd--ab-테스트-15시간)
5. [Phase 2 학습 체크리스트](#5-phase-2-학습-체크리스트)

---

## 1. Phase 2 학습 방향

### 1.1 왜 MLOps가 필요한가?

Phase 1에서 우리는:
- XGBoost 모델 학습 (AUC 0.91)
- FastAPI로 추론 API 배포
- SHAP으로 예측 설명

**하지만 현업에서는 이런 문제가 발생합니다:**

```
[Phase 1만으로 발생하는 문제들]

1. "지난달에 학습한 모델인데, 하이퍼파라미터 뭐였지?"
   → 실험 기록이 없음 (MLflow Tracking 필요)

2. "어제 배포한 모델이 이상해서 롤백하고 싶은데..."
   → 모델 버전 관리 없음 (Model Registry 필요)

3. "배포 후 3개월 지났는데, 성능이 떨어진 것 같아"
   → 모니터링 없음 (Evidently, Prometheus 필요)

4. "새 모델이 진짜 더 좋은 건지 어떻게 알지?"
   → A/B 테스트 없음 (통계적 검증 필요)

5. "매번 수동으로 테스트하고 배포하는데 실수가 나"
   → CI/CD 없음 (GitHub Actions 필요)
```

### 1.2 Phase 1 → Phase 2 연결점

| Phase 1 문제 | Phase 2 해결책 |
|-------------|---------------|
| 하이퍼파라미터 수동 기록 | MLflow Tracking |
| 모델 파일 수동 관리 | Model Registry |
| 성능 저하 모름 | Evidently 드리프트 탐지 |
| 배포 자동화 없음 | GitHub Actions CI/CD |
| 새 모델 효과 불확실 | A/B 테스트 |
| 시스템 상태 모름 | Prometheus + Grafana |

### 1.3 학습-구현 사이클

Phase 2도 Phase 1처럼 **학습 → 즉시 구현** 패턴을 따릅니다:

```
[학습-구현 흐름]

2-S1 MLflow 개념 학습
  ↓
2-1 MLflow Tracking 구현
2-2 Model Registry 구현
  ↓
2-S2 모니터링/드리프트 학습
  ↓
2-3 Evidently 드리프트 구현
  ↓
2-S3 CI/CD + A/B 학습
  ↓
2-4 비용 최적화 + CI/CD 구현
2-5 A/B 테스트 구현
  ↓
2-6 Prometheus + Grafana 구현
```

---

## 2. 2-S1: MLflow 기초 (2시간)

> **학습 후 연결**: 2-1_mlflow_tracking.ipynb, 2-2_mlflow_registry.ipynb

### 2.1 실험 추적이 왜 필요한가?

**현업 시나리오:**
```
PM: "지난달에 Recall 92% 나왔던 모델 다시 써보자"
ML엔지니어: "그때 learning_rate 뭐였더라... 파일 어디 갔지..."
(30분 뒤)
ML엔지니어: "못 찾겠어요. 다시 학습할게요."
PM: "..."
```

**해결책: MLflow Tracking**
- 모든 실험을 자동으로 기록
- 하이퍼파라미터, 메트릭, 아티팩트 추적
- 실험 비교 UI 제공

### 2.2 MLflow 핵심 개념

```
[MLflow 구조]

Experiment (실험 그룹)
├── Run 1 (한 번의 학습)
│   ├── Parameters: {learning_rate: 0.1, max_depth: 6}
│   ├── Metrics: {auc: 0.91, recall: 0.85}
│   └── Artifacts: model.pkl, feature_importance.png
├── Run 2
│   ├── Parameters: {learning_rate: 0.05, max_depth: 8}
│   ├── Metrics: {auc: 0.93, recall: 0.87}
│   └── Artifacts: model.pkl
└── Run 3
    └── ...
```

**핵심 용어:**

| 용어 | 설명 | 예시 |
|------|------|------|
| Experiment | 실험 그룹 | "FDS-XGBoost-Tuning" |
| Run | 한 번의 학습 | Optuna 50회 = 50 Runs |
| Parameter | 입력 설정 | learning_rate, max_depth |
| Metric | 결과 수치 | AUC, Recall, AUPRC |
| Artifact | 결과 파일 | model.pkl, plots |

### 2.3 MLflow 기본 사용법

```python
import mlflow

# 1. 실험 설정
mlflow.set_experiment("FDS-XGBoost")

# 2. 실행 시작
with mlflow.start_run(run_name="baseline"):
    # 3. 파라미터 기록
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 6)

    # (모델 학습...)

    # 4. 메트릭 기록
    mlflow.log_metric("auc", 0.91)
    mlflow.log_metric("recall", 0.85)

    # 5. 아티팩트 저장
    mlflow.log_artifact("model.pkl")
```

### 2.4 Autolog - 자동 기록

XGBoost, LightGBM 등은 **autolog**로 자동 기록 가능:

```python
import mlflow

# 자동 기록 활성화
mlflow.xgboost.autolog()

# 이제 학습하면 자동으로 기록됨
model = xgb.train(params, dtrain)
# → Parameters, Metrics, Model 모두 자동 저장!
```

### 2.5 Model Registry - 모델 저장소

> **중요**: 2024년부터 Stage(Staging/Production) 방식은 **deprecated**!
> 현업에서는 **Alias 방식**을 사용합니다.

**왜 Registry가 필요한가?**
```
[문제 상황]
- models/xgboost_v1.pkl
- models/xgboost_v2.pkl
- models/xgboost_final.pkl
- models/xgboost_final_v2.pkl  ← 어떤 게 프로덕션?
```

**Registry 구조:**
```
[Model Registry]

Model: "fds-xgboost"
├── Version 1 (학습일: 1/1)
├── Version 2 (학습일: 1/15)
├── Version 3 (학습일: 2/1) ← @champion (현재 프로덕션)
└── Version 4 (학습일: 2/10) ← @challenger (테스트 중)
```

### 2.6 Alias vs Stage (면접 필수!)

```
[Stage 방식 - DEPRECATED]
Version 1 → "None"
Version 2 → "Staging"      # 테스트 중
Version 3 → "Production"   # 프로덕션

문제: 한 스테이지에 한 모델만 가능
     → "Production"에 2개 모델 비교 불가
```

```
[Alias 방식 - 현재 권장]
Version 3 → @champion      # 프로덕션
Version 4 → @challenger    # A/B 테스트 중
Version 4 → @experiment    # 같은 버전에 여러 Alias 가능!

장점:
- 유연한 태깅
- A/B 테스트 용이
- 롤백 간단 (@champion을 다른 버전으로)
```

**Alias 사용 코드:**
```python
from mlflow import MlflowClient

client = MlflowClient()

# Alias 설정
client.set_registered_model_alias(
    name="fds-xgboost",
    alias="champion",
    version=3
)

# Alias로 모델 로드
model_uri = "models:/fds-xgboost@champion"
model = mlflow.pyfunc.load_model(model_uri)
```

### 2.7 Champion/Challenger 패턴

FDS에서 새 모델을 안전하게 배포하는 패턴:

```
[Champion/Challenger 흐름]

1. Champion 모델이 100% 트래픽 처리
   └── Version 3 (@champion) → 모든 요청

2. Challenger 모델 등록 (Shadow 모드)
   ├── Version 3 (@champion) → 실제 응답 반환
   └── Version 4 (@challenger) → 예측만 (응답 X, 로깅 O)

3. 성능 비교 (A/B 테스트)
   └── Challenger가 더 좋으면?

4. Champion 승격
   └── Version 4 → @champion으로 변경
   └── Version 3 → @previous (롤백용)
```

### 2.8 체크포인트

학습 완료 후 스스로 답할 수 있어야 합니다:

- [ ] MLflow에서 Experiment, Run, Artifact의 관계는?
- [ ] `mlflow.log_param()`과 `mlflow.log_metric()`의 차이는?
- [ ] Autolog의 장점과 한계는?
- [ ] **Stage 방식이 deprecated된 이유는?** (면접 빈출)
- [ ] Alias로 Champion/Challenger 패턴을 구현하는 방법은?
- [ ] 모델 롤백은 어떻게 하는가?

---

## 3. 2-S2: 모니터링/드리프트 (1.5시간)

> **학습 후 연결**: 2-3_evidently_drift.ipynb, 2-6_prometheus.ipynb

### 3.1 프로덕션 모델이 왜 성능이 떨어지는가?

**현업 시나리오:**
```
[타임라인]

1월: 모델 배포 (AUC 0.91, Recall 90%)
2월: 잘 동작 중
3월: "요즘 이상거래 놓치는 것 같아요" (고객 클레임)
4월: 확인해보니 Recall이 70%로 떨어짐

왜? → 데이터 드리프트!
```

**데이터 드리프트란?**
- 학습 데이터와 실제 데이터의 분포가 달라지는 현상
- 시간이 지나면 **반드시** 발생함

### 3.2 드리프트 종류

```
[드리프트 3종류]

1. Covariate Shift (입력 분포 변화)
   - 학습: 거래금액 평균 10만원
   - 실제: 거래금액 평균 50만원 (고액 거래 증가)
   - 원인: 연말 쇼핑 시즌, 경제 상황 변화

2. Prior Probability Shift (타겟 분포 변화)
   - 학습: 이상거래 비율 1%
   - 실제: 이상거래 비율 3% (사기 급증)
   - 원인: 새로운 사기 수법 등장

3. Concept Drift (입력-타겟 관계 변화)
   - 학습: 해외거래 → 위험
   - 실제: 해외직구 일상화 → 정상
   - 원인: 사용자 행동 패턴 변화
```

### 3.3 드리프트 탐지 지표

**PSI (Population Stability Index)**

가장 널리 쓰이는 드리프트 지표:

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

**PSI 해석 기준 (금융권 표준):**

| PSI 값 | 해석 | 조치 |
|--------|------|------|
| < 0.1 | 안정 | 유지 |
| 0.1 ~ 0.25 | 약한 변화 | 모니터링 강화 |
| **> 0.25** | **심각한 드리프트** | **재학습 필요** |

**예시:**
```python
# 학습 데이터 분포 (Expected)
train_dist = [0.2, 0.3, 0.25, 0.15, 0.1]  # 5개 구간

# 실제 데이터 분포 (Actual)
prod_dist = [0.1, 0.2, 0.3, 0.25, 0.15]   # 분포가 바뀜

# PSI 계산 → 0.15 (약한 변화)
```

### 3.4 Evidently 도구 이해

Evidently는 ML 모니터링을 위한 오픈소스 도구입니다.

**Evidently 0.7+ 주요 변경사항:**

```python
# [구버전] TestSuite 별도 사용
from evidently.test_suite import TestSuite  # deprecated

# [신버전 0.7+] Report에 통합
from evidently import Report
from evidently.presets import DataDriftPreset

report = Report(
    metrics=[DataDriftPreset()],
    include_tests=True  # ← 테스트 결과 포함!
)
```

**주요 Preset:**

| Preset | 용도 |
|--------|------|
| DataDriftPreset | 전체 피처 드리프트 |
| DataQualityPreset | 데이터 품질 (결측치 등) |
| ClassificationPreset | 분류 모델 성능 |
| RegressionPreset | 회귀 모델 성능 |

### 3.5 FDS에서 모니터링할 지표

```
[FDS 모니터링 대시보드]

1. 데이터 드리프트 (Evidently)
   - 거래금액 분포 변화 (PSI)
   - 시간대별 거래 패턴 변화

2. 모델 성능 (일간/주간)
   - Recall (탐지율)
   - AUPRC
   - 비용 지표 (FN×100 + FP×1)

3. 시스템 메트릭 (Prometheus)
   - API 응답 시간 (p50, p95, p99)
   - 요청 처리량 (RPS)
   - 에러율
```

### 3.6 체크포인트

- [ ] 데이터 드리프트의 3가지 종류를 설명할 수 있는가?
- [ ] PSI 0.25가 의미하는 것은?
- [ ] Evidently 0.7+에서 `include_tests=True`의 역할은?
- [ ] FDS에서 드리프트가 발생하면 어떤 조치를 취해야 하는가?
- [ ] 왜 Recall 모니터링이 AUC 모니터링보다 중요한가?

---

## 4. 2-S3: CI/CD + A/B 테스트 (1.5시간)

> **학습 후 연결**: 2-4_cost_cicd.ipynb, 2-5_ab_test.ipynb

### 4.1 왜 수동 배포가 위험한가?

**현업 사고 사례:**
```
[금요일 오후 5시]
개발자: "테스트 다 됐으니까 배포하자"
개발자: (수동으로 서버에 파일 복사)
개발자: "어? 테스트 파일도 같이 올라갔네..."

[금요일 오후 6시]
고객: "결제가 안 돼요!"
개발자: "주말인데... 😱"
```

**CI/CD로 해결:**
- **CI (Continuous Integration)**: 코드 변경 시 자동 테스트
- **CD (Continuous Deployment)**: 테스트 통과 시 자동 배포

### 4.2 GitHub Actions 기초

```yaml
# .github/workflows/ci.yml

name: FDS CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Run tests
      run: pytest tests/ --cov=src
```

**핵심 개념:**

| 용어 | 설명 |
|------|------|
| Workflow | 자동화 프로세스 전체 |
| Job | 독립적인 실행 단위 |
| Step | Job 내 개별 작업 |
| Action | 재사용 가능한 작업 (checkout, setup-python) |
| Runner | 실행 환경 (ubuntu-latest, windows-latest) |

### 4.3 ML 프로젝트의 CI/CD

일반 소프트웨어와 다른 점:

```
[일반 소프트웨어 CI/CD]
코드 변경 → 테스트 → 배포

[ML 프로젝트 CI/CD]
코드 변경 → 테스트 → 배포
데이터 변경 → 재학습 → 검증 → 배포  ← 추가!
모델 드리프트 → 재학습 → 검증 → 배포  ← 추가!
```

**ML CI/CD에서 테스트할 것:**
```python
# 1. 데이터 품질 테스트
def test_data_no_nulls():
    df = load_data()
    assert df.isnull().sum().sum() == 0

# 2. 모델 성능 테스트
def test_model_recall():
    recall = evaluate_model()
    assert recall >= 0.80  # 최소 기준

# 3. 추론 시간 테스트
def test_inference_time():
    start = time.time()
    predict(sample)
    elapsed = time.time() - start
    assert elapsed < 0.1  # 100ms 이내
```

### 4.4 A/B 테스트란?

두 버전을 비교하여 **통계적으로** 더 좋은 것을 선택:

```
[A/B 테스트 구조]

트래픽 100%
    │
    ├── 50% → 모델 A (Champion)
    │         └── 결과 수집
    │
    └── 50% → 모델 B (Challenger)
              └── 결과 수집

4주 후: 통계 검정 → 더 좋은 모델 선택
```

### 4.5 FDS에서 A/B 테스트의 특수성

**일반 A/B 테스트 vs FDS A/B 테스트:**

```
[일반 웹서비스]
- A/B 트래픽 분할: 50% / 50%
- 측정: 클릭률, 전환율
- 리스크: 낮음 (버튼 색상 변경 정도)

[FDS 시스템]
- A/B 트래픽 분할: 100% / Shadow
- 측정: Recall, 비용, 오탐률
- 리스크: 높음 (이상거래 놓치면 금전 손실!)
```

**Shadow 모드:**
```
[FDS A/B 테스트 - Shadow 방식]

거래 요청
    │
    ├── Champion (실제 응답)
    │   └── "차단" or "승인" → 고객에게 반환
    │
    └── Challenger (Shadow)
        └── "차단" or "승인" → 로깅만 (고객에게 X)

비교: 두 모델의 예측을 사후 분석
```

### 4.6 통계적 유의성

**왜 통계 검정이 필요한가?**
```
Champion Recall: 85%
Challenger Recall: 87%

→ Challenger가 더 좋다?
→ 아니면 그냥 운이 좋았던 건가?
→ 통계 검정 필요!
```

**t-검정 기초:**
```python
from scipy import stats

# 두 모델의 일별 Recall 기록
champion_recalls = [0.85, 0.84, 0.86, 0.85, ...]  # 30일
challenger_recalls = [0.87, 0.86, 0.88, 0.87, ...]  # 30일

# t-검정
t_stat, p_value = stats.ttest_ind(champion_recalls, challenger_recalls)

if p_value < 0.05:
    print("통계적으로 유의미한 차이 있음!")
else:
    print("차이가 우연일 수 있음")
```

### 4.7 Champion 승격 조건

FDS에서 Challenger → Champion이 되려면:

```python
승격 조건 = (
    # 1. 성능 향상
    challenger_recall > champion_recall + 0.01  # 1%p 이상 향상

    # 2. 통계적 유의성
    AND p_value < 0.05

    # 3. 비용 개선
    AND challenger_cost < champion_cost

    # 4. 최소 관측 기간
    AND observation_days >= 14  # 2주 이상
)
```

### 4.8 체크포인트

- [ ] CI와 CD의 차이를 설명할 수 있는가?
- [ ] GitHub Actions의 workflow, job, step 관계는?
- [ ] ML 프로젝트에서 테스트해야 할 항목은?
- [ ] FDS에서 Shadow 모드 A/B 테스트를 사용하는 이유는?
- [ ] p-value < 0.05가 의미하는 것은?
- [ ] Champion 승격 조건에서 "비용"이 포함된 이유는?

---

## 5. Phase 2 학습 체크리스트

### 5.1 전체 체크포인트 요약

**2-S1: MLflow (2시간)**
| 항목 | 이해도 |
|------|--------|
| Experiment → Run → Artifact 구조 | ☐ |
| Autolog 사용법 | ☐ |
| **Stage vs Alias 차이 (면접!)** | ☐ |
| Champion/Challenger 패턴 | ☐ |
| 모델 롤백 방법 | ☐ |

**2-S2: 모니터링/드리프트 (1.5시간)**
| 항목 | 이해도 |
|------|--------|
| 3가지 드리프트 종류 | ☐ |
| PSI 해석 (0.25 기준) | ☐ |
| Evidently 0.7+ API 변경 | ☐ |
| FDS 모니터링 지표 | ☐ |

**2-S3: CI/CD + A/B (1.5시간)**
| 항목 | 이해도 |
|------|--------|
| CI/CD 개념 | ☐ |
| GitHub Actions 구조 | ☐ |
| ML 테스트 항목 | ☐ |
| Shadow 모드 A/B | ☐ |
| 통계적 유의성 (p-value) | ☐ |

### 5.2 면접 예상 질문

**MLflow 관련:**
1. "MLflow의 Stage 방식과 Alias 방식의 차이점은?"
   - Stage: deprecated, 스테이지당 1개 모델만
   - Alias: 유연한 태깅, A/B 테스트 용이

2. "모델 버전 관리를 왜 해야 하는가?"
   - 재현성, 롤백, 감사(audit), 협업

3. "Autolog의 한계점은?"
   - 커스텀 메트릭 기록 불가, FDS 비용 지표 등

**드리프트 관련:**
4. "데이터 드리프트가 발생하면 어떻게 대응하는가?"
   - 탐지 → 원인 분석 → 재학습 → 검증 → 배포

5. "PSI 0.3이 나왔다면 어떤 조치를 취하겠는가?"
   - 0.25 초과 → 재학습 필요, 원인 분석 선행

6. "Concept Drift와 Covariate Shift의 차이는?"
   - Covariate: 입력 분포만 변화
   - Concept: 입력-출력 관계 자체가 변화

**CI/CD 관련:**
7. "ML 프로젝트에서 CI/CD가 일반 소프트웨어와 다른 점은?"
   - 데이터 품질 테스트, 모델 성능 테스트, 추론 시간 테스트 추가

8. "FDS에서 A/B 테스트 시 Shadow 모드를 사용하는 이유는?"
   - 리스크 최소화, 실제 거래에 영향 없이 비교 가능

9. "통계적 유의성 없이 A/B 테스트 결과를 신뢰할 수 없는 이유는?"
   - 우연에 의한 차이일 수 있음, 표본 크기 부족 가능

### 5.3 학습 완료 후

각 Study 완료 후 해당 Impl 노트북으로 진행:

| Study | → | Impl |
|-------|---|------|
| 2-S1 MLflow | → | 2-1_mlflow_tracking.ipynb |
| 2-S1 MLflow | → | 2-2_mlflow_registry.ipynb |
| 2-S2 드리프트 | → | 2-3_evidently_drift.ipynb |
| 2-S3 CI/CD | → | 2-4_cost_cicd.ipynb |
| 2-S3 A/B | → | 2-5_ab_test.ipynb |
| 2-S2 모니터링 | → | 2-6_prometheus.ipynb |

---

## 참고 자료

### 공식 문서
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Prometheus Python Client](https://github.com/prometheus/client_python)

### 추천 학습 순서
1. MLflow 공식 튜토리얼 (30분)
2. Evidently Quick Start (20분)
3. GitHub Actions 입문 (30분)
4. 이 문서의 체크포인트 확인
5. 해당 Impl 노트북 진행
