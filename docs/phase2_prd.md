# Phase 2: MLOps + 모니터링

> 모델 운영 체계 구축 - 실험 추적, 버전 관리, 드리프트 감지, CI/CD

---

## 1. Phase 2 개요

### 1.1 왜 Phase 2가 필요한가?

Phase 1에서 모델을 만들고 API로 배포했습니다. 하지만 실제 운영에서는 이런 문제가 발생합니다:

```
[Phase 1만으로는 부족한 이유]

1. "어떤 실험이 가장 좋았지?" → 기록이 없음
2. "모델 성능이 떨어졌는데 롤백하고 싶어" → 이전 버전이 없음
3. "데이터가 변했는데 모델이 여전히 잘 작동하나?" → 모니터링 없음
4. "코드 수정할 때마다 수동 테스트?" → 자동화 없음
5. "새 모델이 기존보다 나은지 어떻게 알지?" → A/B 테스트 없음
```

**Phase 2 = 모델을 "만드는" 것에서 "운영하는" 것으로**

### 1.2 Phase 2 목표

| 항목 | 목표 | 측정 방법 |
|------|------|-----------|
| 실험 추적 | 모든 실험 기록 및 비교 | MLflow UI 확인 |
| 모델 관리 | 버전별 롤백 가능 | Champion/Challenger 패턴 |
| 드리프트 감지 | 데이터 변화 자동 탐지 | Evidently 리포트 |
| CI/CD | 코드 변경 시 자동 테스트 | GitHub Actions |
| A/B 테스트 | 모델 성능 비교 체계 | 통계적 유의성 검증 |
| 시스템 모니터링 | API 상태 실시간 확인 | Grafana 대시보드 |

### 1.3 결과물 예시

```
[MLflow 실험 추적]
┌─────────────────────────────────────────────────────────┐
│ Experiment: FDS_XGBoost_Tuning                          │
├─────────────────────────────────────────────────────────┤
│ Run ID    │ AUC    │ Recall │ learning_rate │ Status   │
├───────────┼────────┼────────┼───────────────┼──────────┤
│ run_001   │ 0.8934 │ 0.8521 │ 0.1           │ ✅       │
│ run_002   │ 0.9114 │ 0.9055 │ 0.05          │ ⭐ Best  │
│ run_003   │ 0.8876 │ 0.8234 │ 0.15          │ ✅       │
└─────────────────────────────────────────────────────────┘

[Model Registry - Champion/Challenger]
┌─────────────────────────────────────────────────────────┐
│ Model: fds_xgboost                                      │
├─────────────────────────────────────────────────────────┤
│ Version │ Alias       │ AUC    │ 배포 상태              │
├─────────┼─────────────┼────────┼────────────────────────┤
│ v3      │ @champion   │ 0.9114 │ Production (90%)       │
│ v4      │ @challenger │ 0.9205 │ A/B Test (10%)         │
│ v2      │ -           │ 0.9042 │ Archived               │
└─────────────────────────────────────────────────────────┘

[Evidently 드리프트 리포트]
┌─────────────────────────────────────────────────────────┐
│ Data Drift Report - 2026-01-11                          │
├─────────────────────────────────────────────────────────┤
│ Feature           │ Drift Score │ Status                │
├───────────────────┼─────────────┼───────────────────────┤
│ TransactionAmt    │ 0.12        │ ✅ No Drift           │
│ card_id_freq      │ 0.45        │ ⚠️ Drift Detected     │
│ hour              │ 0.08        │ ✅ No Drift           │
└─────────────────────────────────────────────────────────┘
→ Alert: 재학습 권장

[Grafana 대시보드]
┌─────────────────────────────────────────────────────────┐
│ FDS API Metrics                                         │
├─────────────────────────────────────────────────────────┤
│ Requests/min: 1,234  │ Avg Latency: 45ms                │
│ Error Rate: 0.01%    │ Model: fds_xgboost@champion      │
│                                                         │
│ [▓▓▓▓▓▓▓▓░░] 80% Approve  [▓▓░░░░░░░░] 15% Verify     │
│ [▓░░░░░░░░░] 4% Hold      [░░░░░░░░░░] 1% Block        │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 기술 스택

### 2.1 핵심 기술

| 기술 | 역할 | JD 빈도 | 버전 |
|------|------|---------|------|
| **MLflow** | 실험 추적 + 모델 레지스트리 | 70% | 2.10+ |
| **Evidently** | 데이터/모델 드리프트 감지 | 55% | 0.4+ |
| **Prometheus** | 메트릭 수집 | 40% | - |
| **Grafana** | 시각화 대시보드 | 40% | 10+ |
| **GitHub Actions** | CI/CD 파이프라인 | 50% | - |

### 2.2 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        Phase 2 Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [개발자] ─────→ [GitHub] ─────→ [GitHub Actions]               │
│     │              │                   │                         │
│     │              │           ┌───────┴───────┐                 │
│     │              │           ▼               ▼                 │
│     │              │      [테스트]        [빌드/배포]            │
│     │              │           │               │                 │
│     ▼              ▼           └───────┬───────┘                 │
│  [실험]  ────→ [MLflow Server]         │                        │
│     │              │                   ▼                         │
│     │         ┌────┴────┐      [FastAPI + Model]                │
│     │         ▼         ▼              │                         │
│     │    [Tracking] [Registry]         │                         │
│     │         │         │              │                         │
│     │         │    @champion ──────────┘                         │
│     │         │    @challenger                                   │
│     │         │                                                  │
│     ▼         ▼                                                  │
│  [Evidently] ◄──── 데이터 드리프트 감지                          │
│     │                                                            │
│     └──────→ [Alert: 재학습 필요]                                │
│                                                                  │
│  [Prometheus] ◄──── API 메트릭 수집                              │
│     │                                                            │
│     └──────→ [Grafana 대시보드]                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 MLflow 상세 (2024-2025 현업 표준)

```python
# ⚠️ 중요: Stage(deprecated) vs Alias(권장)

# ❌ 예전 방식 (deprecated) - 사용하지 마세요
client.transition_model_version_stage(
    name="model", version="1", stage="Production"  # deprecated!
)

# ✅ 현재 방식 (2024+) - Alias 사용
client.set_registered_model_alias(
    name="model", alias="champion", version="1"  # 권장!
)

# Champion/Challenger 패턴
# - @champion: 현재 프로덕션에서 사용 중인 모델
# - @challenger: A/B 테스트 중인 신규 모델
```

**왜 Alias인가?**
| Stage (deprecated) | Alias (권장) |
|-------------------|--------------|
| 고정된 3단계만 가능 (Staging/Production/Archived) | 원하는 이름 자유롭게 사용 |
| 단계별 1개 모델만 | 같은 Alias로 여러 용도 관리 가능 |
| 롤백이 복잡 | 단순히 Alias 재지정으로 롤백 |

---

## 3. Day별 구현 내용

### Day 1: MLflow Tracking (실험 추적)

**목표:** 모든 실험을 기록하고 비교할 수 있는 체계 구축

```python
import mlflow

# 실험 생성 및 실행
mlflow.set_experiment("FDS_XGBoost_Tuning")

with mlflow.start_run(run_name="xgb_lr0.05"):
    # 1. 파라미터 기록
    mlflow.log_params({
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 100
    })

    # 2. 모델 학습
    model = train_xgboost(params)

    # 3. 메트릭 기록
    mlflow.log_metrics({
        "auc": 0.9114,
        "recall": 0.9055,
        "precision": 0.0978
    })

    # 4. 모델 저장
    mlflow.sklearn.log_model(model, "model")
```

**결과물:**
- MLflow UI에서 실험 비교
- 최적 하이퍼파라미터 추적
- 재현 가능한 실험 환경

---

### Day 2: Model Registry (모델 버전 관리)

**목표:** Champion/Challenger 패턴으로 모델 버전 관리

```python
from mlflow import MlflowClient

client = MlflowClient()

# 1. 모델 등록
model_uri = f"runs:/{run_id}/model"
mv = client.create_model_version(
    name="fds_xgboost",
    source=model_uri,
    run_id=run_id
)

# 2. Champion 지정 (현재 프로덕션)
client.set_registered_model_alias(
    name="fds_xgboost",
    alias="champion",
    version=mv.version
)

# 3. 새 모델을 Challenger로 (A/B 테스트용)
client.set_registered_model_alias(
    name="fds_xgboost",
    alias="challenger",
    version=new_version
)

# 4. 모델 로드 (Alias 사용)
champion_model = mlflow.pyfunc.load_model("models:/fds_xgboost@champion")
challenger_model = mlflow.pyfunc.load_model("models:/fds_xgboost@challenger")
```

**면접 포인트:**
> "Stage 방식은 deprecated되어서 Alias 패턴을 사용했습니다. Champion/Challenger로 프로덕션 모델과 테스트 모델을 분리하고, 롤백은 Alias만 변경하면 됩니다."

---

### Day 3: Evidently (드리프트 모니터링)

**목표:** 데이터 변화를 자동으로 감지하고 알림

```python
from evidently import Report
from evidently.presets import DataDriftPreset

# 드리프트 리포트 생성
report = Report([
    DataDriftPreset(method="psi")  # PSI: Population Stability Index
])

# 기준 데이터(학습 시점) vs 현재 데이터 비교
result = report.run(
    reference_data=train_df,  # 학습 때 사용한 데이터
    current_data=prod_df       # 현재 서비스 데이터
)

# HTML 리포트 저장
result.save_html("drift_report.html")

# 프로그래밍 방식으로 결과 확인
drift_detected = result.as_dict()["metrics"][0]["result"]["dataset_drift"]
if drift_detected:
    send_alert("데이터 드리프트 감지! 재학습 권장")
```

**드리프트란?**
```
[학습 시점 데이터]          [운영 시점 데이터]
금액: 평균 $100              금액: 평균 $500 ← 드리프트!
시간: 주로 낮                시간: 주로 밤 ← 드리프트!

→ 모델 성능 저하 가능성 → 재학습 필요
```

**언제 드리프트가 발생하나?**
- 계절성 (명절, 블랙프라이데이)
- 사기 패턴 진화
- 사용자 행동 변화
- 시스템/데이터 수집 방식 변경

---

### Day 4: 비용 최적화 + CI/CD

**목표:** 비즈니스 임팩트 계산 + 자동화 파이프라인

**4.1 비용 함수 (Cost-Sensitive)**

```python
# FDS에서 비용 계산
def calculate_cost(y_true, y_pred, threshold=0.18):
    """
    FN (놓친 사기): 평균 피해액 = 100만원
    FP (오탐): 고객 불편 + 검토 비용 = 5만원
    TN (정상 승인): 0원
    TP (사기 차단): -100만원 (절감 효과)
    """
    y_pred_binary = (y_pred >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    cost_fn = fn * 1_000_000   # 놓친 사기: 100만원 손실
    cost_fp = fp * 50_000      # 오탐: 5만원 비용
    savings_tp = tp * 1_000_000 # 차단: 100만원 절감

    net_savings = savings_tp - cost_fn - cost_fp

    return {
        "total_savings": net_savings,
        "annual_projection": net_savings * 365,  # 연간 환산
        "fn_cost": cost_fn,
        "fp_cost": cost_fp
    }

# 예시: 하루 10만 거래 중 사기 0.5%
# → 연간 약 4억원 손실 절감 가능
```

**4.2 GitHub Actions CI/CD**

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml

      - name: Model validation
        run: |
          python scripts/validate_model.py
          # - 모델 파일 존재 확인
          # - 기본 추론 테스트
          # - 성능 기준 통과 확인

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          # Docker 빌드 및 배포
          docker build -t fds-api:${{ github.sha }} .
          # 실제 환경에서는 ECR/GCR 푸시
```

---

### Day 5: A/B 테스트

**목표:** 새 모델이 기존보다 나은지 통계적으로 검증

```python
import numpy as np
from scipy import stats

class ABTest:
    def __init__(self, champion_model, challenger_model, traffic_split=0.1):
        """
        champion: 현재 프로덕션 모델 (90%)
        challenger: 테스트 모델 (10%)
        """
        self.champion = champion_model
        self.challenger = challenger_model
        self.split = traffic_split

        self.champion_results = []
        self.challenger_results = []

    def predict(self, X):
        if np.random.random() < self.split:
            # Challenger에게 10% 트래픽
            pred = self.challenger.predict(X)
            self.challenger_results.append(pred)
            return pred, "challenger"
        else:
            # Champion에게 90% 트래픽
            pred = self.champion.predict(X)
            self.champion_results.append(pred)
            return pred, "champion"

    def evaluate(self, min_samples=1000):
        """통계적 유의성 검증"""
        if len(self.challenger_results) < min_samples:
            return {"status": "collecting", "samples": len(self.challenger_results)}

        # t-test로 성능 차이 검증
        t_stat, p_value = stats.ttest_ind(
            self.challenger_results,
            self.champion_results
        )

        # p < 0.05면 유의미한 차이
        if p_value < 0.05 and np.mean(self.challenger_results) > np.mean(self.champion_results):
            return {"status": "challenger_wins", "p_value": p_value}
        elif p_value < 0.05:
            return {"status": "champion_wins", "p_value": p_value}
        else:
            return {"status": "no_difference", "p_value": p_value}
```

**A/B 테스트 의사결정 흐름:**
```
새 모델 (v4) 개발 완료
        │
        ▼
Challenger로 등록 (@challenger)
        │
        ▼
트래픽 10% 할당 (1주일)
        │
        ▼
┌───────┴───────┐
▼               ▼
성능 비교      성능 비교
(AUC, Recall)  (응답 시간)
        │
        ▼
통계적 유의성 검증 (p < 0.05)
        │
├── Challenger 승리 → Champion으로 승격
├── Champion 유지 → Challenger 폐기
└── 차이 없음 → 추가 데이터 수집 or 다른 기준 검토
```

---

### Day 6: Prometheus + Grafana

**목표:** API 상태를 실시간으로 모니터링

**6.1 FastAPI 메트릭 수집**

```python
from prometheus_client import Counter, Histogram, make_asgi_app
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
import time

app = FastAPI()

# 메트릭 정의
http_requests = Counter(
    'fds_requests_total',
    'Total FDS requests',
    ['method', 'endpoint', 'status', 'risk_level']
)

prediction_latency = Histogram(
    'fds_prediction_seconds',
    'Prediction latency',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

model_predictions = Counter(
    'fds_predictions_total',
    'Predictions by risk level',
    ['model_version', 'risk_level']
)

# 미들웨어
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        http_requests.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            risk_level=getattr(response, 'risk_level', 'unknown')
        ).inc()

        prediction_latency.observe(duration)
        return response

app.add_middleware(MetricsMiddleware)

# /metrics 엔드포인트
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

**6.2 Grafana 대시보드 구성**

```
[FDS Monitoring Dashboard]

┌─────────────────┬─────────────────┬─────────────────┐
│   Requests/min  │  Avg Latency    │   Error Rate    │
│      1,234      │     45ms        │     0.01%       │
└─────────────────┴─────────────────┴─────────────────┘

┌─────────────────────────────────────────────────────┐
│  Risk Level Distribution (Last 1h)                  │
│  [▓▓▓▓▓▓▓▓░░] Approve 80%                          │
│  [▓▓░░░░░░░░] Verify  15%                          │
│  [▓░░░░░░░░░] Hold     4%                          │
│  [░░░░░░░░░░] Block    1%                          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Latency Percentiles                                │
│  p50: 35ms  |  p95: 120ms  |  p99: 250ms           │
│                                                     │
│  ⚠️ Alert: p99 > 200ms (SLA 위반 가능성)            │
└─────────────────────────────────────────────────────┘
```

---

## 4. 기술 선택 이유 (면접용)

### 4.1 MLflow 선택 이유

**Q: "왜 MLflow를 선택했나요?"**

> "오픈소스 중 가장 성숙하고 커뮤니티가 큽니다. Weights & Biases는 SaaS라 비용이 발생하고, Kubeflow는 K8s 종속성이 강합니다. MLflow는 로컬에서 시작해서 점진적으로 확장할 수 있어 선택했습니다."

| 도구 | 장점 | 단점 |
|------|------|------|
| MLflow | 오픈소스, 유연, 점진적 확장 | UI 기본 기능 |
| W&B | 화려한 UI, 협업 기능 | 유료, 벤더 종속 |
| Kubeflow | K8s 네이티브, 파이프라인 | 복잡, K8s 필수 |

### 4.2 Stage vs Alias

**Q: "Model Registry에서 Stage 안 쓰고 Alias 쓴 이유는?"**

> "Stage는 2024년부터 deprecated입니다. MLflow 공식 문서에서도 Alias 사용을 권장합니다. Champion/Challenger 패턴으로 A/B 테스트와 롤백을 더 유연하게 관리할 수 있습니다."

### 4.3 Evidently 선택 이유

**Q: "드리프트 모니터링은 왜 Evidently?"**

> "Python 네이티브라 MLflow/FastAPI와 통합이 쉽습니다. 100+ 메트릭을 지원하고, PSI/KL-divergence 등 다양한 드리프트 탐지 방법을 제공합니다. HTML 리포트와 JSON 출력 모두 지원해서 자동화에도 좋습니다."

### 4.4 비용 최적화 어필

**Q: "비용 최적화는 어떻게 했나요?"**

> "FN(놓친 사기)과 FP(오탐)의 비용이 다릅니다. FN은 평균 100만원 손실, FP는 5만원 비용으로 가정했습니다. 이 비용 함수로 최적 Threshold(0.18)를 도출했고, 연간 약 4억원 손실 절감 효과를 계산했습니다."

---

## 5. 성공 기준

| 항목 | 기준 | 측정 방법 |
|------|------|-----------|
| MLflow 실험 추적 | 모든 실험 기록 | UI에서 비교 가능 |
| Model Registry | Champion/Challenger 구현 | Alias로 모델 로드 |
| 드리프트 감지 | 자동 리포트 생성 | Evidently HTML |
| CI/CD | PR 시 자동 테스트 | GitHub Actions 통과 |
| A/B 테스트 | 통계적 유의성 검증 | p-value < 0.05 |
| 모니터링 대시보드 | 실시간 메트릭 | Grafana 확인 |

---

## 6. 일정

| Day | 주제 | 핵심 기술 | 결과물 |
|-----|------|-----------|--------|
| 1 | 실험 추적 | MLflow Tracking | 실험 비교 UI |
| 2 | 모델 레지스트리 | MLflow Registry + Alias | Champion/Challenger |
| 3 | 드리프트 모니터링 | Evidently | 자동 리포트 |
| 4 | 비용 최적화 + CI/CD | 비용 함수, GitHub Actions | 연간 절감액, 자동 테스트 |
| 5 | A/B 테스트 | 통계 검증 | 모델 비교 체계 |
| 6 | 시스템 모니터링 | Prometheus + Grafana | 대시보드 |

---

## 7. 면접 어필 포인트 정리

| # | 포인트 | 예상 질문 | 답변 키워드 |
|---|--------|-----------|-------------|
| 1 | MLflow 도입 | "실험 관리 어떻게?" | 파라미터/메트릭/모델 버전 추적 |
| 2 | Stage vs Alias | "왜 Alias?" | deprecated, Champion/Challenger |
| 3 | 드리프트 감지 | "성능 저하 어떻게 감지?" | Evidently, PSI, 자동 알림 |
| 4 | 비용 최적화 | "비즈니스 임팩트?" | FN/FP 비용, 연간 4억 절감 |
| 5 | CI/CD | "배포 자동화?" | GitHub Actions, pytest |
| 6 | A/B 테스트 | "새 모델 검증?" | 통계적 유의성, p < 0.05 |
| 7 | 모니터링 | "운영 상태 파악?" | Prometheus, Grafana |

---

## 8. Phase 2 완료 후 포트폴리오 레벨

```
Phase 1 완료: 기본 포트폴리오
├─ 예상 서류 통과율: 30~40%

Phase 2 완료: 좋음 ⭐
├─ Tier 1 (필수)     : 5/5 ✅
├─ Tier 2 (차별화)   : 5/5 ✅
├─ Tier 3 (강력)     : 1/4 ⚠️
└─ 예상 서류 통과율  : 60~70%
```

**Phase 2에서 추가되는 경쟁력:**
- "모델 만드는 것"에서 "모델 운영하는 것"으로 레벨업
- MLOps 경험 = 신입 중 상위 10%
- 현업 표준 도구 경험 (MLflow, Evidently)

---

## 9. 참고 자료

**공식 문서:**
- [MLflow 공식 문서](https://mlflow.org/docs/latest/)
- [Evidently 공식 문서](https://docs.evidentlyai.com/)
- [Prometheus Python Client](https://prometheus.github.io/client_python/)
- [GitHub Actions 문서](https://docs.github.com/en/actions)

**현업 사례:**
- Uber Michelangelo: MLOps 플랫폼
- Netflix Metaflow: ML 워크플로
- Spotify: A/B 테스트 체계
