# Phase 2: MLOps + 모니터링 - 구현 상세 (AI용)

> 노트북/코드 생성을 위한 상세 스펙

---

## 구현 방식 (필독)

**각 Day 노트북은 반드시 다음 형식으로 구현해야 함:**

### 노트북 구조

```
[마크다운] 제목 + 학습 목표
[코드] 패키지 임포트
[마크다운] 개념 설명 (왜 필요한가?)
[코드] 예제 코드 (완성본)
[마크다운] 실습 N 설명
[코드] 실습 TODO (빈칸)
[코드] 실습 정답 (채운 버전)
[코드] 체크포인트 (assert)
... (반복)
[마크다운] 최종 요약 + 면접 포인트
```

### 포함 요소

| 요소 | 설명 |
|------|------|
| 개념 설명 | 마크다운으로 "왜 필요한가?" 먼저 |
| 예제 코드 | 완성된 예제 (학습용) |
| 실습 TODO | 빈칸/TODO 포함된 실습 코드 |
| 실습 정답 | TODO 채운 정답 코드 |
| 체크포인트 | assert로 검증 |
| 면접 포인트 | 해당 Day 관련 면접 Q&A |

---

## 파일 구조

```
fds-system/
├── notebooks/
│   └── phase2/
│       ├── 2-1_mlflow_tracking.ipynb    # ✅ 완료
│       ├── 2-2_mlflow_registry.ipynb    # ✅ 완료
│       ├── 2-3_evidently_drift.ipynb    # ✅ 완료
│       ├── 2-4_cost_cicd.ipynb          # ⏳ 미완료
│       ├── 2-5_ab_testing.ipynb         # ⏳ 미완료
│       └── 2-6_prometheus_grafana.ipynb # ⏳ 미완료
│
├── src/
│   └── monitoring/                       # Phase 2 모듈
│       ├── __init__.py
│       ├── drift.py                      # 드리프트 모니터링
│       ├── metrics.py                    # FDS 지표 계산
│       └── ab_test.py                    # A/B 테스트
│
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml               # CI/CD 파이프라인
│
├── tests/
│   ├── test_model.py                     # 모델 테스트
│   └── test_api.py                       # API 테스트
│
└── reports/                              # Evidently 리포트
    └── drift_report_YYYYMMDD.html
```

---

## 2-1: MLflow Tracking (Day 1) ✅ 완료

### 필요 패키지
```python
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import optuna
```

### FDS 현업 지표 함수

```python
def calculate_fds_metrics(y_true, y_prob, threshold=0.18, fn_cost=100, fp_cost=1):
    """
    FDS 현업 표준 지표 계산

    Args:
        threshold: 1-3에서 최적화한 값 (기본 0.18)
        fn_cost: FN 1건당 비용 (사기 놓침 = 큰 손실)
        fp_cost: FP 1건당 비용 (오탐 = 검토 인건비)

    Returns:
        dict: Tier별 FDS 지표
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        # Tier 1: 필수 지표
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "recall": recall_score(y_true, y_pred),      # FDS 핵심!
        "precision": precision_score(y_true, y_pred),

        # Tier 2: 비용 기반
        "threshold": threshold,
        "cost": int(fn * fn_cost + fp * fp_cost),

        # Tier 3: 운영 지표
        "f1_score": f1_score(y_true, y_pred),
        "fpr": fp / (fp + tn),
    }
```

### 핵심 코드: Autolog + 수동 로깅

```python
# Autolog 활성화 (모델 저장은 수동으로)
mlflow.xgboost.autolog(log_models=False)

with mlflow.start_run(run_name="xgb_autolog"):
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # FDS 현업 지표 수동 로깅 (Autolog가 못하는 것들)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = calculate_fds_metrics(y_test, y_prob, threshold=0.18)

    mlflow.log_metrics({
        "test_recall": metrics["recall"],      # FDS 핵심!
        "test_auprc": metrics["auprc"],
        "cost": metrics["cost"],
        "threshold": 0.18,
    })

    # 모델 수동 저장 (joblib)
    joblib.dump(model, "model.joblib")
    mlflow.log_artifact("model.joblib", artifact_path="model")
```

### 핵심 코드: Optuna + Nested Runs

```python
def objective(trial):
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
        mlflow.log_params(params)

        model = XGBClassifier(**params).fit(X_tr, y_tr)
        auprc = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

        mlflow.log_metric("auprc", auprc)
        return auprc

with mlflow.start_run(run_name="optuna_50trials"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_auprc", study.best_value)
```

### 실습 목록
- 실습 1: MLflow 연결 확인 (Tracking URI, Experiment)
- 실습 2: XGBoost 수동 로깅 (파라미터, 메트릭, 모델)
- 실습 3: Autolog + 커스텀 메트릭
- 실습 4: Optuna 50 trials (Nested Runs, GPU)
- 실습 5: Best model 저장 및 Feature Importance

### 면접 포인트

**Q: "MLflow를 왜 도입했나요?"**
> "실험 추적이 체계적으로 안 되면 어떤 파라미터가 좋았는지 기억을 못합니다.
> MLflow Tracking으로 모든 실험을 자동 기록하고, UI에서 바로 비교합니다."

**Q: "FDS에서 어떤 지표를 추적하나요?"**
> "AUC, AUPRC 외에도 **Recall(사기 탐지율)**을 핵심으로 추적합니다.
> FN(놓친 사기) 비용이 크기 때문에 비용 함수(FN×100 + FP×1)도 로깅합니다."

---

## 2-2: MLflow Model Registry (Day 2) ✅ 완료

### Stage vs Alias - 중요!

```python
# ❌ 예전 방식 (DEPRECATED) - 사용하지 마세요!
client.transition_model_version_stage(
    name="model", version="1", stage="Production"  # deprecated!
)

# ✅ 현재 방식 (2024+ 권장) - Alias 사용
client.set_registered_model_alias(
    name="model", alias="champion", version="1"  # 권장!
)
```

| 구분 | Stage (deprecated) | Alias (권장) |
|------|-------------------|--------------|
| 상태 | **Deprecated** | **권장** |
| 유연성 | 3단계 고정 | 무제한 alias |
| URI | `models:/name/Production` | `models:/name@champion` |
| 롤백 | stage 재전환 | alias 재지정 |

### 핵심 코드: Champion/Challenger 패턴

```python
from mlflow import MlflowClient

client = MlflowClient()
MODEL_NAME = "fraud_detector"

# 1. 모델 등록
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, MODEL_NAME)

# 2. Champion 설정 (현재 프로덕션)
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=mv.version
)

# 3. Challenger 설정 (A/B 테스트용)
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="challenger",
    version=new_version
)

# 4. Alias로 모델 로드 (프로덕션 코드)
champion_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
```

### 핵심 코드: 태그로 상태 관리

```python
# 모델 버전 태그 설정
client.set_model_version_tag(
    name=MODEL_NAME,
    version=mv.version,
    key="validation_status",
    value="passed"  # pending | passed | failed
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=mv.version,
    key="deployment_env",
    value="prod"  # dev | staging | prod
)
```

### 실습 목록
- 실습 1: MlflowClient 연결
- 실습 2: 2-1의 best_model 등록
- 실습 3: Champion alias 설정 및 로드
- 실습 4: Challenger alias 설정
- 실습 5: 태그 및 설명 추가

### 면접 포인트

**Q: "Stage 방식과 Alias 방식의 차이는?"**
> "Stage는 MLflow 2.9 이전 레거시로 **deprecated**입니다.
> Alias 방식은 `@champion`, `@challenger`처럼 자유롭게 지정할 수 있어서
> A/B 테스트와 롤백이 훨씬 유연합니다."

**Q: "모델 롤백은 어떻게 하나요?"**
> "Alias만 바꾸면 됩니다. 코드 배포 없이 즉시 롤백 가능합니다."
> ```python
> client.set_registered_model_alias("fraud_detector", "champion", "1")
> ```

---

## 2-3: Evidently 드리프트 (Day 3) ✅ 완료

### Evidently 0.7+ API 변경사항

```python
# ❌ 구버전 (0.6.x) - TestSuite 별도 클래스
from evidently.test_suite import TestSuite  # deprecated

# ✅ 신버전 (0.7+) - Report에 테스트 포함
from evidently import Report
from evidently.presets import DataDriftPreset

report = Report(
    metrics=[DataDriftPreset()],
    include_tests=True  # ⭐ 테스트 자동 포함
)
result = report.run(reference_data=train_df, current_data=test_df)
```

### PSI 해석 기준 (금융권 표준)

| PSI 값 | 해석 | 조치 |
|--------|------|------|
| **< 0.1** | 안정 | 유지 |
| **0.1 ~ 0.25** | 약간 변화 | 모니터링 강화 |
| **> 0.25** | 드리프트 | **재학습 검토** |

### 핵심 코드: 드리프트 리포트

```python
from evidently import Report
from evidently.presets import DataDriftPreset

# 리포트 생성 (테스트 포함)
report = Report(
    metrics=[DataDriftPreset()],
    include_tests=True
)

# 실행: reference(학습 데이터) vs current(최근 데이터)
result = report.run(
    reference_data=train_df,
    current_data=current_df
)

# 결과 확인
result_dict = result.dict()
for metric in result_dict.get('metrics', []):
    if 'DriftedColumnsCount' in str(metric.get('metric_name', '')):
        drift_share = metric.get('value', {}).get('share', 0)
        print(f"드리프트 비율: {drift_share:.1%}")

# HTML 저장
result.save_html("reports/drift_report.html")
```

### 핵심 코드: 모니터링 자동화 함수

```python
def run_drift_monitoring(reference_df, current_df, drift_threshold=0.3):
    """
    드리프트 모니터링 (CI/CD 연동용)

    Returns:
        dict: {passed: bool, drift_share: float, report_path: str}
    """
    report = Report(
        metrics=[DataDriftPreset()],
        include_tests=True
    )
    result = report.run(reference_data=reference_df, current_data=current_df)

    # 테스트 결과 확인
    results_dict = result.dict()
    tests_info = results_dict.get('tests', [])
    failed_count = sum(1 for t in tests_info if t.get('status') == 'FAIL')
    passed = (failed_count == 0)

    # 실패 시 리포트 저장
    report_path = None
    if not passed:
        report_path = f"reports/drift_alert_{datetime.now():%Y%m%d}.html"
        result.save_html(report_path)

    return {'passed': passed, 'report_path': report_path}
```

### 실습 목록
- 실습 1: Evidently 설치 및 버전 확인
- 실습 2: Reference/Current 데이터 준비
- 실습 3: 데이터 드리프트 리포트 생성
- 실습 4: 타겟 드리프트 분석
- 실습 5: include_tests로 자동화 테스트
- 실습 6: HTML 리포트 저장

### 면접 포인트

**Q: "PSI를 왜 사용했나요?"**
> "**PSI는 금융권 표준 지표**입니다. 해석 기준이 명확하고(0.25 이상 = 재학습),
> 금융감독원 모델 검증 가이드라인에서도 권장합니다."

**Q: "드리프트가 감지되면 어떻게 대응하나요?"**
> "3단계입니다: 알림(Slack) → 분석(Evidently Report) → 조치(재학습 트리거)"

---

## 2-4: 비용 최적화 + CI/CD (Day 4) ⏳ 미완료

### 필요 패키지
```python
# CI/CD
import pytest
import yaml

# 비용 계산
from sklearn.metrics import confusion_matrix
import numpy as np
```

### 비용 함수 (Cost-Sensitive)

```python
def calculate_business_cost(y_true, y_prob, threshold=0.18):
    """
    FDS 비즈니스 임팩트 계산

    비용 가정:
    - FN (놓친 사기): 평균 피해액 100만원
    - FP (오탐): 고객 불편 + 검토 비용 5만원
    - TP (사기 차단): 100만원 절감
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    cost_fn = fn * 1_000_000    # 놓친 사기 손실
    cost_fp = fp * 50_000       # 오탐 비용
    savings_tp = tp * 1_000_000 # 차단 절감

    net_savings = savings_tp - cost_fn - cost_fp

    return {
        "daily_savings": net_savings,
        "annual_savings": net_savings * 365,
        "fn_cost": cost_fn,
        "fp_cost": cost_fp,
        "optimal_threshold": threshold
    }
```

### Threshold 최적화

```python
def find_optimal_threshold(y_true, y_prob, fn_cost=100, fp_cost=1):
    """
    비용 함수를 최소화하는 threshold 찾기
    """
    best_threshold = 0.5
    best_cost = float('inf')

    for threshold in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        cost = fn * fn_cost + fp * fp_cost

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return best_threshold, best_cost
```

### GitHub Actions CI/CD

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
        run: pytest tests/ --cov=src --cov-report=xml

      - name: Model validation
        run: python scripts/validate_model.py

  drift-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run drift monitoring
        run: python scripts/check_drift.py

      - name: Upload drift report
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: drift-report
          path: reports/drift_*.html
```

### pytest 테스트 구조

```python
# tests/test_model.py
import pytest
import joblib
from pathlib import Path

class TestModel:
    @pytest.fixture
    def model(self):
        return joblib.load(Path("models/xgb_model.joblib"))

    def test_model_exists(self):
        """모델 파일 존재 확인"""
        assert Path("models/xgb_model.joblib").exists()

    def test_model_prediction(self, model):
        """기본 추론 테스트"""
        import numpy as np
        X_dummy = np.random.rand(10, 447)
        proba = model.predict_proba(X_dummy)

        assert proba.shape == (10, 2)
        assert all(0 <= p <= 1 for p in proba[:, 1])

    def test_model_performance(self, model):
        """성능 기준 테스트 (AUC > 0.85)"""
        # 테스트 데이터 로드
        test_df = pd.read_csv("data/processed/test_features.csv")
        X_test = test_df.drop("isFraud", axis=1)
        y_test = test_df["isFraud"]

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        assert auc > 0.85, f"AUC {auc:.4f} < 0.85"
```

### 실습 목록
- 실습 1: 비용 함수 구현
- 실습 2: 최적 Threshold 찾기 (비용 최소화)
- 실습 3: pytest 테스트 작성
- 실습 4: GitHub Actions 워크플로 작성
- 실습 5: 드리프트 체크 스크립트

### 체크포인트
```python
# 체크포인트 1: 비용 함수
cost = calculate_business_cost(y_test, y_prob)
assert "annual_savings" in cost, "비용 함수 구현 실패!"

# 체크포인트 2: Threshold 최적화
threshold, _ = find_optimal_threshold(y_test, y_prob)
assert 0.1 <= threshold <= 0.5, "Threshold 범위 이상!"

# 체크포인트 3: pytest 실행
# !pytest tests/ -v
```

### 면접 포인트

**Q: "비용 최적화는 어떻게 했나요?"**
> "FN(놓친 사기)과 FP(오탐)의 비용이 다릅니다.
> FN은 100만원 손실, FP는 5만원 비용으로 가정하고
> 비용 함수를 최소화하는 threshold(0.18)를 찾았습니다."

**Q: "CI/CD 파이프라인에 뭐가 포함되나요?"**
> "pytest로 모델 테스트, 드리프트 체크, 성능 기준 검증을 자동화했습니다.
> 실패 시 Slack 알림 + 드리프트 리포트가 저장됩니다."

---

## 2-5: A/B 테스트 (Day 5) ⏳ 미완료

### 필요 패키지
```python
from scipy import stats
import numpy as np
from mlflow import MlflowClient
```

### A/B 테스트 클래스

```python
class ABTest:
    """
    Champion vs Challenger A/B 테스트

    워크플로:
    1. Champion (90%) vs Challenger (10%) 트래픽 분배
    2. 충분한 샘플 수집 (min_samples)
    3. 통계적 유의성 검증 (t-test, p < 0.05)
    4. 결과에 따라 승격/폐기
    """

    def __init__(self, champion_model, challenger_model, traffic_split=0.1):
        self.champion = champion_model
        self.challenger = challenger_model
        self.split = traffic_split

        self.champion_results = []
        self.challenger_results = []

    def predict(self, X):
        """트래픽 분배 + 예측"""
        if np.random.random() < self.split:
            pred = self.challenger.predict(X)
            self.challenger_results.append(pred)
            return pred, "challenger"
        else:
            pred = self.champion.predict(X)
            self.champion_results.append(pred)
            return pred, "champion"

    def evaluate(self, y_true_champion, y_true_challenger, min_samples=1000):
        """통계적 유의성 검증"""
        if len(self.challenger_results) < min_samples:
            return {
                "status": "collecting",
                "samples": len(self.challenger_results),
                "required": min_samples
            }

        # 성능 메트릭 계산 (예: AUC)
        champion_auc = roc_auc_score(y_true_champion, self.champion_results)
        challenger_auc = roc_auc_score(y_true_challenger, self.challenger_results)

        # t-test
        t_stat, p_value = stats.ttest_ind(
            self.challenger_results,
            self.champion_results
        )

        # 판정
        if p_value < 0.05 and challenger_auc > champion_auc:
            return {"status": "challenger_wins", "p_value": p_value}
        elif p_value < 0.05:
            return {"status": "champion_wins", "p_value": p_value}
        else:
            return {"status": "no_difference", "p_value": p_value}

    def promote_challenger(self, client, model_name):
        """Challenger를 Champion으로 승격"""
        challenger_mv = client.get_model_version_by_alias(model_name, "challenger")

        # 기존 champion을 previous로
        champion_mv = client.get_model_version_by_alias(model_name, "champion")
        client.set_registered_model_alias(
            model_name, "previous_champion", champion_mv.version
        )

        # challenger를 champion으로
        client.set_registered_model_alias(
            model_name, "champion", challenger_mv.version
        )

        # challenger alias 제거
        client.delete_registered_model_alias(model_name, "challenger")
```

### FastAPI A/B 테스트 엔드포인트

```python
from fastapi import FastAPI
import mlflow

app = FastAPI()

# 모델 로드
champion = mlflow.pyfunc.load_model("models:/fraud_detector@champion")
challenger = mlflow.pyfunc.load_model("models:/fraud_detector@challenger")

ab_test = ABTest(champion, challenger, traffic_split=0.1)

@app.post("/predict")
async def predict(transaction: Transaction):
    X = preprocess(transaction)

    # A/B 테스트 예측
    proba, model_used = ab_test.predict(X)

    # 로깅 (분석용)
    log_prediction(
        transaction_id=transaction.id,
        model=model_used,
        probability=proba
    )

    return {
        "fraud_probability": proba,
        "model_version": model_used
    }

@app.get("/ab-test/status")
async def ab_test_status():
    return ab_test.evaluate(min_samples=1000)
```

### 실습 목록
- 실습 1: ABTest 클래스 구현
- 실습 2: 트래픽 분배 테스트
- 실습 3: 통계적 유의성 검증 (t-test)
- 실습 4: Champion 승격 로직
- 실습 5: FastAPI 엔드포인트 통합

### 체크포인트
```python
# 체크포인트 1: 트래픽 분배
ab_test = ABTest(model1, model2, traffic_split=0.1)
results = [ab_test.predict(X)[1] for _ in range(1000)]
challenger_ratio = results.count("challenger") / len(results)
assert 0.08 <= challenger_ratio <= 0.12, "트래픽 분배 오류!"

# 체크포인트 2: 통계 검증
evaluation = ab_test.evaluate(min_samples=100)
assert "status" in evaluation, "평가 함수 구현 실패!"
```

### 면접 포인트

**Q: "A/B 테스트는 어떻게 구현했나요?"**
> "Champion에 90%, Challenger에 10% 트래픽을 분배합니다.
> 충분한 샘플(1000건+)이 모이면 t-test로 통계적 유의성(p < 0.05)을 검증합니다.
> Challenger가 유의미하게 좋으면 Champion으로 승격합니다."

**Q: "롤백은 어떻게 하나요?"**
> "Alias 패턴이라 간단합니다. `previous_champion` alias를 `champion`으로 바꾸면 즉시 롤백됩니다."

---

## 2-6: Prometheus + Grafana (Day 6) ⏳ 미완료

### 필요 패키지
```python
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
import time
```

### FDS 전용 메트릭 정의

```python
from prometheus_client import Counter, Histogram, Gauge

# === Tier 1: 요청 메트릭 ===
fds_requests_total = Counter(
    'fds_requests_total',
    'Total FDS prediction requests',
    ['endpoint', 'status', 'model_version']
)

fds_latency_seconds = Histogram(
    'fds_latency_seconds',
    'Prediction latency in seconds',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# === Tier 2: 비즈니스 메트릭 ===
fds_predictions_total = Counter(
    'fds_predictions_total',
    'Predictions by risk level',
    ['risk_level', 'model_version']
    # risk_level: approve, verify, hold, block
)

fds_fraud_probability = Histogram(
    'fds_fraud_probability',
    'Distribution of fraud probabilities',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# === Tier 3: 시스템 메트릭 ===
fds_model_load_time = Gauge(
    'fds_model_load_time_seconds',
    'Time to load model',
    ['model_name']
)

fds_active_requests = Gauge(
    'fds_active_requests',
    'Currently processing requests'
)
```

### FastAPI 메트릭 미들웨어

```python
from prometheus_client import Counter, Histogram, make_asgi_app
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
import time

app = FastAPI()

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        fds_active_requests.inc()

        try:
            response = await call_next(request)
            duration = time.time() - start

            # 요청 메트릭
            fds_requests_total.labels(
                endpoint=request.url.path,
                status=response.status_code,
                model_version="champion"
            ).inc()

            # 지연 시간 메트릭
            fds_latency_seconds.labels(
                endpoint=request.url.path
            ).observe(duration)

            return response
        finally:
            fds_active_requests.dec()

app.add_middleware(MetricsMiddleware)

# /metrics 엔드포인트 마운트
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### 예측 엔드포인트에 메트릭 추가

```python
@app.post("/predict")
async def predict(transaction: Transaction):
    # 예측
    probability = model.predict_proba(X)[0, 1]

    # 위험 레벨 결정
    if probability < 0.18:
        risk_level = "approve"
    elif probability < 0.40:
        risk_level = "verify"
    elif probability < 0.65:
        risk_level = "hold"
    else:
        risk_level = "block"

    # 비즈니스 메트릭 기록
    fds_predictions_total.labels(
        risk_level=risk_level,
        model_version="champion"
    ).inc()

    fds_fraud_probability.observe(probability)

    return {
        "fraud_probability": probability,
        "risk_level": risk_level
    }
```

### Docker Compose (Prometheus + Grafana)

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana

  fds-api:
    build: .
    ports:
      - "8000:8000"

volumes:
  grafana-data:
```

### Prometheus 설정

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fds-api'
    static_configs:
      - targets: ['fds-api:8000']
    metrics_path: '/metrics'
```

### Grafana 대시보드 패널

```
[FDS Monitoring Dashboard]

Row 1: Overview
├── Total Requests (Counter)
├── Avg Latency (Histogram)
├── Error Rate (Counter)
└── Active Requests (Gauge)

Row 2: Risk Distribution
├── Approve/Verify/Hold/Block 비율 (Pie Chart)
└── Fraud Probability 분포 (Histogram)

Row 3: Latency
├── p50/p95/p99 Latency (Time Series)
└── Latency Heatmap

Row 4: Alerts
├── p99 > 200ms (SLA 위반)
├── Error Rate > 1%
└── Block Rate > 5% (사기 급증)
```

### 실습 목록
- 실습 1: Prometheus 메트릭 정의 (Counter, Histogram, Gauge)
- 실습 2: FastAPI 미들웨어 구현
- 실습 3: /predict 엔드포인트에 비즈니스 메트릭 추가
- 실습 4: Docker Compose로 Prometheus + Grafana 실행
- 실습 5: Grafana 대시보드 구성

### 체크포인트
```python
# 체크포인트 1: 메트릭 정의
assert fds_requests_total is not None, "Counter 정의 실패!"
assert fds_latency_seconds is not None, "Histogram 정의 실패!"

# 체크포인트 2: /metrics 엔드포인트
import requests
response = requests.get("http://localhost:8000/metrics")
assert response.status_code == 200, "/metrics 접근 실패!"
assert "fds_requests_total" in response.text, "메트릭 노출 실패!"
```

### 면접 포인트

**Q: "모니터링 지표는 어떻게 설계했나요?"**
> "3단계로 구성했습니다:
> - Tier 1 (요청): 총 요청 수, 지연 시간, 에러율
> - Tier 2 (비즈니스): 위험 레벨별 비율, 확률 분포
> - Tier 3 (시스템): 모델 로드 시간, 활성 요청 수"

**Q: "SLA 위반은 어떻게 감지하나요?"**
> "Prometheus Alertmanager로 규칙을 설정합니다.
> p99 지연 > 200ms면 Slack 알림, 에러율 > 1%면 PagerDuty 호출합니다."

---

## 면접 어필 포인트 총정리

| # | 포인트 | 관련 Day | 예상 질문 |
|---|--------|----------|-----------|
| 1 | MLflow 실험 추적 | 2-1 | "실험 관리 어떻게?" |
| 2 | **Stage vs Alias** | 2-2 | "왜 Alias?" |
| 3 | Champion/Challenger | 2-2, 2-5 | "모델 배포 패턴?" |
| 4 | PSI 드리프트 | 2-3 | "드리프트 어떻게 감지?" |
| 5 | 비용 함수 | 2-4 | "비즈니스 임팩트?" |
| 6 | CI/CD 파이프라인 | 2-4 | "자동화?" |
| 7 | A/B 테스트 | 2-5 | "새 모델 검증?" |
| 8 | Prometheus 메트릭 | 2-6 | "모니터링 어떻게?" |

**핵심 메시지:**
> "모델을 만드는 것에서 운영하는 것으로 레벨업했습니다.
> MLflow로 실험/모델 관리, Evidently로 드리프트 감지,
> CI/CD로 자동화, Prometheus로 실시간 모니터링을 구축했습니다."

---

## 환경 설정

### requirements-phase2.txt
```
# MLOps
mlflow>=2.10.0
evidently>=0.7.0

# 모니터링
prometheus-client>=0.20.0

# CI/CD
pytest>=8.0.0
pytest-cov>=4.0.0

# 기존 (Phase 1)
xgboost>=2.0.0
scikit-learn>=1.4.0
fastapi>=0.110.0
uvicorn>=0.25.0
```

### Docker 추가 설정
```dockerfile
# Dockerfile 추가
RUN pip install prometheus-client

EXPOSE 8000
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc
```
