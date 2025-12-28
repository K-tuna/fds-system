# 하이퍼파라미터 튜닝 방법 비교

## 상황 (Situation)
- XGBoost 모델의 최적 하이퍼파라미터를 찾아야 함
- 불균형 데이터: 사기 3.5%
- 3가지 방법 비교: Early Stopping, CV, Optuna

## 문제 (Problem)
- Early Stopping 적용 시 Validation AUC가 높았지만 Test에서 성능 하락
- Val AUC 0.91 → Test AUC 0.88 (Gap +0.03)

## 원인 분석 (Analysis)

### Early Stopping의 문제점
1. **단일 validation set 의존**
   - 80:20 시간순 분할의 validation set 하나에 의존
   - 이 split이 우연히 쉽거나 어려우면 결과 왜곡

2. **불균형 데이터 문제**
   - 전체 사기 비율 3.5%
   - Validation set의 사기 비율이 불안정할 수 있음

3. **Data Drift**
   - 시간순 분할 시 validation(과거)과 test(미래)의 분포 차이
   - Validation 최적 ≠ Test 최적

### CV가 안정적인 이유
1. 여러 fold의 평균 → 단일 split보다 안정적
2. 모든 데이터가 검증에 참여 → 일반화 성능 향상
3. Val-Test gap이 작음 → 실제 성능 예측이 정확

## 해결 (Solution)
3가지 튜닝 방법을 비교 실험:

```python
# 1. Early Stopping
xgb_early = XGBClassifier(n_estimators=1000, early_stopping_rounds=50)
xgb_early.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

# 2. Cross-Validation
for n in search_range:
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')

# 3. Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        ...
    }
    scores = cross_val_score(model, X_train, y_train, cv=3)
    return scores.mean()
```

## 결과 (Result)

| 방법 | 시간 | Test AUC | Val-Test Gap |
|------|------|----------|--------------|
| Early Stopping | 20초 | 0.8821 | +0.031 |
| CV (3-Fold) | 4.4분 | 0.9030 | -0.004 |
| Optuna (50 trials) | 33분 | 0.9132 | -0.005 |

### 핵심 발견
1. **Early Stopping의 함정**: Val AUC 0.91로 좋아 보이지만, Test에서 0.88로 떨어짐
2. **CV의 안정성**: Gap이 거의 없어서 실제 성능 예측 정확
3. **Optuna의 가치**: 33분 투자로 최고 성능, 하지만 시간 대비 효율은 CV가 좋음

## 결론
- **빠른 프로토타이핑**: Early Stopping (20초)
- **안정적 튜닝**: CV (4.4분) ← 추천
- **최고 성능 필요**: Optuna (33분)

## 면접 포인트

> "3가지 튜닝 방법을 비교한 결과, Optuna가 AUC 0.91로 가장 높았지만 33분 걸렸습니다.
> CV는 4분만에 AUC 0.90을 달성했고, Val-Test gap이 작아서 실제 성능 예측이 정확했습니다.
>
> Early Stopping은 Validation에서 0.91로 좋아 보였지만, Test에서 0.88로 떨어졌습니다.
> 이는 단일 validation set에 과적합되었기 때문입니다.
>
> FDS처럼 빠른 재학습이 필요한 경우, **시간 대비 성능**을 고려해 CV를 선택했습니다.
> 다만 최초 배포 시에는 Optuna로 튜닝하고, 이후 재학습은 CV로 하는 전략도 가능합니다."
