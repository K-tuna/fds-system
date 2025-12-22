# 1-2: Feature Engineering + Baseline - 완전한 구현 상세

> 노트북 + src 모듈 생성을 위한 완전한 코드

---

## 메타 정보

- **파일명**: `notebooks/phase1/1-2_feature_baseline.ipynb`
- **예상 시간**: 3시간
- **입력 데이터**: `data/processed/train.csv`, `test.csv`
- **산출물**: 
  - `src/ml/feature_engineering.py`
  - `models/baseline_rf.pkl`

---

## 노트북 셀 구조

### [마크다운] 셀 1: 제목

```markdown
# 1-2: Feature Engineering + Baseline

## 학습 목표
1. FDS용 피처 엔지니어링 패턴 학습
2. 시간/금액/집계/범주형 피처 생성
3. Baseline 모델로 성능 기준선 확립
4. 검증된 코드를 src/로 모듈화

## 핵심 개념
- **Feature Engineering**: Raw 데이터를 모델이 학습하기 좋은 형태로 변환
- **Data Leakage 방지**: 집계 피처는 train으로만 계산 후 test에 적용
- **Baseline**: 복잡한 튜닝 전에 기준 성능 확립
```

---

### [코드] 셀 2: 패키지 임포트

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("✅ 패키지 로드 완료")
```

---

### [코드] 셀 3: 데이터 로드

```python
# 1-1에서 저장한 데이터 로드
train_df = pd.read_csv('../data/processed/train.csv')
test_df = pd.read_csv('../data/processed/test.csv')

print(f"Train: {train_df.shape}")
print(f"Test: {test_df.shape}")
print(f"Train 사기율: {train_df['isFraud'].mean():.2%}")
print(f"Test 사기율: {test_df['isFraud'].mean():.2%}")
```

---

### [마크다운] 셀 4: 시간 피처 설명

```markdown
## 1. 시간 피처 (Time Features)

### 왜 필요한가?
- 사기는 특정 시간대에 많이 발생 (야간, 새벽)
- 요일별 패턴 존재 (주말 vs 평일)

### 생성할 피처
| 피처 | 설명 | 계산 |
|------|------|------|
| hour | 시간 (0-23) | (TransactionDT // 3600) % 24 |
| dayofweek | 요일 (0-6) | (TransactionDT // 86400) % 7 |
| is_weekend | 주말 여부 | dayofweek >= 5 |
| is_night | 야간 여부 | hour in [22,23,0,1,2,3,4,5] |
```

---

### [코드] 셀 5: 시간 피처 예제

```python
# 📚 시간 피처 예제
sample = train_df[['TransactionDT']].head(5).copy()

# TransactionDT는 첫 거래 이후 경과 시간 (초)
sample['hour'] = (sample['TransactionDT'] // 3600) % 24
sample['dayofweek'] = (sample['TransactionDT'] // 86400) % 7

print(sample)
```

---

### [마크다운] 셀 6: 실습 1 설명

```markdown
## 💻 실습 1: 시간 피처 함수 작성

`create_time_features(df)` 함수를 완성하세요.

**요구사항:**
- hour, dayofweek, is_weekend, is_night 피처 생성
- 원본 df를 수정하지 않고 복사본 반환
```

---

### [코드] 셀 7: 실습 1 - TODO

```python
# 💻 실습 1: 시간 피처 함수
def create_time_features(df):
    """
    시간 관련 피처 생성
    
    Args:
        df: TransactionDT 컬럼이 있는 DataFrame
    
    Returns:
        시간 피처가 추가된 DataFrame
    """
    df = df.copy()
    
    # TODO: hour (0-23)
    df['hour'] = None
    
    # TODO: dayofweek (0-6, 0=월요일)
    df['dayofweek'] = None
    
    # TODO: is_weekend (토=5, 일=6)
    df['is_weekend'] = None
    
    # TODO: is_night (22-6시)
    df['is_night'] = None
    
    return df

# 테스트
# test_result = create_time_features(train_df.head())
# print(test_result[['TransactionDT', 'hour', 'dayofweek', 'is_weekend', 'is_night']])
```

---

### [코드] 셀 8: 실습 1 - 정답

```python
# ✅ 실습 1 정답
def create_time_features(df):
    """
    시간 관련 피처 생성
    """
    df = df.copy()
    
    # hour (0-23)
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    
    # dayofweek (0-6)
    df['dayofweek'] = (df['TransactionDT'] // 86400) % 7
    
    # is_weekend
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # is_night (22, 23, 0, 1, 2, 3, 4, 5)
    df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    
    return df

# 적용
train_df = create_time_features(train_df)
test_df = create_time_features(test_df)

print("시간 피처 생성 완료!")
print(train_df[['TransactionDT', 'hour', 'dayofweek', 'is_weekend', 'is_night']].head())
```

---

### [코드] 셀 9: 체크포인트 1

```python
# 체크포인트 1
assert 'hour' in train_df.columns, "❌ hour 피처 없음"
assert 'dayofweek' in train_df.columns, "❌ dayofweek 피처 없음"
assert 'is_weekend' in train_df.columns, "❌ is_weekend 피처 없음"
assert 'is_night' in train_df.columns, "❌ is_night 피처 없음"

assert train_df['hour'].min() >= 0 and train_df['hour'].max() <= 23, "❌ hour 범위 오류"
assert train_df['dayofweek'].min() >= 0 and train_df['dayofweek'].max() <= 6, "❌ dayofweek 범위 오류"
assert set(train_df['is_weekend'].unique()).issubset({0, 1}), "❌ is_weekend는 0,1만"
assert set(train_df['is_night'].unique()).issubset({0, 1}), "❌ is_night는 0,1만"

print("✅ 체크포인트 1 통과!")
```

---

### [마크다운] 셀 10: 금액 피처 설명

```markdown
## 2. 금액 피처 (Amount Features)

### 왜 필요한가?
- 금액 분포가 심하게 skewed (대부분 소액, 일부 고액)
- 로그 변환으로 정규화
- 사기는 "딱 떨어지는 금액"이 많음 (100, 500 등)

### 생성할 피처
| 피처 | 설명 |
|------|------|
| amt_log | log(1 + amount) |
| amt_decimal | 소수점 유무 (0 or 1) |
| amt_bin | 금액 구간 (0: ~50, 1: ~200, 2: ~500, 3: 500+) |
```

---

### [코드] 셀 11: 실습 2 - TODO

```python
# 💻 실습 2: 금액 피처 함수
def create_amount_features(df):
    """
    금액 관련 피처 생성
    """
    df = df.copy()
    
    # TODO: 로그 변환 (np.log1p 사용)
    df['amt_log'] = None
    
    # TODO: 소수점 유무 (amount % 1 != 0이면 소수점 있음)
    df['amt_decimal'] = None
    
    # TODO: 금액 구간화
    # 0: $0-50, 1: $50-200, 2: $200-500, 3: $500+
    # 힌트: pd.cut(df['TransactionAmt'], bins=[0, 50, 200, 500, np.inf], labels=[0,1,2,3])
    df['amt_bin'] = None
    
    return df
```

---

### [코드] 셀 12: 실습 2 - 정답

```python
# ✅ 실습 2 정답
def create_amount_features(df):
    """
    금액 관련 피처 생성
    """
    df = df.copy()
    
    # 로그 변환
    df['amt_log'] = np.log1p(df['TransactionAmt'])
    
    # 소수점 유무
    df['amt_decimal'] = (df['TransactionAmt'] % 1 != 0).astype(int)
    
    # 금액 구간화
    df['amt_bin'] = pd.cut(
        df['TransactionAmt'], 
        bins=[0, 50, 200, 500, np.inf], 
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    return df

# 적용
train_df = create_amount_features(train_df)
test_df = create_amount_features(test_df)

print("금액 피처 생성 완료!")
print(train_df[['TransactionAmt', 'amt_log', 'amt_decimal', 'amt_bin']].head(10))
```

---

### [코드] 셀 13: 체크포인트 2

```python
# 체크포인트 2
assert 'amt_log' in train_df.columns, "❌ amt_log 없음"
assert 'amt_decimal' in train_df.columns, "❌ amt_decimal 없음"
assert 'amt_bin' in train_df.columns, "❌ amt_bin 없음"

assert train_df['amt_log'].min() >= 0, "❌ log1p는 항상 >= 0"
assert set(train_df['amt_decimal'].unique()).issubset({0, 1}), "❌ amt_decimal은 0,1만"

print("✅ 체크포인트 2 통과!")
```

---

### [마크다운] 셀 14: 집계 피처 설명

```markdown
## 3. 집계 피처 (Aggregation Features)

### 왜 필요한가?
- 카드별 거래 패턴이 사기 탐지에 중요
- "이 카드는 평소 얼마나 자주, 얼마씩 쓰는가?"

### ⚠️ Data Leakage 주의!
```
❌ 잘못된 방법: 전체 데이터로 집계 → train/test에 적용
✅ 올바른 방법: train으로만 집계 → test에 merge
```

### 생성할 피처
| 피처 | 설명 |
|------|------|
| card1_count | card1별 거래 횟수 |
| card1_amt_mean | card1별 평균 금액 |
| card1_amt_std | card1별 금액 표준편차 |
```

---

### [코드] 셀 15: 실습 3 - TODO

```python
# 💻 실습 3: 집계 피처 함수
def create_agg_features(train_df, test_df, group_col='card1'):
    """
    집계 피처 생성 (Data Leakage 방지)
    
    Args:
        train_df: 학습 데이터
        test_df: 테스트 데이터
        group_col: 그룹핑 기준 컬럼
    
    Returns:
        train_df, test_df (집계 피처 추가됨)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # TODO: train에서 집계 계산
    # 힌트: train_df.groupby(group_col)['TransactionAmt'].agg(['count', 'mean', 'std'])
    agg_df = None
    
    # TODO: 컬럼명 변경
    # agg_df.columns = [f'{group_col}_count', f'{group_col}_amt_mean', f'{group_col}_amt_std']
    
    # TODO: train에 merge
    # train_df = pd.merge(train_df, agg_df, on=group_col, how='left')
    
    # TODO: test에 merge (train 집계값 사용!)
    # test_df = pd.merge(test_df, agg_df, on=group_col, how='left')
    
    return train_df, test_df
```

---

### [코드] 셀 16: 실습 3 - 정답

```python
# ✅ 실습 3 정답
def create_agg_features(train_df, test_df, group_col='card1'):
    """
    집계 피처 생성 (Data Leakage 방지)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # train에서만 집계 계산
    agg_df = train_df.groupby(group_col)['TransactionAmt'].agg(['count', 'mean', 'std'])
    agg_df.columns = [f'{group_col}_count', f'{group_col}_amt_mean', f'{group_col}_amt_std']
    agg_df = agg_df.reset_index()
    
    # 결측 처리 (std는 거래 1건이면 NaN)
    agg_df[f'{group_col}_amt_std'] = agg_df[f'{group_col}_amt_std'].fillna(0)
    
    # train에 merge
    train_df = pd.merge(train_df, agg_df, on=group_col, how='left')
    
    # test에 merge (train 집계값 사용!)
    test_df = pd.merge(test_df, agg_df, on=group_col, how='left')
    
    # test에만 있는 card1은 NaN → 전체 평균으로 대체
    for col in [f'{group_col}_count', f'{group_col}_amt_mean', f'{group_col}_amt_std']:
        fill_value = train_df[col].mean()
        test_df[col] = test_df[col].fillna(fill_value)
    
    return train_df, test_df

# 적용
train_df, test_df = create_agg_features(train_df, test_df, 'card1')

print("집계 피처 생성 완료!")
print(train_df[['card1', 'card1_count', 'card1_amt_mean', 'card1_amt_std']].head())
```

---

### [코드] 셀 17: 체크포인트 3

```python
# 체크포인트 3
assert 'card1_count' in train_df.columns, "❌ card1_count 없음"
assert 'card1_amt_mean' in train_df.columns, "❌ card1_amt_mean 없음"
assert 'card1_amt_std' in train_df.columns, "❌ card1_amt_std 없음"

# test에도 있는지 확인
assert 'card1_count' in test_df.columns, "❌ test에 card1_count 없음"

# 결측 확인
assert train_df['card1_count'].isna().sum() == 0, "❌ train에 결측 있음"
assert test_df['card1_count'].isna().sum() == 0, "❌ test에 결측 있음"

print("✅ 체크포인트 3 통과!")
print("   → Data Leakage 방지: train 집계값을 test에 적용")
```

---

### [마크다운] 셀 18: 범주형 인코딩 설명

```markdown
## 4. 범주형 인코딩 (Categorical Encoding)

### Label Encoding
- 문자열 → 숫자로 변환
- 트리 모델에서 잘 작동
- NaN은 'unknown'으로 처리

### 인코딩할 컬럼
- ProductCD: 상품 종류
- card4: 카드 종류 (visa, mastercard 등)
- card6: 카드 타입 (debit, credit 등)
- P_emaildomain: 구매자 이메일 도메인
```

---

### [코드] 셀 19: 실습 4 - TODO

```python
# 💻 실습 4: 범주형 인코딩 함수
def encode_categorical(train_df, test_df, cat_cols):
    """
    범주형 컬럼 Label Encoding
    
    Args:
        train_df, test_df: 데이터프레임
        cat_cols: 인코딩할 컬럼 리스트
    
    Returns:
        train_df, test_df, encoders (딕셔너리)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    encoders = {}
    
    for col in cat_cols:
        if col not in train_df.columns:
            continue
            
        le = LabelEncoder()
        
        # TODO: NaN을 'unknown'으로 채우기
        # train_df[col] = train_df[col].fillna('unknown').astype(str)
        # test_df[col] = test_df[col].fillna('unknown').astype(str)
        
        # TODO: train에서 fit
        # le.fit(train_df[col])
        
        # TODO: train, test에 transform
        # train_df[col] = le.transform(train_df[col])
        # test_df[col] = ... (test에 새로운 값 있으면 처리 필요)
        
        # encoders[col] = le
        pass
    
    return train_df, test_df, encoders
```

---

### [코드] 셀 20: 실습 4 - 정답

```python
# ✅ 실습 4 정답
def encode_categorical(train_df, test_df, cat_cols):
    """
    범주형 컬럼 Label Encoding
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    encoders = {}
    
    for col in cat_cols:
        if col not in train_df.columns:
            print(f"⚠️ {col} 컬럼 없음, 스킵")
            continue
        
        le = LabelEncoder()
        
        # NaN → 'unknown'
        train_df[col] = train_df[col].fillna('unknown').astype(str)
        test_df[col] = test_df[col].fillna('unknown').astype(str)
        
        # train + test 합쳐서 fit (test에만 있는 값 처리)
        all_values = pd.concat([train_df[col], test_df[col]]).unique()
        le.fit(all_values)
        
        # transform
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        
        encoders[col] = le
        print(f"✅ {col}: {len(le.classes_)}개 클래스")
    
    return train_df, test_df, encoders

# 적용
cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
train_df, test_df, encoders = encode_categorical(train_df, test_df, cat_cols)

print("\n범주형 인코딩 완료!")
```

---

### [코드] 셀 21: 체크포인트 4

```python
# 체크포인트 4
for col in ['ProductCD', 'card4', 'card6']:
    if col in train_df.columns:
        assert train_df[col].dtype in ['int64', 'int32'], f"❌ {col} 인코딩 안됨"

assert len(encoders) > 0, "❌ 인코더가 없음"

print("✅ 체크포인트 4 통과!")
```

---

### [마크다운] 셀 22: Baseline 모델 설명

```markdown
## 5. Baseline 모델

### 왜 Baseline이 필요한가?
- 복잡한 튜닝 전에 기준 성능 확립
- "최소한 이 정도는 나와야 한다"
- RandomForest: 빠르고 안정적

### 평가 지표
- **AUC-ROC**: 불균형 데이터에서 주요 지표
- (Accuracy는 의미 없음)
```

---

### [코드] 셀 23: 피처 선택

```python
# 피처 선택
feature_cols = [
    # 시간 피처
    'hour', 'dayofweek', 'is_weekend', 'is_night',
    # 금액 피처
    'TransactionAmt', 'amt_log', 'amt_decimal', 'amt_bin',
    # 집계 피처
    'card1_count', 'card1_amt_mean', 'card1_amt_std',
    # 범주형 (인코딩됨)
    'ProductCD', 'card4', 'card6',
    # 기타 수치형
    'card1', 'card2', 'card3', 'card5',
]

# 실제 있는 컬럼만 선택
feature_cols = [col for col in feature_cols if col in train_df.columns]
print(f"선택된 피처: {len(feature_cols)}개")
print(feature_cols)
```

---

### [코드] 셀 24: 학습/테스트 데이터 준비

```python
# X, y 분리
X_train = train_df[feature_cols].copy()
y_train = train_df['isFraud'].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df['isFraud'].copy()

# 결측치 처리 (트리 모델용 -999)
X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train 사기율: {y_train.mean():.2%}")
print(f"y_test 사기율: {y_test.mean():.2%}")
```

---

### [마크다운] 셀 25: 실습 5 설명

```markdown
## 💻 실습 5: Baseline 모델 학습

RandomForestClassifier로 Baseline 성능을 확인합니다.

**요구사항:**
1. RandomForest 모델 생성 (n_estimators=100, random_state=42)
2. 학습 및 예측
3. AUC-ROC 계산
```

---

### [코드] 셀 26: 실습 5 - TODO

```python
# 💻 실습 5: Baseline RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# TODO: 모델 생성
# rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

# TODO: 학습
# rf_model.fit(X_train, y_train)

# TODO: 확률 예측
# y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# TODO: AUC 계산
# auc_score = roc_auc_score(y_test, y_pred_proba)
# print(f"Baseline AUC: {auc_score:.4f}")
```

---

### [코드] 셀 27: 실습 5 - 정답

```python
# ✅ 실습 5 정답
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 모델 생성
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # 불균형 처리
)

# 학습
print("학습 중...")
rf_model.fit(X_train, y_train)
print("학습 완료!")

# 예측
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\n🎯 Baseline AUC: {auc_score:.4f}")
```

---

### [코드] 셀 28: 체크포인트 5

```python
# 체크포인트 5
assert auc_score > 0.7, "❌ AUC가 0.7 이상이어야 함"

print("✅ 체크포인트 5 통과!")
print(f"   Baseline AUC: {auc_score:.4f}")
print("   → 1-3에서 XGBoost로 개선 예정")
```

---

### [코드] 셀 29: 피처 중요도

```python
# 피처 중요도
import matplotlib.pyplot as plt

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# 상위 15개 시각화
fig, ax = plt.subplots(figsize=(10, 6))
top_features = importance_df.head(15)
ax.barh(top_features['feature'], top_features['importance'], color='steelblue')
ax.set_xlabel('Importance')
ax.set_title('Top 15 Feature Importance (RandomForest)')
ax.invert_yaxis()

plt.tight_layout()
plt.show()

print("=== Top 10 피처 ===")
print(importance_df.head(10).to_string(index=False))
```

---

### [코드] 셀 30: 모델 저장

```python
# 모델 저장
os.makedirs('../models', exist_ok=True)

# 모델 저장
joblib.dump(rf_model, '../models/baseline_rf.pkl')

# 메타데이터 저장
metadata = {
    'feature_cols': feature_cols,
    'auc_score': auc_score,
    'model_type': 'RandomForestClassifier'
}
joblib.dump(metadata, '../models/baseline_rf_metadata.pkl')

# 인코더 저장
joblib.dump(encoders, '../models/label_encoders.pkl')

print("✅ 모델 저장 완료!")
print("   - models/baseline_rf.pkl")
print("   - models/baseline_rf_metadata.pkl")
print("   - models/label_encoders.pkl")
```

---

### [마크다운] 셀 31: 모듈화

```markdown
## 6. 코드 모듈화

검증된 피처 엔지니어링 코드를 `src/ml/feature_engineering.py`로 모듈화합니다.

이후 노트북에서:
```python
from src.ml.feature_engineering import create_time_features, create_amount_features
```
```

---

### [코드] 셀 32: 모듈 파일 생성

```python
# src/ml/feature_engineering.py 생성
module_code = '''"""
Feature Engineering for FDS
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def create_time_features(df):
    """
    시간 관련 피처 생성
    
    Args:
        df: TransactionDT 컬럼이 있는 DataFrame
    
    Returns:
        시간 피처가 추가된 DataFrame
    """
    df = df.copy()
    
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['dayofweek'] = (df['TransactionDT'] // 86400) % 7
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    
    return df


def create_amount_features(df):
    """
    금액 관련 피처 생성
    """
    df = df.copy()
    
    df['amt_log'] = np.log1p(df['TransactionAmt'])
    df['amt_decimal'] = (df['TransactionAmt'] % 1 != 0).astype(int)
    df['amt_bin'] = pd.cut(
        df['TransactionAmt'], 
        bins=[0, 50, 200, 500, np.inf], 
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    return df


def create_agg_features(train_df, test_df, group_col='card1'):
    """
    집계 피처 생성 (Data Leakage 방지)
    
    Args:
        train_df: 학습 데이터 (집계 계산용)
        test_df: 테스트 데이터 (집계값 적용)
        group_col: 그룹핑 기준 컬럼
    
    Returns:
        train_df, test_df (집계 피처 추가됨)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # train에서만 집계 계산
    agg_df = train_df.groupby(group_col)['TransactionAmt'].agg(['count', 'mean', 'std'])
    agg_df.columns = [f'{group_col}_count', f'{group_col}_amt_mean', f'{group_col}_amt_std']
    agg_df = agg_df.reset_index()
    agg_df[f'{group_col}_amt_std'] = agg_df[f'{group_col}_amt_std'].fillna(0)
    
    # merge
    train_df = pd.merge(train_df, agg_df, on=group_col, how='left')
    test_df = pd.merge(test_df, agg_df, on=group_col, how='left')
    
    # test 결측 처리
    for col in [f'{group_col}_count', f'{group_col}_amt_mean', f'{group_col}_amt_std']:
        fill_value = train_df[col].mean()
        test_df[col] = test_df[col].fillna(fill_value)
    
    return train_df, test_df


def encode_categorical(train_df, test_df, cat_cols):
    """
    범주형 컬럼 Label Encoding
    
    Args:
        train_df, test_df: 데이터프레임
        cat_cols: 인코딩할 컬럼 리스트
    
    Returns:
        train_df, test_df, encoders (딕셔너리)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    encoders = {}
    
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        
        le = LabelEncoder()
        
        train_df[col] = train_df[col].fillna('unknown').astype(str)
        test_df[col] = test_df[col].fillna('unknown').astype(str)
        
        all_values = pd.concat([train_df[col], test_df[col]]).unique()
        le.fit(all_values)
        
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        
        encoders[col] = le
    
    return train_df, test_df, encoders


def prepare_features(train_df, test_df, cat_cols=None):
    """
    전체 피처 엔지니어링 파이프라인
    
    Args:
        train_df, test_df: 원본 데이터프레임
        cat_cols: 범주형 컬럼 리스트
    
    Returns:
        train_df, test_df, encoders
    """
    if cat_cols is None:
        cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
    
    # 시간 피처
    train_df = create_time_features(train_df)
    test_df = create_time_features(test_df)
    
    # 금액 피처
    train_df = create_amount_features(train_df)
    test_df = create_amount_features(test_df)
    
    # 집계 피처
    train_df, test_df = create_agg_features(train_df, test_df, 'card1')
    
    # 범주형 인코딩
    train_df, test_df, encoders = encode_categorical(train_df, test_df, cat_cols)
    
    return train_df, test_df, encoders
'''

# 파일 저장
os.makedirs('../src/ml', exist_ok=True)

# __init__.py 생성
with open('../src/__init__.py', 'w') as f:
    f.write('')
    
with open('../src/ml/__init__.py', 'w') as f:
    f.write('from .feature_engineering import *\n')

# 모듈 저장
with open('../src/ml/feature_engineering.py', 'w') as f:
    f.write(module_code)

print("✅ 모듈 생성 완료!")
print("   - src/ml/__init__.py")
print("   - src/ml/feature_engineering.py")
```

---

### [코드] 셀 33: 모듈 테스트

```python
# 모듈 임포트 테스트
import sys
sys.path.append('..')

from src.ml.feature_engineering import create_time_features, prepare_features

# 테스트
test_data = pd.DataFrame({
    'TransactionDT': [86400, 172800, 259200],
    'TransactionAmt': [100, 200, 300]
})

result = create_time_features(test_data)
print("모듈 임포트 테스트:")
print(result[['TransactionDT', 'hour', 'dayofweek']])

print("\n✅ 모듈 테스트 성공!")
```

---

### [마크다운] 셀 34: 최종 체크포인트

```markdown
## ✅ 최종 체크포인트
```

---

### [코드] 셀 35: 최종 요약

```python
print("="*60)
print("🎉 1-2 완료: Feature Engineering + Baseline")
print("="*60)
print()
print("📊 생성한 피처:")
print("   - 시간: hour, dayofweek, is_weekend, is_night")
print("   - 금액: amt_log, amt_decimal, amt_bin")
print("   - 집계: card1_count, card1_amt_mean, card1_amt_std")
print("   - 범주형: ProductCD, card4, card6 (Label Encoded)")
print()
print(f"📈 Baseline 성능:")
print(f"   - 모델: RandomForest")
print(f"   - AUC: {auc_score:.4f}")
print()
print("📂 산출물:")
print("   - src/ml/feature_engineering.py (모듈)")
print("   - models/baseline_rf.pkl")
print("   - models/baseline_rf_metadata.pkl")
print("   - models/label_encoders.pkl")
print()
print("🎯 면접 포인트:")
print("   Q: 피처 엔지니어링에서 주의할 점은?")
print("   A: Data Leakage 방지입니다. 집계 피처는 train에서만")
print("      계산하고 test에 적용해야 합니다. 그렇지 않으면")
print("      미래 정보가 누출되어 과적합됩니다.")
print()
print("➡️ 다음: 1-3 모델 고도화 (XGBoost vs LightGBM vs CatBoost)")
```

---

## src/ml/feature_engineering.py (최종본)

위 셀 32에서 생성되는 모듈의 전체 코드입니다.

```python
"""
Feature Engineering for FDS
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def create_time_features(df):
    """시간 관련 피처 생성"""
    df = df.copy()
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['dayofweek'] = (df['TransactionDT'] // 86400) % 7
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    return df


def create_amount_features(df):
    """금액 관련 피처 생성"""
    df = df.copy()
    df['amt_log'] = np.log1p(df['TransactionAmt'])
    df['amt_decimal'] = (df['TransactionAmt'] % 1 != 0).astype(int)
    df['amt_bin'] = pd.cut(
        df['TransactionAmt'], 
        bins=[0, 50, 200, 500, np.inf], 
        labels=[0, 1, 2, 3]
    ).astype(int)
    return df


def create_agg_features(train_df, test_df, group_col='card1'):
    """집계 피처 생성 (Data Leakage 방지)"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    agg_df = train_df.groupby(group_col)['TransactionAmt'].agg(['count', 'mean', 'std'])
    agg_df.columns = [f'{group_col}_count', f'{group_col}_amt_mean', f'{group_col}_amt_std']
    agg_df = agg_df.reset_index()
    agg_df[f'{group_col}_amt_std'] = agg_df[f'{group_col}_amt_std'].fillna(0)
    
    train_df = pd.merge(train_df, agg_df, on=group_col, how='left')
    test_df = pd.merge(test_df, agg_df, on=group_col, how='left')
    
    for col in [f'{group_col}_count', f'{group_col}_amt_mean', f'{group_col}_amt_std']:
        fill_value = train_df[col].mean()
        test_df[col] = test_df[col].fillna(fill_value)
    
    return train_df, test_df


def encode_categorical(train_df, test_df, cat_cols):
    """범주형 컬럼 Label Encoding"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    encoders = {}
    
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        
        le = LabelEncoder()
        train_df[col] = train_df[col].fillna('unknown').astype(str)
        test_df[col] = test_df[col].fillna('unknown').astype(str)
        
        all_values = pd.concat([train_df[col], test_df[col]]).unique()
        le.fit(all_values)
        
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        encoders[col] = le
    
    return train_df, test_df, encoders


def prepare_features(train_df, test_df, cat_cols=None):
    """전체 피처 엔지니어링 파이프라인"""
    if cat_cols is None:
        cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
    
    train_df = create_time_features(train_df)
    test_df = create_time_features(test_df)
    
    train_df = create_amount_features(train_df)
    test_df = create_amount_features(test_df)
    
    train_df, test_df = create_agg_features(train_df, test_df, 'card1')
    train_df, test_df, encoders = encode_categorical(train_df, test_df, cat_cols)
    
    return train_df, test_df, encoders
```

---

## 예상 산출물

1. **노트북**: `notebooks/phase1/1-2_feature_baseline.ipynb`
2. **모듈**: `src/ml/feature_engineering.py`
3. **모델**:
   - `models/baseline_rf.pkl`
   - `models/baseline_rf_metadata.pkl`
   - `models/label_encoders.pkl`
