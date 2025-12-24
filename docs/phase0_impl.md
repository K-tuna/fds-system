# Phase 0: 기초 학습 - 구현 상세 (AI용)

> 노트북 생성을 위한 상세 스펙

---

## 파일 구조

```
notebooks/
└── phase0/
    ├── 0-0_setup.ipynb
    ├── 0-1_class_typing.ipynb
    ├── 0-2_numpy.ipynb
    ├── 0-3_pandas.ipynb
    └── 0-4_matplotlib.ipynb
```

> Note: 0-5, 0-6은 `notebooks/phase1/study/`로 이동됨

---

## 노트북 공통 구조

```
1. 제목 + 학습 목표 (마크다운)
2. 패키지 임포트 (코드 셀)
3. 개념 설명 (마크다운)
4. 예제 코드 (코드 셀)
5. 실습 (코드 셀, TODO 포함)
6. 체크포인트 (assert 검증)
7. 다음 주제로 반복...
8. 최종 체크포인트
```

---

## 0-0: 환경 세팅

### 목적
Phase 0~1 진행을 위한 conda 환경 구성

### 세부 설명 리스트

**1. conda 환경이란**
- 프로젝트별 독립된 Python 환경
- 패키지 충돌 방지
- 환경 생성/활성화 방법

**2. fds 환경 생성**
- Python 3.11 사용
- conda create 명령어
- conda activate 명령어

**3. 패키지 설치**
- Phase 0 기본: numpy, pandas, matplotlib, scikit-learn
- Phase 0 후반: xgboost, optuna, shap
- ipykernel: VSCode 연동

**4. VSCode 커널 연결**
- ipykernel 설치
- 커널 등록
- VSCode에서 선택

### 노트북 구조

```
[마크다운] # 0-0: 환경 세팅
[마크다운] ## 1. conda 환경 생성
[마크다운] 터미널에서 실행할 명령어 안내
[코드] # 터미널 명령어 (복사용, 실행 X)
[마크다운] ## 2. 패키지 설치
[코드] Phase 0 기본 패키지
[코드] Phase 0 후반 패키지
[마크다운] ## 3. VSCode 커널 연결
[코드] ipykernel 설치 및 등록
[마크다운] ## 4. 설치 확인
[코드] 임포트 테스트
[코드] 체크포인트
```

### 상세 코드

```python
# [마크다운]
"""
# 0-0: 환경 세팅

Phase 0~1 진행을 위한 환경을 구성합니다.

## 1. conda 환경 생성

**터미널**(VSCode 하단 터미널 또는 Anaconda Prompt)에서 아래 명령어를 실행하세요.
이 셀은 실행하지 마세요!
"""
```

```python
# 터미널에서 실행 (이 셀은 실행 X)
# 아래 명령어를 복사해서 터미널에 붙여넣기

# conda create -n fds python=3.11 -y
# conda activate fds
```

```python
# Phase 0 기본 패키지 설치
!pip install numpy pandas matplotlib scikit-learn -q
print("✅ 기본 패키지 설치 완료")
```

```python
# Phase 0 후반 패키지 설치
!pip install xgboost optuna shap -q
print("✅ 후반 패키지 설치 완료")
```

```python
# VSCode 커널 연결
!pip install ipykernel -q
!python -m ipykernel install --user --name=fds --display-name="Python (fds)"
print("✅ 커널 등록 완료")
print("→ VSCode에서 커널 선택: Python (fds)")
```

```python
# 설치 확인
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

print("✅ 기본 패키지 임포트 성공")
print(f"  numpy: {np.__version__}")
print(f"  pandas: {pd.__version__}")
```

```python
# 체크포인트
import importlib

packages = ['numpy', 'pandas', 'matplotlib', 'sklearn', 'xgboost', 'optuna', 'shap']
for pkg in packages:
    spec = importlib.util.find_spec(pkg)
    assert spec is not None, f"{pkg} 설치 안됨"
    print(f"  ✓ {pkg}")

print()
print("="*50)
print("🎉 환경 세팅 완료!")
print("="*50)
print("다음: 0-1 클래스 + 타입 힌트")
```

---

## 0-1: 클래스 + 타입 힌트 (1시간)

### 필요 패키지
```
없음 (기본 Python)
```

### 세부 설명 리스트

**1. 클래스 기초**
- 클래스란 무엇인가 (데이터 + 함수 묶음)
- 클래스 vs 함수 비교
- 객체(인스턴스) 생성

**2. __init__과 self**
- `__init__`: 생성자, 초기화 담당
- `self`: 객체 자신을 가리킴
- 속성(attribute) 정의

**3. 메서드**
- 메서드란 (클래스 안의 함수)
- self를 첫 인자로 받는 이유
- 메서드 호출 방법

**4. sklearn이 클래스인 이유**
- 모델은 상태(학습된 파라미터)를 저장해야 함
- fit/predict 패턴
- 직접 만들어보기

**5. 타입 힌트 기초**
- 왜 쓰는가 (가독성, 에러 방지)
- 기본 타입: str, int, float, bool
- 함수 파라미터/리턴 타입

**6. typing 모듈**
- `List[str]`: 문자열 리스트
- `Dict[str, int]`: 딕셔너리
- `Optional[int]`: None일 수도 있음
- `TypedDict`: 딕셔너리 구조 정의

### 실습 목록
- 실습 1: 간단한 클래스 만들기 (Dog)
- 실습 2: __init__으로 초기화
- 실습 3: fit/predict 패턴 구현 (MaxModel)
- 실습 4: 함수에 타입 힌트 추가
- 실습 5: TypedDict 정의 (FDSResult)

### 노트북 구조

```
[마크다운] # 0-1: 클래스 + 타입 힌트
[마크다운] ## 학습 목표
[마크다운] ## 1. 클래스 기초
[마크다운] ### 📚 클래스란?
[코드] 클래스 vs 함수 비교
[마크다운] ### 💻 실습 1: Dog 클래스
[코드] 실습 1 (TODO)
[코드] 체크포인트 1
[마크다운] ## 2. __init__과 self
[마크다운] ### 📚 __init__이란?
[코드] __init__ 예제
[마크다운] ## 3. sklearn 패턴
[마크다운] ### 📚 왜 클래스인가?
[코드] SimpleModel 예제
[마크다운] ### 💻 실습 2: MaxModel
[코드] 실습 2 (TODO)
[코드] 체크포인트 2
[마크다운] ## 4. 타입 힌트
[마크다운] ### 📚 기본 타입 힌트
[코드] 타입 힌트 예제
[마크다운] ### 📚 typing 모듈
[코드] List, Dict, Optional 예제
[마크다운] ### 💻 실습 3: 타입 힌트 추가
[코드] 실습 3 (TODO)
[코드] 체크포인트 3
[마크다운] ## 5. TypedDict
[마크다운] ### 📚 TypedDict란?
[코드] TypedDict 예제
[마크다운] ### 💻 실습 4: FDSResult 정의
[코드] 실습 4 (TODO)
[코드] 체크포인트 4
[마크다운] ## ✅ 최종 체크포인트
[코드] 최종 체크포인트
```

### 상세 코드

#### 1. 클래스 기초

```python
# 📚 클래스란?
# 관련된 데이터와 함수를 묶어놓은 것

# 함수로만 하면:
def get_area_func(width, height):
    return width * height

# 매번 width, height를 전달해야 함
print(get_area_func(10, 5))  # 50
```

```python
# 클래스로 하면:
class Rectangle:
    def __init__(self, width, height):
        self.width = width    # 데이터 저장
        self.height = height
    
    def get_area(self):
        return self.width * self.height

# 한번 만들면 데이터가 저장되어 있음
rect = Rectangle(10, 5)
print(rect.get_area())  # 50
print(rect.width)       # 10
```

```python
# 💻 실습 1: Dog 클래스 만들기
# TODO: name, age 속성을 가지는 Dog 클래스
# TODO: bark() 메서드는 f"{self.name}가 짖습니다!" 출력

class Dog:
    def __init__(self, name, age):
        # TODO: 속성 저장
        pass
    
    def bark(self):
        # TODO: 짖기
        pass

# 테스트
my_dog = Dog("멍멍이", 3)
my_dog.bark()
```

```python
# 체크포인트 1
assert hasattr(my_dog, 'name'), "name 속성이 없습니다"
assert hasattr(my_dog, 'age'), "age 속성이 없습니다"
assert my_dog.name == "멍멍이", "name이 올바르지 않습니다"
assert my_dog.age == 3, "age가 올바르지 않습니다"

print("✅ 체크포인트 1 통과!")
```

#### 2. sklearn 패턴

```python
# 📚 sklearn이 왜 클래스인가?
# 모델은 학습된 파라미터를 저장해야 함

class SimpleModel:
    def __init__(self):
        self.is_fitted = False
        self.mean_value = None
    
    def fit(self, X, y):
        """학습: 평균값 저장"""
        self.mean_value = sum(y) / len(y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """예측: 항상 평균값 반환"""
        if not self.is_fitted:
            raise ValueError("먼저 fit()을 호출하세요")
        return [self.mean_value] * len(X)

model = SimpleModel()
model.fit([1,2,3], [10, 20, 30])
print(model.predict([4, 5]))  # [20.0, 20.0]
```

```python
# 💻 실습 2: MaxModel 만들기
# TODO: fit()에서 y의 최대값을 저장
# TODO: predict()에서 항상 최대값 반환

class MaxModel:
    def __init__(self):
        self.is_fitted = False
        self.max_value = None
    
    def fit(self, X, y):
        # TODO: y의 최대값을 self.max_value에 저장
        # TODO: self.is_fitted = True
        # TODO: return self
        pass
    
    def predict(self, X):
        # TODO: 최대값을 len(X)개 담은 리스트 반환
        pass

# 테스트
max_model = MaxModel()
max_model.fit([1,2,3], [10, 50, 30])
predictions = max_model.predict([4, 5, 6])
print(predictions)  # [50, 50, 50]
```

```python
# 체크포인트 2
assert max_model.is_fitted == True, "fit() 후 is_fitted가 True여야 함"
assert max_model.max_value == 50, "max_value가 50이어야 함"
assert predictions == [50, 50, 50], "predict 결과가 [50, 50, 50]이어야 함"

print("✅ 체크포인트 2 통과!")
print("→ sklearn의 fit/predict 패턴 이해 완료")
```

#### 3. 타입 힌트

```python
# 📚 타입 힌트
# 실행에 영향 없음. 가독성 + IDE 자동완성용

# 기본 타입
name: str = "홍길동"
age: int = 25
score: float = 95.5
is_fraud: bool = False

# 함수에 타입 힌트
def greet(name: str) -> str:
    return f"안녕, {name}!"

def add(a: int, b: int) -> int:
    return a + b

print(greet("철수"))
print(add(1, 2))
```

```python
# 📚 typing 모듈
from typing import List, Dict, Optional

# List
numbers: List[int] = [1, 2, 3]
names: List[str] = ["홍길동", "김철수"]

# Dict
user: Dict[str, int] = {"age": 25, "score": 100}

# Optional (None일 수도 있음)
def find_user(user_id: int) -> Optional[str]:
    if user_id == 1:
        return "홍길동"
    return None

print(find_user(1))  # "홍길동"
print(find_user(2))  # None
```

```python
# 💻 실습 3: 타입 힌트 추가
from typing import List

# TODO: 아래 함수에 타입 힌트 추가
# calculate_fraud_rate(total: int, fraud_count: int) -> float
# get_top_features(features: List[str], n: int) -> List[str]

def calculate_fraud_rate(total, fraud_count):
    return fraud_count / total

def get_top_features(features, n):
    return features[:n]
```

```python
# 체크포인트 3
import inspect

sig1 = inspect.signature(calculate_fraud_rate)
sig2 = inspect.signature(get_top_features)

# 리턴 타입 확인
assert sig1.return_annotation == float, "calculate_fraud_rate 리턴 타입이 float여야 함"

print("✅ 체크포인트 3 통과!")
```

#### 4. TypedDict

```python
# 📚 TypedDict
from typing import TypedDict, Optional, List

class Transaction(TypedDict):
    transaction_id: int
    amount: float
    is_fraud: Optional[bool]

tx: Transaction = {
    "transaction_id": 1,
    "amount": 50000.0,
    "is_fraud": False
}
print(tx["amount"])
```

```python
# 💻 실습 4: FDSResult TypedDict 정의
from typing import TypedDict, Optional, List

# TODO: FDSResult 정의
# - is_fraud: bool
# - probability: float
# - top_features: List[str]
# - explanation: Optional[str]

class FDSResult(TypedDict):
    # TODO
    pass

# 테스트
result: FDSResult = {
    "is_fraud": True,
    "probability": 0.85,
    "top_features": ["amount", "hour"],
    "explanation": None
}
```

```python
# 체크포인트 4
fields = FDSResult.__annotations__
assert "is_fraud" in fields, "is_fraud 필드 필요"
assert "probability" in fields, "probability 필드 필요"
assert "top_features" in fields, "top_features 필드 필요"
assert "explanation" in fields, "explanation 필드 필요"

print("✅ 체크포인트 4 통과!")
```

#### 최종 체크포인트

```python
print("="*50)
print("🎉 0-1 완료: 클래스 + 타입 힌트")
print("="*50)
print()
print("배운 것:")
print("  - 클래스: 데이터 + 함수 묶음")
print("  - __init__: 초기화, self: 자기 자신")
print("  - sklearn 패턴: fit() → predict()")
print("  - 타입 힌트: List, Dict, Optional, TypedDict")
print()
print("다음: 0-2 Numpy")
```

---

## 0-2 ~ 0-4: 동일 패턴

나머지 섹션(0-2 ~ 0-4)도 위와 동일한 패턴으로 구성:

1. **필요 패키지** - 해당 노트북에서 쓸 패키지
2. **세부 설명 리스트** - 어떤 개념을 설명할지
3. **실습 목록** - 몇 개의 실습이 있는지
4. **노트북 구조** - 마크다운/코드 셀 순서
5. **상세 코드** - 예제, 실습, 체크포인트

---

## 0-2: Numpy (1.5시간)

### 필요 패키지
```python
import numpy as np
```

### 세부 설명 리스트

**1. Numpy란**
- 왜 쓰는가 (빠른 수치 연산)
- 리스트 vs numpy 배열
- ndarray 타입

**2. 배열 생성**
- `np.array()`: 리스트로 생성
- `np.zeros()`, `np.ones()`: 특수 배열
- `np.arange()`: 범위 배열
- `shape`, `dtype` 속성

**3. 인덱싱과 슬라이싱**
- 1차원: `arr[0]`, `arr[-1]`, `arr[1:3]`
- 2차원: `arr[0, 1]`, `arr[:, 0]`

**4. 연산**
- 집계: `sum()`, `mean()`, `std()`, `max()`, `min()`
- 배열 간 연산: `+`, `-`, `*`, `/`
- axis 개념

**5. 조건 필터링**
- 불리언 배열: `arr > 0`
- 조건 필터링: `arr[arr > 0]`
- `np.where()`

**6. 브로드캐스팅**
- 스칼라 연산: `arr * 2`
- 정규화 예시

### 실습 목록
- 실습 1: 배열 생성
- 실습 2: 인덱싱
- 실습 3: 집계 연산
- 실습 4: 조건 필터링 (사기 거래 찾기)
- 실습 5: 정규화

---

## 0-3: Pandas (2.5시간)

### 필요 패키지
```python
import pandas as pd
import numpy as np
```

### 세부 설명 리스트

**1. DataFrame 생성**
- 딕셔너리로 생성
- `pd.read_csv()`
- `head()`, `shape`, `info()`, `dtypes`

**2. 컬럼 선택**
- 단일: `df['col']` → Series
- 다중: `df[['a', 'b']]` → DataFrame

**3. 행 필터링**
- `df[df['col'] > 0]`
- `&`, `|` 조건

**4. groupby**
- `groupby().mean()`, `sum()`, `count()`
- `agg()`

**5. merge**
- LEFT, INNER JOIN
- `on`, `how` 파라미터

**6. 결측치**
- `isna()`, `fillna()`, `dropna()`

### 실습 목록
- 실습 1: DataFrame 생성
- 실습 2: 컬럼 선택
- 실습 3: 행 필터링
- 실습 4: groupby
- 실습 5: merge
- 실습 6: 결측치 처리

---

## 0-4: Matplotlib (1시간)

### 필요 패키지
```python
import matplotlib.pyplot as plt
import numpy as np
```

### 세부 설명 리스트

**1. 기초**
- pyplot, figure, axes
- `plt.show()`

**2. 선 그래프**
- `plt.plot()`
- 색상, 스타일, 범례

**3. 히스토그램**
- `plt.hist()`
- bins 개념

**4. 막대 그래프**
- `plt.bar()`

**5. 꾸미기**
- title, xlabel, ylabel
- xlim, ylim, grid

**6. subplot**
- `plt.subplots()`

### 실습 목록
- 실습 1: 선 그래프
- 실습 2: 히스토그램
- 실습 3: 막대 그래프
- 실습 4: subplot

---

## 전체 요약

| 파일 | 시간 | 필요 패키지 | 실습 수 |
|------|------|------------|--------|
| 0-0_setup | - | - | 1 |
| 0-1_class_typing | 1h | 없음 | 4 |
| 0-2_numpy | 1.5h | numpy | 5 |
| 0-3_pandas | 2.5h | numpy, pandas | 6 |
| 0-4_matplotlib | 1h | numpy, matplotlib | 4 |

**총 약 6시간, 20개 실습**

> Note: 0-5, 0-6은 `notebooks/phase1/study/`로 이동됨 (상세: phase1_impl.md)
