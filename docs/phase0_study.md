# Phase 0: 기초 학습

> 공통 기초 지식 (Python, Numpy, Pandas, Matplotlib)

---

## 개요

### 목적
모든 Phase에서 필요한 공통 기초 학습

### 대상
- Python 기본 문법은 아는 상태
- 클래스, numpy, pandas는 처음

### 총 학습 시간
약 6시간

---

## 학습 목차

| 번호 | 주제 | 시간 | 연관 |
|------|------|------|------|
| 0-1 | 클래스 + 타입 힌트 | 1h | 전체 코드 구조 |
| 0-2 | Numpy | 1.5h | 데이터 연산 |
| 0-3 | Pandas | 2.5h | 데이터 전처리 |
| 0-4 | Matplotlib | 1h | EDA 시각화 |

---

## 0-1: 클래스 + 타입 힌트 (1시간)

### 학습 내용

**클래스 기초**
- `__init__`: 객체 생성 시 초기화
- `self`: 객체 자신을 가리킴
- 메서드: 클래스 안의 함수

**왜 배우나?**
sklearn, XGBoost 등 ML 라이브러리가 전부 클래스 기반:
```python
model = XGBClassifier()  # 객체 생성 (__init__ 호출)
model.fit(X, y)          # 메서드 호출
model.predict(X)         # 메서드 호출
```

**타입 힌트**
- `List`, `Dict`, `Optional`: 타입 명시
- `TypedDict`: 딕셔너리 구조 정의
- 함수 파라미터/리턴 타입 표기

### 체크포인트
- [ ] 클래스 정의하고 객체 생성할 수 있다
- [ ] `__init__`과 `self` 역할을 설명할 수 있다
- [ ] `List[str]`, `Optional[int]` 의미를 안다

---

## 0-2: Numpy (1.5시간)

### 학습 내용

**배열 기초**
- `np.array()`: 배열 생성
- 인덱싱, 슬라이싱
- `shape`, `dtype`

**연산**
- `mean()`, `sum()`, `std()`
- 배열 간 연산 (브로드캐스팅)
- 조건 필터링: `arr[arr > 0]`

**왜 배우나?**
ML에서 데이터는 전부 numpy 배열로 처리됨:
```python
y_true = np.array([0, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1])
accuracy = (y_true == y_pred).mean()  # 0.75
```

### 체크포인트
- [ ] 배열 생성하고 인덱싱할 수 있다
- [ ] `mean()`, `sum()` 등 집계 연산을 쓸 수 있다
- [ ] 조건 필터링 (`arr[arr > 0]`)을 이해한다

---

## 0-3: Pandas (2.5시간)

### 학습 내용

**DataFrame 기초**
- `pd.read_csv()`: CSV 읽기
- 컬럼 선택: `df['col']`, `df[['a', 'b']]`
- 행 필터링: `df[df['col'] > 0]`

**데이터 조작**
- `groupby()`: 그룹별 집계
- `merge()`: 테이블 조인
- `fillna()`, `dropna()`: 결측치 처리

**왜 배우나?**
데이터 전처리의 90%가 pandas:
```python
df = pd.read_csv('transactions.csv')
fraud_rate = df.groupby('card_type')['isFraud'].mean()
```

### 체크포인트
- [ ] CSV 읽고 컬럼 선택할 수 있다
- [ ] 조건으로 행 필터링할 수 있다
- [ ] `groupby()`로 집계할 수 있다
- [ ] 두 테이블을 `merge()`로 합칠 수 있다

---

## 0-4: Matplotlib (1시간)

### 학습 내용

**기본 그래프**
- `plt.hist()`: 히스토그램
- `plt.bar()`: 막대 그래프
- `plt.plot()`: 선 그래프

**꾸미기**
- `plt.title()`, `plt.xlabel()`, `plt.ylabel()`
- `plt.subplot()`: 여러 그래프 배치

**왜 배우나?**
EDA에서 데이터 분포 확인:
```python
plt.hist(df['TransactionAmt'], bins=50)
plt.title('거래 금액 분포')
```

### 체크포인트
- [ ] 히스토그램, 막대 그래프를 그릴 수 있다
- [ ] 제목, 축 레이블을 추가할 수 있다
- [ ] subplot으로 여러 그래프를 배치할 수 있다

---

## 학습 완료 후

### 전체 체크리스트
- [ ] 0-1: 클래스 + 타입 힌트
- [ ] 0-2: Numpy
- [ ] 0-3: Pandas
- [ ] 0-4: Matplotlib

### 다음 단계
Phase 0 완료 → Phase 1 Study (ML 개념) 시작
