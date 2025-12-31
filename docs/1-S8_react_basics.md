# 1-S8: React + Tailwind + shadcn/ui 기초

> 1-8 React Admin 구현 전 필요한 프론트엔드 기초

## 학습 목표
1. **React 기본** - 컴포넌트, JSX, Virtual DOM
2. **Props와 State** - 데이터 전달, 상태 관리
3. **Hooks** - useState, useEffect
4. **Tailwind CSS** - 유틸리티 클래스, 조건부 스타일링
5. **shadcn/ui** - DataTable, Badge 컴포넌트
6. **Storybook** - 컴포넌트 문서화 (현업 표준)

## 기술 스택 (2024-2025 트렌드)

| 기술 | 역할 | npm 다운로드/주 |
|------|------|-----------------|
| React 18 | UI 라이브러리 | ~2,500만 |
| Tailwind CSS | CSS 프레임워크 1위 | ~1,490만 |
| shadcn/ui | 컴포넌트 라이브러리 | Radix 기반 |
| Storybook | 컴포넌트 문서화 | 현업 표준 |

---

## 1. React 기본 개념

### 1-1. React란?

**React**: Meta(Facebook)가 만든 UI 라이브러리

| 특징 | 설명 |
|------|------|
| **컴포넌트 기반** | UI를 독립적인 조각으로 분리 |
| **선언적** | "어떻게"가 아닌 "무엇을" 렌더링할지 선언 |
| **Virtual DOM** | 실제 DOM 조작 최소화 → 성능 향상 |

### 1-2. JSX 문법

```jsx
function App() {
  const name = "FDS";
  const isFraud = true;

  return (
    <div>
      <h1>Hello, {name}!</h1>           {/* 변수 사용: {} */}
      <p>상태: {isFraud ? '사기' : '정상'}</p>  {/* 조건부 렌더링 */}
      <button className="btn">클릭</button>   {/* class → className */}
    </div>
  );
}
```

### 1-3. JSX 규칙

| 규칙 | HTML | JSX |
|------|------|-----|
| 클래스 | `class=""` | `className=""` |
| 스타일 | `style="color: red"` | `style={{ color: 'red' }}` |
| 닫는 태그 | `<img>` | `<img />` |
| 변수 | 불가 | `{변수명}` |

---

## 2. 컴포넌트와 Props

### 2-1. 함수형 컴포넌트

```jsx
// 컴포넌트 = UI의 독립적인 조각
function Welcome() {
  return <h1>환영합니다!</h1>;
}

// 화살표 함수
const Welcome = () => <h1>환영합니다!</h1>;

// 사용
<Welcome />
```

### 2-2. Props (속성)

```jsx
// 부모 → 자식으로 데이터 전달
function TransactionCard({ amount, isFraud, transactionId }) {
  return (
    <div className="p-4 border rounded-lg">
      <p>거래ID: {transactionId}</p>
      <p>금액: ₩{amount.toLocaleString()}</p>
      <p>상태: {isFraud ? '사기' : '정상'}</p>
    </div>
  );
}

// 사용
<TransactionCard
  transactionId="TXN_001"
  amount={150000}
  isFraud={false}
/>
```

### Props 특징
- **읽기 전용** (자식에서 수정 불가)
- 객체, 배열, 함수도 전달 가능
- 구조 분해 할당으로 받음: `{ amount, isFraud }`

---

## 3. State와 Hooks

### 3-1. useState Hook

```jsx
import { useState } from 'react';

function Counter() {
  // [현재값, 설정함수] = useState(초기값)
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>카운트: {count}</p>
      <button onClick={() => setCount(count + 1)}>
        증가
      </button>
    </div>
  );
}
```

### Props vs State

| 구분 | Props | State |
|------|-------|-------|
| 출처 | 부모에서 전달 | 컴포넌트 내부 |
| 변경 | 읽기 전용 | `setState`로 변경 |
| 용도 | 외부 데이터 | 내부 상태 |

### 3-2. useEffect Hook

```jsx
import { useState, useEffect } from 'react';

function TransactionList() {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);

  // 컴포넌트 마운트 시 API 호출
  useEffect(() => {
    fetch('http://localhost:8000/transactions')
      .then(res => res.json())
      .then(data => {
        setTransactions(data);
        setLoading(false);
      });
  }, []);  // [] = 마운트 시 한 번만 실행

  if (loading) return <p>로딩 중...</p>;

  return (
    <ul>
      {transactions.map(tx => (
        <li key={tx.id}>{tx.id}: ₩{tx.amount}</li>
      ))}
    </ul>
  );
}
```

### useEffect 의존성 배열

| 형태 | 실행 시점 |
|------|----------|
| `useEffect(() => {}, [])` | 마운트 시 **1번** |
| `useEffect(() => {}, [value])` | `value` 변경 시마다 |
| `useEffect(() => {})` | **매 렌더링**마다 (주의!) |

---

## 4. Tailwind CSS 기초

### 4-1. Tailwind란?

**유틸리티 우선 CSS 프레임워크**

```jsx
// 전통적인 CSS
<div class="card">  // .card { padding: 16px; ... }

// Tailwind CSS
<div className="p-4 bg-white rounded-lg shadow">
```

### 4-2. 자주 쓰는 클래스

#### 레이아웃

| 클래스 | CSS |
|--------|-----|
| `flex` | `display: flex` |
| `grid` | `display: grid` |
| `items-center` | `align-items: center` |
| `justify-between` | `justify-content: space-between` |
| `gap-4` | `gap: 1rem` |

#### 여백 & 패딩

| 클래스 | CSS |
|--------|-----|
| `p-4` | `padding: 1rem` |
| `px-4` | `padding-left/right: 1rem` |
| `py-2` | `padding-top/bottom: 0.5rem` |
| `m-4` | `margin: 1rem` |
| `mt-2` | `margin-top: 0.5rem` |

#### 색상 & 텍스트

| 클래스 | 설명 |
|--------|------|
| `bg-white` | 흰색 배경 |
| `bg-red-500` | 빨간색 배경 (500 = 중간 톤) |
| `text-gray-700` | 회색 텍스트 |
| `text-lg` | 큰 텍스트 |
| `font-bold` | 굵은 텍스트 |

### 4-3. 조건부 스타일링

```jsx
function StatusBadge({ isFraud }) {
  return (
    <span className={`
      px-2 py-1 rounded text-sm font-medium
      ${isFraud
        ? 'bg-red-100 text-red-800'   // 사기: 빨간 배경
        : 'bg-green-100 text-green-800'  // 정상: 초록 배경
      }
    `}>
      {isFraud ? '사기' : '정상'}
    </span>
  );
}
```

---

## 5. shadcn/ui + DataTable

### 5-1. shadcn/ui란?

**복사/붙여넣기 방식의 컴포넌트 라이브러리**

| 특징 | 설명 |
|------|------|
| **복사/붙여넣기** | npm 패키지가 아닌 코드 복사 |
| **커스터마이징** | 코드가 내 프로젝트에 있으므로 자유롭게 수정 |
| **Radix UI 기반** | 접근성 (a11y) 좋음 |
| **Tailwind 기반** | Tailwind 클래스로 스타일링 |

### 5-2. 기본 컴포넌트

```jsx
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

// Button
<Button>기본</Button>
<Button variant="destructive">삭제</Button>
<Button variant="outline">테두리</Button>

// Badge
<Badge>기본</Badge>
<Badge variant="destructive">사기</Badge>
```

### 5-3. DataTable (TanStack Table)

```tsx
import { ColumnDef } from "@tanstack/react-table";
import { Badge } from "@/components/ui/badge";

type Transaction = {
  transaction_id: string;
  fraud_probability: number;
  is_fraud: boolean;
};

const columns: ColumnDef<Transaction>[] = [
  {
    accessorKey: "transaction_id",
    header: "거래ID",
  },
  {
    accessorKey: "fraud_probability",
    header: "사기확률",
    cell: ({ row }) => {
      const prob = row.getValue("fraud_probability") as number;
      return `${(prob * 100).toFixed(1)}%`;
    },
  },
  {
    accessorKey: "is_fraud",
    header: "판정",
    cell: ({ row }) => {
      const isFraud = row.getValue("is_fraud") as boolean;
      return (
        <Badge variant={isFraud ? "destructive" : "default"}>
          {isFraud ? "사기" : "정상"}
        </Badge>
      );
    },
  },
];
```

---

## 6. Storybook (현업 표준)

### 6-1. Storybook이란?

**컴포넌트 문서화 + 실습 도구**

| 기능 | 설명 |
|------|------|
| **컴포넌트 카탈로그** | 모든 컴포넌트를 한눈에 |
| **Props 테스트** | Controls 패널에서 실시간 변경 |
| **문서화** | 디자이너/PM과 공유 |
| **UI 테스트** | 시각적 회귀 테스트 |

### 6-2. Story 작성

```tsx
// TransactionCard.stories.tsx
import type { Meta, StoryObj } from '@storybook/react';
import { TransactionCard } from './TransactionCard';

const meta: Meta<typeof TransactionCard> = {
  title: 'Components/TransactionCard',
  component: TransactionCard,
  tags: ['autodocs'],
};

export default meta;
type Story = StoryObj<typeof meta>;

// 정상 거래
export const Normal: Story = {
  args: {
    transaction_id: 'TXN_001',
    amount: 150000,
    fraud_probability: 0.12,
    is_fraud: false,
  },
};

// 사기 거래
export const Fraud: Story = {
  args: {
    transaction_id: 'TXN_002',
    amount: 5000000,
    fraud_probability: 0.87,
    is_fraud: true,
  },
};
```

### 6-3. 사용법

```bash
# Storybook 실행
npm run storybook

# http://localhost:6006 접속
# 좌측 메뉴에서 컴포넌트 선택
# Controls 패널에서 Props 변경
```

---

## 7. API 연동

### 7-1. fetch 기본

```jsx
// POST 요청
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    transaction_id: 'TXN_001',
    TransactionAmt: 150000,
    ProductCD: 'W',
    hour: 14,
  }),
});
const result = await response.json();
```

### 7-2. API 클라이언트 패턴

```ts
// api/client.ts
const API_BASE = 'http://localhost:8000';

export async function predictFraud(transaction: Transaction) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(transaction),
  });

  if (!response.ok) {
    throw new Error(`API 오류: ${response.status}`);
  }

  return response.json();
}
```

---

## 면접 Q&A

### Q: "왜 Storybook을 사용했나요?"
> "현업에서 컴포넌트 문서화 표준입니다. 디자이너/PM과 협업할 때
> 컴포넌트를 공유하고, Props별 다양한 상태를 테스트할 수 있습니다.
> 또한 UI 테스트 자동화에도 활용됩니다."

### Q: "왜 Tailwind CSS를 선택했나요?"
> "2024년 기준 CSS 프레임워크 npm 다운로드 1위입니다 (주간 1,490만).
> 유틸리티 클래스 방식으로 빠르게 스타일링할 수 있습니다."

### Q: "왜 shadcn/ui를 선택했나요?"
> "Tailwind 기반 컴포넌트 중 가장 인기 있고, 복사/붙여넣기 방식이라
> 커스터마이징이 자유롭습니다. DataTable이 TanStack Table 기반입니다."

### Q: "useState와 useEffect의 차이는?"
> "useState는 컴포넌트의 상태를 관리하고,
> useEffect는 API 호출 같은 사이드 이펙트를 처리합니다."

---

## 체크포인트

- [ ] React 컴포넌트 개념을 안다
- [ ] Props로 데이터 전달하는 법을 안다
- [ ] useState로 상태 관리할 수 있다
- [ ] Tailwind 유틸리티 클래스를 사용할 수 있다
- [ ] shadcn/ui DataTable 컬럼을 정의할 수 있다
- [ ] Storybook에서 컴포넌트를 테스트할 수 있다
- [ ] fetch로 API 호출할 수 있다

---

## 다음 단계

**실습**: `frontend/` 폴더에서 Storybook 실행 후 컴포넌트 확인

```bash
cd frontend
npm run storybook
```
