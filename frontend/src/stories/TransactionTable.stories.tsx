import type { Meta, StoryObj } from "@storybook/react"
import { TransactionTable, type Transaction } from "../components/TransactionTable"

const meta: Meta<typeof TransactionTable> = {
  title: "Components/TransactionTable",
  component: TransactionTable,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
  },
}

export default meta
type Story = StoryObj<typeof meta>

// 샘플 데이터
const sampleTransactions: Transaction[] = [
  {
    transaction_id: "TXN_001",
    amount: 150000,
    fraud_probability: 0.12,
    is_fraud: false,
    top_factors: [
      { feature: "거래금액", impact: -0.05 },
      { feature: "거래시간", impact: -0.03 },
    ],
    explanation_text: "정상적인 거래 패턴입니다.",
  },
  {
    transaction_id: "TXN_002",
    amount: 5000000,
    fraud_probability: 0.87,
    is_fraud: true,
    top_factors: [
      { feature: "거래금액", impact: 0.35 },
      { feature: "거래시간", impact: 0.22 },
    ],
    explanation_text: "고액 거래와 비정상 시간대 거래가 탐지되었습니다.",
  },
  {
    transaction_id: "TXN_003",
    amount: 800000,
    fraud_probability: 0.52,
    is_fraud: true,
    top_factors: [
      { feature: "거래금액", impact: 0.12 },
      { feature: "디바이스타입", impact: 0.08 },
    ],
    explanation_text: "경계선 거래입니다. 추가 확인이 필요합니다.",
  },
  {
    transaction_id: "TXN_004",
    amount: 9500000,
    fraud_probability: 0.23,
    is_fraud: false,
    top_factors: [
      { feature: "고객등급", impact: -0.25 },
      { feature: "거래이력", impact: -0.12 },
    ],
    explanation_text: "VIP 고객의 정상 거래입니다.",
  },
  {
    transaction_id: "TXN_005",
    amount: 50000,
    fraud_probability: 0.05,
    is_fraud: false,
    top_factors: [{ feature: "거래금액", impact: -0.08 }],
    explanation_text: "소액 정상 거래입니다.",
  },
  {
    transaction_id: "TXN_006",
    amount: 3200000,
    fraud_probability: 0.78,
    is_fraud: true,
    top_factors: [
      { feature: "이메일도메인", impact: 0.28 },
      { feature: "거래금액", impact: 0.18 },
    ],
    explanation_text: "의심스러운 이메일 도메인이 사용되었습니다.",
  },
  {
    transaction_id: "TXN_007",
    amount: 280000,
    fraud_probability: 0.08,
    is_fraud: false,
    top_factors: [{ feature: "거래시간", impact: -0.04 }],
    explanation_text: "정상 거래입니다.",
  },
  {
    transaction_id: "TXN_008",
    amount: 1500000,
    fraud_probability: 0.65,
    is_fraud: true,
    top_factors: [
      { feature: "카드타입", impact: 0.15 },
      { feature: "거래금액", impact: 0.12 },
    ],
    explanation_text: "새로운 카드로 고액 거래가 발생했습니다.",
  },
  {
    transaction_id: "TXN_009",
    amount: 75000,
    fraud_probability: 0.03,
    is_fraud: false,
    top_factors: [{ feature: "거래이력", impact: -0.1 }],
    explanation_text: "기존 패턴과 일치하는 거래입니다.",
  },
  {
    transaction_id: "TXN_010",
    amount: 4800000,
    fraud_probability: 0.91,
    is_fraud: true,
    top_factors: [
      { feature: "거래금액", impact: 0.4 },
      { feature: "거래시간", impact: 0.25 },
      { feature: "디바이스타입", impact: 0.15 },
    ],
    explanation_text: "여러 위험 지표가 동시에 탐지되었습니다.",
  },
  {
    transaction_id: "TXN_011",
    amount: 120000,
    fraud_probability: 0.11,
    is_fraud: false,
    top_factors: [{ feature: "거래금액", impact: -0.06 }],
    explanation_text: "정상 거래입니다.",
  },
  {
    transaction_id: "TXN_012",
    amount: 2100000,
    fraud_probability: 0.72,
    is_fraud: true,
    top_factors: [
      { feature: "거래금액", impact: 0.22 },
      { feature: "이메일도메인", impact: 0.18 },
    ],
    explanation_text: "고액 거래와 의심스러운 이메일이 탐지되었습니다.",
  },
]

// 기본 테이블
export const Default: Story = {
  args: {
    data: sampleTransactions,
  },
}

// 빈 테이블
export const Empty: Story = {
  args: {
    data: [],
  },
}

// 클릭 가능한 테이블
export const Clickable: Story = {
  args: {
    data: sampleTransactions,
    onRowClick: (transaction) => {
      alert(
        `선택된 거래: ${transaction.transaction_id}\n${transaction.explanation_text}`
      )
    },
  },
}

// 소량 데이터
export const FewItems: Story = {
  args: {
    data: sampleTransactions.slice(0, 3),
  },
}
