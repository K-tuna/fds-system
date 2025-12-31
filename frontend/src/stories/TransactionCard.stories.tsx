import type { Meta, StoryObj } from "@storybook/react"
import { TransactionCard } from "../components/TransactionCard"

const meta: Meta<typeof TransactionCard> = {
  title: "Components/TransactionCard",
  component: TransactionCard,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
  },
  argTypes: {
    amount: {
      control: { type: "number", min: 0, max: 10000000, step: 10000 },
      description: "거래 금액",
    },
    fraud_probability: {
      control: { type: "range", min: 0, max: 1, step: 0.01 },
      description: "사기 확률 (0-1)",
    },
    is_fraud: {
      control: "boolean",
      description: "사기 여부",
    },
  },
}

export default meta
type Story = StoryObj<typeof meta>

// 정상 거래
export const Normal: Story = {
  args: {
    transaction_id: "TXN_001",
    amount: 150000,
    fraud_probability: 0.12,
    is_fraud: false,
    top_factors: [
      { feature: "거래금액", impact: -0.05 },
      { feature: "거래시간", impact: -0.03 },
      { feature: "카드타입", impact: -0.02 },
    ],
  },
}

// 사기 거래
export const Fraud: Story = {
  args: {
    transaction_id: "TXN_002",
    amount: 5000000,
    fraud_probability: 0.87,
    is_fraud: true,
    top_factors: [
      { feature: "거래금액", impact: 0.35 },
      { feature: "거래시간", impact: 0.22 },
      { feature: "이메일도메인", impact: 0.15 },
    ],
  },
}

// 경계선 거래 (50% 근처)
export const Borderline: Story = {
  args: {
    transaction_id: "TXN_003",
    amount: 800000,
    fraud_probability: 0.52,
    is_fraud: true,
    top_factors: [
      { feature: "거래금액", impact: 0.12 },
      { feature: "디바이스타입", impact: 0.08 },
      { feature: "거래시간", impact: -0.05 },
    ],
  },
}

// 고액 정상 거래
export const HighAmountNormal: Story = {
  args: {
    transaction_id: "TXN_004",
    amount: 9500000,
    fraud_probability: 0.23,
    is_fraud: false,
    top_factors: [
      { feature: "고객등급", impact: -0.25 },
      { feature: "거래이력", impact: -0.12 },
      { feature: "거래금액", impact: 0.08 },
    ],
  },
}

// 요인 없는 카드
export const NoFactors: Story = {
  args: {
    transaction_id: "TXN_005",
    amount: 50000,
    fraud_probability: 0.05,
    is_fraud: false,
  },
}
