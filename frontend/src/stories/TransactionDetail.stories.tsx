import type { Meta, StoryObj } from "@storybook/react"
import { TransactionDetail } from "../components/TransactionDetail"

const meta: Meta<typeof TransactionDetail> = {
  title: "Components/TransactionDetail",
  component: TransactionDetail,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
  },
}

export default meta
type Story = StoryObj<typeof meta>

// 사기 거래 (고위험)
export const HighRiskFraud: Story = {
  args: {
    transaction_id: "TXN_002",
    amount: 5000000,
    fraud_probability: 0.91,
    is_fraud: true,
    top_factors: [
      { feature: "거래금액", impact: 0.35 },
      { feature: "거래시간", impact: 0.22 },
      { feature: "이메일도메인", impact: 0.15 },
      { feature: "디바이스타입", impact: 0.08 },
      { feature: "카드사용횟수", impact: -0.05 },
    ],
    explanation_text:
      "이 거래는 높은 사기 확률(91%)로 분류되었습니다. 주요 위험 요인은 평균 대비 매우 높은 거래금액(₩5,000,000), 새벽 시간대(03:42)의 거래, 그리고 일회용 이메일 도메인 사용입니다. 해당 카드로는 최근 30일간 거래가 없었던 점도 의심 요인입니다.",
  },
}

// 정상 거래 (저위험)
export const LowRiskNormal: Story = {
  args: {
    transaction_id: "TXN_001",
    amount: 150000,
    fraud_probability: 0.08,
    is_fraud: false,
    top_factors: [
      { feature: "거래이력", impact: -0.12 },
      { feature: "고객등급", impact: -0.08 },
      { feature: "거래시간", impact: -0.05 },
      { feature: "거래금액", impact: 0.02 },
    ],
    explanation_text:
      "이 거래는 정상 거래로 분류되었습니다(사기 확률 8%). 해당 고객은 VIP 등급으로 6개월 이상의 안정적인 거래 이력이 있으며, 거래 패턴이 기존 행동과 일치합니다. 업무 시간대(14:23)에 발생한 일반적인 금액의 거래입니다.",
  },
}

// 경계선 거래 (중위험)
export const BorderlineRisk: Story = {
  args: {
    transaction_id: "TXN_003",
    amount: 1200000,
    fraud_probability: 0.52,
    is_fraud: true,
    top_factors: [
      { feature: "거래금액", impact: 0.15 },
      { feature: "디바이스타입", impact: 0.08 },
      { feature: "거래이력", impact: -0.08 },
      { feature: "카드타입", impact: -0.04 },
    ],
    explanation_text:
      "이 거래는 경계선 위험도(52%)로 분류되었습니다. 평균보다 높은 거래금액과 새로운 디바이스에서의 접속이 위험 요인이나, 기존 거래 이력과 카드 정보는 정상적입니다. 추가 확인을 권장합니다.",
  },
}

// VIP 고객 고액 정상 거래
export const VIPHighAmount: Story = {
  args: {
    transaction_id: "TXN_004",
    amount: 9500000,
    fraud_probability: 0.18,
    is_fraud: false,
    top_factors: [
      { feature: "고객등급", impact: -0.28 },
      { feature: "거래이력", impact: -0.15 },
      { feature: "이메일도메인", impact: -0.05 },
      { feature: "거래금액", impact: 0.12 },
    ],
    explanation_text:
      "고액 거래(₩9,500,000)이지만 정상으로 분류되었습니다. VIP 고객으로서 유사한 금액의 거래 이력이 있으며, 인증된 기업 이메일을 사용했습니다. 거래금액이 높아 위험 점수가 일부 상승했으나, 전체적인 프로필은 안전합니다.",
  },
}

// 닫기 버튼 있는 버전
export const WithCloseButton: Story = {
  args: {
    transaction_id: "TXN_005",
    amount: 350000,
    fraud_probability: 0.25,
    is_fraud: false,
    top_factors: [
      { feature: "거래시간", impact: -0.08 },
      { feature: "거래금액", impact: 0.05 },
    ],
    explanation_text: "일반적인 거래 패턴입니다. 특이사항이 발견되지 않았습니다.",
    onClose: () => alert("닫기 버튼 클릭!"),
  },
}
