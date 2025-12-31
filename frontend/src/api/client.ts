// API 기본 URL (환경변수로 설정 가능)
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000"

// 거래 예측 요청 타입
export interface PredictRequest {
  transaction_id: string
  TransactionAmt: number
  ProductCD?: string
  card1?: number
  card2?: number
  card3?: number
  card4?: string
  card5?: number
  card6?: string
  hour?: number
  addr1?: number
  addr2?: number
  dist1?: number
  P_emaildomain?: string
  R_emaildomain?: string
  DeviceType?: string
  DeviceInfo?: string
}

// 거래 예측 응답 타입
export interface PredictResponse {
  transaction_id: string
  fraud_probability: number
  is_fraud: boolean
  top_factors: Array<{ feature: string; impact: number }>
  explanation_text: string
}

// 헬스 체크 응답 타입
export interface HealthResponse {
  status: string
  model_loaded: boolean
  version: string
}

// API 에러 타입
export class APIError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = "APIError"
    this.status = status
  }
}

// 헬스 체크
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`)

  if (!response.ok) {
    throw new APIError(response.status, "서버 상태 확인 실패")
  }

  return response.json()
}

// 단일 거래 예측
export async function predictFraud(
  transaction: PredictRequest
): Promise<PredictResponse> {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(transaction),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new APIError(
      response.status,
      error.detail || `예측 실패 (${response.status})`
    )
  }

  return response.json()
}

// 배치 거래 예측
export async function predictFraudBatch(
  transactions: PredictRequest[]
): Promise<PredictResponse[]> {
  const response = await fetch(`${API_BASE}/predict/batch`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(transactions),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new APIError(
      response.status,
      error.detail || `배치 예측 실패 (${response.status})`
    )
  }

  return response.json()
}
