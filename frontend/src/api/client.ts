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

// 위험도 레벨 타입
export type RiskLevel = "approve" | "verify" | "hold" | "block"

// 거래 예측 응답 타입
export interface PredictResponse {
  transaction_id: string
  fraud_probability: number
  is_fraud: boolean
  risk_level: RiskLevel  // 다단계 위험도
  top_factors: Array<{ feature: string; impact: number }>
  explanation_text: string
}

// 샘플 거래 타입 (분석 전 - 실제 데이터)
export interface SampleTransaction extends PredictRequest {
  _actual_label?: number  // 정답 라벨 (0: 정상, 1: 사기)
}

// 분석된 거래 타입 (분석 후)
export interface AnalyzedTransaction extends PredictResponse {
  TransactionAmt?: number  // 원본 금액
  _actual_label?: number   // 정답 라벨
  _analyzed: boolean       // 분석 완료 여부
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

// 샘플 데이터 로드 (447개 피처 전체 반환)
export async function getSamples(count: number = 10): Promise<Record<string, unknown>[]> {
  const response = await fetch(`${API_BASE}/samples?count=${count}`)

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new APIError(
      response.status,
      error.detail || `샘플 데이터 로드 실패 (${response.status})`
    )
  }

  return response.json()
}

// 직접 예측 (이미 전처리된 피처 사용 - /samples에서 받은 데이터용)
export async function predictDirectBatch(
  features: Record<string, unknown>[]
): Promise<PredictResponse[]> {
  const response = await fetch(`${API_BASE}/predict/direct/batch`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(features),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new APIError(
      response.status,
      error.detail || `직접 예측 실패 (${response.status})`
    )
  }

  return response.json()
}
