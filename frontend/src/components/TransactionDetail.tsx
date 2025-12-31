import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

export interface TransactionDetailProps {
  transaction_id: string
  amount: number
  fraud_probability: number
  is_fraud: boolean
  top_factors: Array<{ feature: string; impact: number }>
  explanation_text: string
  onClose?: () => void
}

export function TransactionDetail({
  transaction_id,
  amount,
  fraud_probability,
  is_fraud,
  top_factors,
  explanation_text,
  onClose,
}: TransactionDetailProps) {
  const probabilityPercent = (fraud_probability * 100).toFixed(1)

  // SHAP 영향도 계산: 양수(사기 방향)와 음수(정상 방향) 분리
  const positiveFactors = top_factors.filter((f) => f.impact > 0)
  const negativeFactors = top_factors.filter((f) => f.impact < 0)

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl">거래 상세 분석</CardTitle>
            <CardDescription>SHAP 기반 설명</CardDescription>
          </div>
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              닫기
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* 기본 정보 */}
        <div className="grid grid-cols-2 gap-4 p-4 bg-muted rounded-lg">
          <div>
            <p className="text-sm text-muted-foreground">거래 ID</p>
            <p className="font-medium">{transaction_id}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">거래 금액</p>
            <p className="font-medium text-lg">₩{amount.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">사기 확률</p>
            <p
              className={`font-bold text-xl ${
                fraud_probability >= 0.5 ? "text-red-500" : "text-green-500"
              }`}
            >
              {probabilityPercent}%
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">최종 판정</p>
            <Badge
              variant={is_fraud ? "destructive" : "success"}
              className="text-sm"
            >
              {is_fraud ? "사기 거래" : "정상 거래"}
            </Badge>
          </div>
        </div>

        {/* 사기 확률 바 */}
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-green-600 font-medium">정상</span>
            <span className="text-red-600 font-medium">사기</span>
          </div>
          <div className="relative h-4 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="absolute left-0 top-0 h-full bg-gradient-to-r from-green-500 to-red-500"
              style={{ width: "100%" }}
            />
            <div
              className="absolute top-0 h-full w-1 bg-black"
              style={{ left: `${probabilityPercent}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </div>

        {/* SHAP 요인 분석 */}
        <div>
          <h3 className="font-semibold mb-3">SHAP 요인 분석</h3>

          {/* 사기 방향 요인 */}
          {positiveFactors.length > 0 && (
            <div className="mb-4">
              <p className="text-sm text-red-600 font-medium mb-2">
                사기 위험 증가 요인
              </p>
              <div className="space-y-2">
                {positiveFactors.map((factor, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <div className="flex-1">
                      <div className="flex justify-between text-sm">
                        <span>{factor.feature}</span>
                        <span className="text-red-500 font-medium">
                          +{(factor.impact * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-2 bg-gray-100 rounded-full mt-1">
                        <div
                          className="h-full bg-red-500 rounded-full"
                          style={{
                            width: `${Math.min(Math.abs(factor.impact) * 200, 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 정상 방향 요인 */}
          {negativeFactors.length > 0 && (
            <div>
              <p className="text-sm text-green-600 font-medium mb-2">
                정상 거래 지지 요인
              </p>
              <div className="space-y-2">
                {negativeFactors.map((factor, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <div className="flex-1">
                      <div className="flex justify-between text-sm">
                        <span>{factor.feature}</span>
                        <span className="text-green-500 font-medium">
                          {(factor.impact * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-2 bg-gray-100 rounded-full mt-1">
                        <div
                          className="h-full bg-green-500 rounded-full"
                          style={{
                            width: `${Math.min(Math.abs(factor.impact) * 200, 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 자연어 설명 */}
        <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
          <h3 className="font-semibold text-blue-800 mb-2">AI 분석 요약</h3>
          <p className="text-blue-700">{explanation_text}</p>
        </div>
      </CardContent>
    </Card>
  )
}
