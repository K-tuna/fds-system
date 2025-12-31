import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

export interface TransactionCardProps {
  transaction_id: string
  amount: number
  fraud_probability: number
  is_fraud: boolean
  top_factors?: Array<{ feature: string; impact: number }>
}

export function TransactionCard({
  transaction_id,
  amount,
  fraud_probability,
  is_fraud,
  top_factors,
}: TransactionCardProps) {
  const probabilityPercent = (fraud_probability * 100).toFixed(1)

  return (
    <Card className="w-full max-w-md">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">
          거래 ID: {transaction_id}
        </CardTitle>
        <Badge variant={is_fraud ? "destructive" : "success"}>
          {is_fraud ? "사기" : "정상"}
        </Badge>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-muted-foreground">금액</span>
            <span className="font-bold text-lg">
              ₩{amount.toLocaleString()}
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-muted-foreground">사기 확률</span>
            <span
              className={`font-bold text-lg ${
                fraud_probability >= 0.5 ? "text-red-500" : "text-green-500"
              }`}
            >
              {probabilityPercent}%
            </span>
          </div>

          {/* 사기 확률 바 */}
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all ${
                fraud_probability >= 0.5 ? "bg-red-500" : "bg-green-500"
              }`}
              style={{ width: `${probabilityPercent}%` }}
            />
          </div>

          {/* 주요 요인 */}
          {top_factors && top_factors.length > 0 && (
            <div className="pt-2 border-t">
              <p className="text-sm font-medium mb-2">주요 요인</p>
              <ul className="space-y-1">
                {top_factors.slice(0, 3).map((factor, index) => (
                  <li
                    key={index}
                    className="text-sm flex justify-between items-center"
                  >
                    <span className="text-muted-foreground">
                      {factor.feature}
                    </span>
                    <span
                      className={
                        factor.impact > 0 ? "text-red-500" : "text-green-500"
                      }
                    >
                      {factor.impact > 0 ? "+" : ""}
                      {(factor.impact * 100).toFixed(1)}%
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
