import { useState, useEffect } from "react"
import { TransactionTable, type Transaction } from "@/components/TransactionTable"
import { TransactionDetail } from "@/components/TransactionDetail"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  checkHealth,
  predictFraudBatch,
  type PredictRequest,
  type HealthResponse,
} from "@/api/client"

// 샘플 거래 데이터 생성
function generateSampleTransactions(): PredictRequest[] {
  const productCDs = ["W", "C", "R", "H", "S"]
  const card4s = ["visa", "mastercard", "discover", "american express"]
  const card6s = ["debit", "credit"]
  const deviceTypes = ["desktop", "mobile"]

  return Array.from({ length: 20 }, (_, i) => ({
    transaction_id: `TXN_${String(i + 1).padStart(3, "0")}`,
    TransactionAmt: Math.floor(Math.random() * 5000000) + 10000,
    ProductCD: productCDs[Math.floor(Math.random() * productCDs.length)],
    card1: Math.floor(Math.random() * 10000) + 1000,
    card2: Math.floor(Math.random() * 500) + 100,
    card3: Math.floor(Math.random() * 200) + 50,
    card4: card4s[Math.floor(Math.random() * card4s.length)],
    card5: Math.floor(Math.random() * 300) + 100,
    card6: card6s[Math.floor(Math.random() * card6s.length)],
    hour: Math.floor(Math.random() * 24),
    addr1: Math.floor(Math.random() * 500) + 100,
    addr2: Math.floor(Math.random() * 100) + 10,
    dist1: Math.random() * 100,
    P_emaildomain: Math.random() > 0.7 ? "gmail.com" : "anonymous.com",
    DeviceType: deviceTypes[Math.floor(Math.random() * deviceTypes.length)],
  }))
}

export function Dashboard() {
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [selectedTransaction, setSelectedTransaction] =
    useState<Transaction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)

  // 서버 상태 확인
  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  // 거래 예측 실행
  const handlePredict = async () => {
    setLoading(true)
    setError(null)

    try {
      const sampleData = generateSampleTransactions()
      const results = await predictFraudBatch(sampleData)

      setTransactions(
        results.map((r) => ({
          transaction_id: r.transaction_id,
          amount:
            sampleData.find((s) => s.transaction_id === r.transaction_id)
              ?.TransactionAmt || 0,
          fraud_probability: r.fraud_probability,
          is_fraud: r.is_fraud,
          top_factors: r.top_factors,
          explanation_text: r.explanation_text,
        }))
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : "예측 중 오류 발생")
    } finally {
      setLoading(false)
    }
  }

  // 통계 계산
  const stats = {
    total: transactions.length,
    fraud: transactions.filter((t) => t.is_fraud).length,
    normal: transactions.filter((t) => !t.is_fraud).length,
    avgProbability:
      transactions.length > 0
        ? transactions.reduce((sum, t) => sum + t.fraud_probability, 0) /
          transactions.length
        : 0,
  }

  return (
    <div className="min-h-screen bg-background">
      {/* 헤더 */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">FDS Admin</h1>
            <p className="text-sm text-muted-foreground">
              이상거래 탐지 시스템
            </p>
          </div>
          <div className="flex items-center gap-4">
            <Badge variant={health?.model_loaded ? "success" : "destructive"}>
              {health?.model_loaded ? "모델 로드됨" : "모델 미로드"}
            </Badge>
            <Badge variant={health ? "outline" : "destructive"}>
              API: {health ? "연결됨" : "연결 안됨"}
            </Badge>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6 space-y-6">
        {/* 컨트롤 */}
        <div className="flex gap-4">
          <Button onClick={handlePredict} disabled={loading || !health}>
            {loading ? "분석 중..." : "샘플 거래 분석"}
          </Button>
          {error && (
            <Badge variant="destructive" className="self-center">
              {error}
            </Badge>
          )}
        </div>

        {/* 통계 카드 */}
        {transactions.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">
                  총 거래
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold">{stats.total}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">
                  사기 거래
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold text-red-500">{stats.fraud}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">
                  정상 거래
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold text-green-500">
                  {stats.normal}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">
                  평균 사기 확률
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold">
                  {(stats.avgProbability * 100).toFixed(1)}%
                </p>
              </CardContent>
            </Card>
          </div>
        )}

        {/* 거래 테이블 + 상세 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <TransactionTable
              data={transactions}
              onRowClick={setSelectedTransaction}
            />
          </div>
          <div>
            {selectedTransaction ? (
              <TransactionDetail
                {...selectedTransaction}
                top_factors={selectedTransaction.top_factors || []}
                explanation_text={
                  selectedTransaction.explanation_text || "설명 없음"
                }
                onClose={() => setSelectedTransaction(null)}
              />
            ) : (
              <Card className="h-full flex items-center justify-center min-h-[400px]">
                <CardContent>
                  <p className="text-muted-foreground text-center">
                    거래를 선택하면
                    <br />
                    상세 분석이 표시됩니다
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
