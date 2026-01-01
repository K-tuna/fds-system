import { useState, useEffect } from "react"
import { TransactionTable, type Transaction } from "@/components/TransactionTable"
import { TransactionDetail } from "@/components/TransactionDetail"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  checkHealth,
  predictDirectBatch,
  getSamples,
  type HealthResponse,
} from "@/api/client"

export function Dashboard() {
  // 샘플 관련 상태 (447개 피처 전체)
  const [sampleCount, setSampleCount] = useState(10)
  const [samples, setSamples] = useState<Record<string, unknown>[]>([])
  const [isAnalyzed, setIsAnalyzed] = useState(false)

  // 분석 결과 상태
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [selectedTransaction, setSelectedTransaction] =
    useState<Transaction | null>(null)

  // 로딩/에러 상태
  const [loadingSamples, setLoadingSamples] = useState(false)
  const [loadingAnalysis, setLoadingAnalysis] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)

  // 서버 상태 확인
  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  // 샘플 데이터 로드 (447개 피처 전체)
  const handleLoadSamples = async () => {
    setLoadingSamples(true)
    setError(null)
    setIsAnalyzed(false)
    setSelectedTransaction(null)

    try {
      const data = await getSamples(sampleCount)
      setSamples(data)

      // 샘플을 테이블에 표시 (분석 전 상태)
      setTransactions(
        data.map((s) => ({
          transaction_id: s.transaction_id as string,
          amount: s.TransactionAmt as number,
          fraud_probability: 0,
          is_fraud: false,
          top_factors: [],
          explanation_text: "",
          _analyzed: false,
          _actual_label: s._actual_label as number,
        }))
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : "샘플 로드 중 오류 발생")
    } finally {
      setLoadingSamples(false)
    }
  }

  // 분석 실행 (직접 예측 - 447개 피처 그대로 전달)
  const handleAnalyze = async () => {
    if (samples.length === 0) return

    setLoadingAnalysis(true)
    setError(null)

    try {
      // 직접 예측 API 호출 (인코딩 변환 없이 바로 모델에 입력)
      const results = await predictDirectBatch(samples)

      setTransactions(
        results.map((r) => {
          const sample = samples.find((s) => s.transaction_id === r.transaction_id)
          return {
            transaction_id: r.transaction_id,
            amount: (sample?.TransactionAmt as number) || 0,
            fraud_probability: r.fraud_probability,
            is_fraud: r.is_fraud,
            risk_level: r.risk_level,
            top_factors: r.top_factors,
            explanation_text: r.explanation_text,
            _analyzed: true,
            _actual_label: sample?._actual_label as number,
          }
        })
      )
      setIsAnalyzed(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : "분석 중 오류 발생")
    } finally {
      setLoadingAnalysis(false)
    }
  }

  // 통계 계산 (분석 후에만)
  const stats = {
    total: transactions.length,
    fraud: isAnalyzed ? transactions.filter((t) => t.is_fraud).length : 0,
    normal: isAnalyzed ? transactions.filter((t) => !t.is_fraud).length : 0,
    pending: isAnalyzed ? 0 : transactions.length,
    avgProbability:
      isAnalyzed && transactions.length > 0
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
        <div className="flex items-center gap-4">
          {/* 드롭다운 */}
          <div className="flex items-center gap-2">
            <label htmlFor="sampleCount" className="text-sm font-medium">
              샘플 개수:
            </label>
            <select
              id="sampleCount"
              value={sampleCount}
              onChange={(e) => setSampleCount(Number(e.target.value))}
              className="border rounded px-3 py-2 bg-background"
            >
              {[10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map((n) => (
                <option key={n} value={n}>
                  {n}개
                </option>
              ))}
            </select>
          </div>

          {/* 샘플 로드 버튼 */}
          <Button
            onClick={handleLoadSamples}
            disabled={loadingSamples || !health}
            variant="outline"
          >
            {loadingSamples ? "로드 중..." : "샘플 로드"}
          </Button>

          {/* 분석 실행 버튼 */}
          <Button
            onClick={handleAnalyze}
            disabled={loadingAnalysis || samples.length === 0 || !health}
          >
            {loadingAnalysis ? "분석 중..." : "분석 실행"}
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
                  {isAnalyzed ? "사기 거래" : "판단 대기"}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className={`text-3xl font-bold ${isAnalyzed ? "text-red-500" : "text-gray-400"}`}>
                  {isAnalyzed ? stats.fraud : stats.pending}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">
                  정상 거래
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className={`text-3xl font-bold ${isAnalyzed ? "text-green-500" : "text-gray-400"}`}>
                  {isAnalyzed ? stats.normal : "-"}
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
                  {isAnalyzed ? `${(stats.avgProbability * 100).toFixed(1)}%` : "-"}
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
                  selectedTransaction.explanation_text || "분석 실행 후 확인 가능"
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
