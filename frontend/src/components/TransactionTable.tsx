import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  type SortingState,
  useReactTable,
} from "@tanstack/react-table"
import { useState } from "react"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

import type { RiskLevel } from "@/api/client"

export interface Transaction {
  transaction_id: string
  amount: number
  fraud_probability: number
  is_fraud: boolean
  risk_level?: RiskLevel    // 다단계 위험도
  top_factors?: Array<{ feature: string; impact: number }>
  explanation_text?: string
  _analyzed?: boolean       // 분석 완료 여부
  _actual_label?: number    // 실제 정답 (0: 정상, 1: 사기)
}

// 위험도 레벨별 표시 설정
const RISK_LEVEL_CONFIG = {
  approve: { label: "승인", variant: "success" as const, color: "text-green-600" },
  verify: { label: "추가인증", variant: "warning" as const, color: "text-yellow-600" },
  hold: { label: "보류", variant: "secondary" as const, color: "text-orange-600" },
  block: { label: "차단", variant: "destructive" as const, color: "text-red-600" },
}

const columns: ColumnDef<Transaction>[] = [
  {
    accessorKey: "transaction_id",
    header: "거래 ID",
  },
  {
    accessorKey: "amount",
    header: "금액",
    cell: ({ row }) => {
      const amount = row.getValue("amount") as number
      return <span className="font-medium">${amount.toLocaleString()}</span>
    },
  },
  {
    accessorKey: "fraud_probability",
    header: "사기 확률",
    cell: ({ row }) => {
      const analyzed = row.original._analyzed
      if (!analyzed) {
        return <span className="text-gray-400">-</span>
      }
      const prob = row.getValue("fraud_probability") as number
      const percent = (prob * 100).toFixed(1)
      return (
        <span
          className={`font-medium ${
            prob >= 0.5 ? "text-red-500" : "text-green-500"
          }`}
        >
          {percent}%
        </span>
      )
    },
  },
  {
    accessorKey: "risk_level",
    header: "판정",
    cell: ({ row }) => {
      const analyzed = row.original._analyzed
      if (!analyzed) {
        return (
          <Badge variant="secondary">
            판단 안함
          </Badge>
        )
      }
      const riskLevel = row.original.risk_level || "approve"
      const config = RISK_LEVEL_CONFIG[riskLevel]
      return (
        <Badge variant={config.variant} className={config.color}>
          {config.label}
        </Badge>
      )
    },
  },
  {
    accessorKey: "top_factors",
    header: "주요 요인",
    cell: ({ row }) => {
      const factors = row.getValue("top_factors") as
        | Array<{ feature: string; impact: number }>
        | undefined
      if (!factors || factors.length === 0) return "-"
      const topFactor = factors[0]
      return (
        <span className="text-sm text-muted-foreground">
          {topFactor.feature} ({topFactor.impact > 0 ? "+" : ""}
          {(topFactor.impact * 100).toFixed(0)}%)
        </span>
      )
    },
  },
]

interface TransactionTableProps {
  data: Transaction[]
  onRowClick?: (transaction: Transaction) => void
}

export function TransactionTable({ data, onRowClick }: TransactionTableProps) {
  const [sorting, setSorting] = useState<SortingState>([])

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    onSortingChange: setSorting,
    state: {
      sorting,
    },
    initialState: {
      pagination: {
        pageSize: 10,
      },
    },
  })

  return (
    <div className="space-y-4">
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead
                    key={header.id}
                    onClick={header.column.getToggleSortingHandler()}
                    className={
                      header.column.getCanSort()
                        ? "cursor-pointer select-none"
                        : ""
                    }
                  >
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                    {{
                      asc: " ↑",
                      desc: " ↓",
                    }[header.column.getIsSorted() as string] ?? null}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  className={onRowClick ? "cursor-pointer" : ""}
                  onClick={() => onRowClick?.(row.original)}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-24 text-center"
                >
                  거래 데이터가 없습니다.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      {/* 페이지네이션 */}
      <div className="flex items-center justify-between px-2">
        <div className="text-sm text-muted-foreground">
          총 {table.getFilteredRowModel().rows.length}건 중{" "}
          {table.getState().pagination.pageIndex *
            table.getState().pagination.pageSize +
            1}
          -
          {Math.min(
            (table.getState().pagination.pageIndex + 1) *
              table.getState().pagination.pageSize,
            table.getFilteredRowModel().rows.length
          )}
          건
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            이전
          </Button>
          <span className="text-sm">
            {table.getState().pagination.pageIndex + 1} /{" "}
            {table.getPageCount()}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            다음
          </Button>
        </div>
      </div>
    </div>
  )
}
