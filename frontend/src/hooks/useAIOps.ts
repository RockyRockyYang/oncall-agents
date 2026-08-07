import { useState, useCallback } from 'react'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import { getActiveSessionId } from '../lib/session'

export type AIOpsStatus = 'idle' | 'running' | 'done' | 'error'

interface AIOpsEvent {
  type: 'plan' | 'step_complete' | 'status' | 'report' | 'complete' | 'error'
  message?: string
  plan?: string[]
  report?: string
}

export function useAIOps() {
  const [status, setStatus] = useState<AIOpsStatus>('idle')
  const [plan, setPlan] = useState<string[]>([])
  const [completedCount, setCompletedCount] = useState(0)
  const [statusMessage, setStatusMessage] = useState('')
  const [report, setReport] = useState('')
  const [error, setError] = useState('')

  const investigate = useCallback(
    async (alert: string) => {
      if (!alert.trim() || status === 'running') return

      setStatus('running')
      setPlan([])
      setCompletedCount(0)
      setStatusMessage('')
      setReport('')
      setError('')

      try {
        await fetchEventSource('/api/aiops/investigate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ alert, session_id: getActiveSessionId() }),
          openWhenHidden: true, // tab 切到后台时不要断流
          onmessage(ev) {
            // 防御性解析：中间层（代理/心跳帧）可能混入空帧或非 JSON 数据，
            // 单帧解析失败不应该打断整条流。
            if (!ev.data) return
            let event: AIOpsEvent
            try {
              event = JSON.parse(ev.data)
            } catch {
              console.warn('Skipping malformed SSE frame:', ev.data)
              return
            }

            if (event.type === 'plan') {
              setPlan(event.plan ?? [])
            } else if (event.type === 'step_complete') {
              setCompletedCount((c) => c + 1)
            } else if (event.type === 'status') {
              setStatusMessage(event.message ?? '')
            } else if (event.type === 'report') {
              setReport(event.report ?? '')
            } else if (event.type === 'complete') {
              setStatus('done')
            } else if (event.type === 'error') {
              setError(event.message ?? '未知错误')
              setStatus('error')
            }
          },
          onerror(err) {
            // 不 throw 的话库会自动重试；一次性调查不需要重试，直接把错误抛出去终止
            throw err
          },
        })
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
        setStatus('error')
      }
    },
    [status],
  )

  return { status, plan, completedCount, statusMessage, report, error, investigate }
}
