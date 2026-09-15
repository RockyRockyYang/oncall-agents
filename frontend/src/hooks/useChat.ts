import { useState, useCallback, useRef, useEffect } from 'react'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import {
  getActiveSessionId,
  setActiveSessionId,
  createSessionId,
  fetchSessionHistory,
  removeSession,
  type SessionSummary,
} from '../lib/session'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface ChatEvent {
  type: 'content' | 'tool_call' | 'done' | 'error'
  data?: unknown
}

export function useChat() {
  const [sessionId, setSessionId] = useState(getActiveSessionId)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [history, setHistory] = useState<SessionSummary[]>([])
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    fetchSessionHistory().then(setHistory)
  }, [])

  const sendMessage = useCallback(
    async (text: string): Promise<boolean> => {
      if (!text.trim() || isStreaming) return false

      // 用户消息 + 一条待填充的空 assistant 消息，一起加入历史
      setMessages((prev) => [...prev, { role: 'user', content: text }, { role: 'assistant', content: '' }])
      setIsStreaming(true)

      const appendToLast = (chunk: string) => {
        setMessages((prev) => {
          const next = [...prev]
          const last = next[next.length - 1]
          next[next.length - 1] = { ...last, content: last.content + chunk }
          return next
        })
      }

      const controller = new AbortController()
      abortRef.current = controller
      let ok = true

      try {
        await fetchEventSource('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, message: text }),
          signal: controller.signal,
          openWhenHidden: true, // tab 切到后台时不要断流
          onmessage(ev) {
            // 防御性解析：中间层（代理/心跳帧）可能混入空帧或非 JSON 数据，
            // 单帧解析失败不应该打断整条流。
            if (!ev.data) return
            let event: ChatEvent
            try {
              event = JSON.parse(ev.data)
            } catch {
              console.warn('Skipping malformed SSE frame:', ev.data)
              return
            }
            if (event.type === 'content') {
              appendToLast(String(event.data))
            } else if (event.type === 'error') {
              appendToLast(`\n\n⚠️ ${String(event.data)}`)
            }
            // done：不用特殊处理，流结束后下面的 await 会自然返回
            // tool_call：Step 33 再处理，这里先忽略
          },
          onerror(err) {
            // 不 throw 的话库会自动重试；一次性对话不需要重试，直接把错误抛出去终止
            throw err
          },
        })
      } catch (err) {
        console.error('Chat request failed:', err)
        ok = false
        appendToLast('\n\n⚠️ 连接后端失败，请确认服务已启动后重试')
      } finally {
        setIsStreaming(false)
        fetchSessionHistory().then(setHistory)
      }
      return ok
    },
    [isStreaming, sessionId],
  )

  const newChat = useCallback(() => {
    setSessionId(createSessionId())
    setMessages([])
  }, [])

  const loadSession = useCallback(
    async (id: string) => {
      if (id === sessionId) return
      setActiveSessionId(id)
      setSessionId(id)
      setMessages([])
      const res = await fetch(`/api/chat/session/${id}`)
      const data = await res.json()
      setMessages(data.messages)
    },
    [sessionId],
  )

  const deleteSession = useCallback(
    async (id: string) => {
      await removeSession(id)
      if (id === sessionId) {
        newChat()
      }
      fetchSessionHistory().then(setHistory)
    },
    [sessionId, newChat],
  )

  return { messages, sendMessage, isStreaming, sessionId, history, newChat, loadSession, deleteSession }
}
