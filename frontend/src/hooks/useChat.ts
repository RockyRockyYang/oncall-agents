import { useState, useCallback, useRef } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { fetchEventSource } from '@microsoft/fetch-event-source'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface ChatEvent {
  type: 'content' | 'tool_call' | 'done' | 'error'
  data?: unknown
}

const SESSION_KEY = 'oncall_session_id'

function getSessionId(): string {
  let id = localStorage.getItem(SESSION_KEY)
  if (!id) {
    id = uuidv4()
    localStorage.setItem(SESSION_KEY, id)
  }
  return id
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [sessionId] = useState(getSessionId)
  const abortRef = useRef<AbortController | null>(null)

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isStreaming) return

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
        appendToLast(`\n\n⚠️ 连接中断: ${err instanceof Error ? err.message : String(err)}`)
      } finally {
        setIsStreaming(false)
      }
    },
    [isStreaming, sessionId],
  )

  return { messages, sendMessage, isStreaming }
}
