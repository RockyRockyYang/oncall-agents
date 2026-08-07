import { v4 as uuidv4 } from 'uuid'

const ACTIVE_KEY = 'oncall_session_id'
const HISTORY_KEY = 'oncall_session_history'

export interface SessionSummary {
  id: string
  title: string
}

export function getActiveSessionId(): string {
  let id = localStorage.getItem(ACTIVE_KEY)
  if (!id) {
    id = uuidv4()
    localStorage.setItem(ACTIVE_KEY, id)
  }
  return id
}

export function setActiveSessionId(id: string): void {
  localStorage.setItem(ACTIVE_KEY, id)
}

export function createSessionId(): string {
  const id = uuidv4()
  setActiveSessionId(id)
  return id
}

export function getSessionHistory(): SessionSummary[] {
  const raw = localStorage.getItem(HISTORY_KEY)
  return raw ? JSON.parse(raw) : []
}

// 只在这个 session 第一次出现时插入一条记录，之后同一 session 里再调用都是无操作
export function addSessionToHistory(id: string, title: string): SessionSummary[] {
  const list = getSessionHistory()
  if (list.some((s) => s.id === id)) return list
  const next = [{ id, title: title.slice(0, 30) }, ...list]
  localStorage.setItem(HISTORY_KEY, JSON.stringify(next))
  return next
}
