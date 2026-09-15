import { v4 as uuidv4 } from 'uuid'

const ACTIVE_KEY = 'oncall_session_id'

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

export async function fetchSessionHistory(): Promise<SessionSummary[]> {
  const res = await fetch('/api/sessions')
  const data = await res.json()
  return data.sessions
}
