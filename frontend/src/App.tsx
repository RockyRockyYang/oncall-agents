import { useState } from 'react'
import { Box } from '@mui/material'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import AIOpsPanel from './components/AIOpsPanel'
import KnowledgeBaseDialog from './components/KnowledgeBaseDialog'
import { useChat } from './hooks/useChat'

function App() {
  const [isAIOpsOpen, setIsAIOpsOpen] = useState(false)
  const [isKbOpen, setIsKbOpen] = useState(false)
  const chat = useChat()

  return (
    <Box sx={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden' }}>
      <Sidebar
        history={chat.history}
        activeSessionId={chat.sessionId}
        onNewChat={chat.newChat}
        onSelectSession={chat.loadSession}
        onOpenKnowledgeBase={() => setIsKbOpen(true)}
      />
      <ChatArea chat={chat} onOpenAIOps={() => setIsAIOpsOpen(true)} />
      <AIOpsPanel isOpen={isAIOpsOpen} onClose={() => setIsAIOpsOpen(false)} />
      <KnowledgeBaseDialog open={isKbOpen} onClose={() => setIsKbOpen(false)} />
    </Box>
  )
}

export default App
