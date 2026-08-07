import { useState } from 'react'
import { Box } from '@mui/material'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import AIOpsPanel from './components/AIOpsPanel'

function App() {
  const [isAIOpsOpen, setIsAIOpsOpen] = useState(false)

  return (
    <Box sx={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden' }}>
      <Sidebar />
      <ChatArea onOpenAIOps={() => setIsAIOpsOpen(true)} />
      <AIOpsPanel isOpen={isAIOpsOpen} onClose={() => setIsAIOpsOpen(false)} />
    </Box>
  )
}

export default App
