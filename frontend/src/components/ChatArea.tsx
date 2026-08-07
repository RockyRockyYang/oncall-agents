import { useState, useRef, useEffect, type KeyboardEvent } from 'react'
import { Box, Toolbar, Typography, Button, Avatar, TextField, IconButton, CircularProgress } from '@mui/material'
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward'
import SmartToyOutlinedIcon from '@mui/icons-material/SmartToyOutlined'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useChat } from '../hooks/useChat'

interface ChatAreaProps {
  onOpenAIOps: () => void
}

function ChatArea({ onOpenAIOps }: ChatAreaProps) {
  const { messages, sendMessage, isStreaming } = useChat()
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    if (!input.trim() || isStreaming) return
    sendMessage(input)
    setInput('')
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, bgcolor: '#fff' }}>
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        <Typography sx={{ fontWeight: 600, color: '#333' }}>Oncall Agent</Typography>
        <Button
          variant="outlined"
          onClick={onOpenAIOps}
          sx={{ textTransform: 'none', borderRadius: '10px', borderColor: '#e5e5e5', color: '#333' }}
        >
          AI Ops
        </Button>
      </Toolbar>

      <Box sx={{ flex: 1, overflowY: 'auto', px: 2 }}>
        <Box sx={{ maxWidth: 720, mx: 'auto', display: 'flex', flexDirection: 'column', gap: 3, py: 3 }}>
          {messages.map((m, i) =>
            m.role === 'user' ? (
              <Box
                key={i}
                sx={{ alignSelf: 'flex-end', maxWidth: '70%', bgcolor: '#f4f4f4', borderRadius: '18px', px: 2, py: 1.25 }}
              >
                {m.content}
              </Box>
            ) : (
              <Box key={i} sx={{ display: 'flex', gap: 1.5 }}>
                <Avatar sx={{ bgcolor: '#000', width: 30, height: 30, flexShrink: 0 }}>
                  <SmartToyOutlinedIcon sx={{ fontSize: 18 }} />
                </Avatar>
                {m.content ? (
                  <Box
                    sx={{
                      pt: 0.5,
                      lineHeight: 1.6,
                      minWidth: 0,
                      '& p': { m: 0, mb: 1 },
                      '& p:last-child': { mb: 0 },
                      '& table': { borderCollapse: 'collapse', my: 1.5, fontSize: 14 },
                      '& th, & td': { border: '1px solid #e5e5e5', px: 1.25, py: 0.5, textAlign: 'left' },
                      '& th': { bgcolor: '#f7f7f7' },
                    }}
                  >
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                  </Box>
                ) : (
                  <Box sx={{ pt: 1 }}>
                    <CircularProgress size={14} sx={{ color: '#8e8ea0' }} />
                  </Box>
                )}
              </Box>
            ),
          )}
          <div ref={bottomRef} />
        </Box>
      </Box>

      <Box sx={{ px: 2, pb: 3, pt: 1 }}>
        <Box
          sx={{
            maxWidth: 720,
            mx: 'auto',
            display: 'flex',
            alignItems: 'flex-end',
            gap: 1,
            border: '1px solid #e5e5e5',
            borderRadius: '28px',
            px: 2,
            py: 1,
            boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
          }}
        >
          <TextField
            fullWidth
            multiline
            maxRows={6}
            placeholder="Message Oncall Agent…"
            variant="standard"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            slotProps={{ input: { disableUnderline: true } }}
          />
          <IconButton
            onClick={handleSend}
            disabled={isStreaming || !input.trim()}
            sx={{ bgcolor: '#000', color: '#fff', '&:hover': { bgcolor: '#333' }, '&.Mui-disabled': { bgcolor: '#e5e5e5' } }}
          >
            <ArrowUpwardIcon fontSize="small" />
          </IconButton>
        </Box>
      </Box>
    </Box>
  )
}

export default ChatArea
