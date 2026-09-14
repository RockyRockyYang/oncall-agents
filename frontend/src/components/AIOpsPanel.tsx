import { useState } from 'react'
import {
  Drawer,
  Box,
  Typography,
  IconButton,
  TextField,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  CircularProgress,
  Alert,
  Divider,
} from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import RadioButtonUncheckedIcon from '@mui/icons-material/RadioButtonUnchecked'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useAIOps } from '../hooks/useAIOps'

interface AIOpsPanelProps {
  isOpen: boolean
  onClose: () => void
}

function AIOpsPanel({ isOpen, onClose }: AIOpsPanelProps) {
  const { status, plan, completedCount, statusMessage, report, error, investigate } = useAIOps()
  const [alertText, setAlertText] = useState('')

  const isRunning = status === 'running'

  return (
    <Drawer anchor="right" open={isOpen} onClose={onClose}>
      <Box
        sx={{
          width: { xs: '100vw', sm: 420 },
          maxWidth: '100vw',
          p: 2.5,
          display: 'flex',
          flexDirection: 'column',
          gap: 1.5,
          height: '100%',
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography sx={{ fontWeight: 600 }}>故障调查</Typography>
          <IconButton onClick={onClose} size="small">
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>

        <TextField
          multiline
          rows={4}
          placeholder="粘贴告警内容…"
          value={alertText}
          onChange={(e) => setAlertText(e.target.value)}
          disabled={isRunning}
          sx={{ '& .MuiOutlinedInput-root': { borderRadius: '12px' } }}
        />
        <Button
          variant="contained"
          disableElevation
          onClick={() => investigate(alertText)}
          disabled={isRunning || !alertText.trim()}
          sx={{ bgcolor: '#000', textTransform: 'none', borderRadius: '10px', '&:hover': { bgcolor: '#333' } }}
        >
          {isRunning ? '调查中…' : '开始调查'}
        </Button>

        {(plan.length > 0 || error) && <Divider />}

        <Box sx={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          {plan.length > 0 && (
            <List dense disablePadding>
              {plan.map((step, i) => (
                <ListItem key={i} disableGutters>
                  <ListItemIcon sx={{ minWidth: 32 }}>
                    {i < completedCount ? (
                      <CheckCircleIcon fontSize="small" color="success" />
                    ) : (
                      <RadioButtonUncheckedIcon fontSize="small" sx={{ color: '#c4c4c4' }} />
                    )}
                  </ListItemIcon>
                  <ListItemText primary={step} slotProps={{ primary: { sx: { fontSize: 14 } } }} />
                </ListItem>
              ))}
            </List>
          )}

          {isRunning && statusMessage && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={14} sx={{ color: '#8e8ea0' }} />
              <Typography variant="caption" sx={{ color: '#8e8ea0' }}>
                {statusMessage}
              </Typography>
            </Box>
          )}

          {report && (
            <Box
              sx={{
                fontSize: 14,
                lineHeight: 1.6,
                '& p': { m: 0, mb: 1 },
                '& p:last-child': { mb: 0 },
                '& table': { borderCollapse: 'collapse', my: 1.5, fontSize: 13 },
                '& th, & td': { border: '1px solid #e5e5e5', px: 1, py: 0.5, textAlign: 'left' },
                '& th': { bgcolor: '#f7f7f7' },
              }}
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
            </Box>
          )}

          {status === 'error' && <Alert severity="error">{error}</Alert>}
        </Box>
      </Box>
    </Drawer>
  )
}

export default AIOpsPanel
