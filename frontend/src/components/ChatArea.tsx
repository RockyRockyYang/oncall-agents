import { Box, Toolbar, Typography, Button, Avatar, TextField, IconButton } from '@mui/material'
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward'
import SmartToyOutlinedIcon from '@mui/icons-material/SmartToyOutlined'

interface ChatAreaProps {
  onOpenAIOps: () => void
}

function ChatArea({ onOpenAIOps }: ChatAreaProps) {
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
          <Box sx={{ alignSelf: 'flex-end', maxWidth: '70%', bgcolor: '#f4f4f4', borderRadius: '18px', px: 2, py: 1.25 }}>
            这个服务 CPU 一直 100%，帮我看看
          </Box>

          <Box sx={{ display: 'flex', gap: 1.5 }}>
            <Avatar sx={{ bgcolor: '#000', width: 30, height: 30 }}>
              <SmartToyOutlinedIcon sx={{ fontSize: 18 }} />
            </Avatar>
            <Box sx={{ pt: 0.5, lineHeight: 1.6 }}>好的，我先查一下最近的 runbook…</Box>
          </Box>
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
            slotProps={{ input: { disableUnderline: true } }}
          />
          <IconButton sx={{ bgcolor: '#000', color: '#fff', '&:hover': { bgcolor: '#333' } }}>
            <ArrowUpwardIcon fontSize="small" />
          </IconButton>
        </Box>
      </Box>
    </Box>
  )
}

export default ChatArea
