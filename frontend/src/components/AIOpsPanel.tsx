import { Drawer, Box, Typography, IconButton, TextField, Button } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'

interface AIOpsPanelProps {
  isOpen: boolean
  onClose: () => void
}

function AIOpsPanel({ isOpen, onClose }: AIOpsPanelProps) {
  return (
    <Drawer anchor="right" open={isOpen} onClose={onClose}>
      <Box sx={{ width: 360, p: 2.5, display: 'flex', flexDirection: 'column', gap: 1.5 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography sx={{ fontWeight: 600 }}>AI Ops Investigation</Typography>
          <IconButton onClick={onClose} size="small">
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
        <TextField
          multiline
          rows={4}
          placeholder="Paste alert text here…"
          sx={{ '& .MuiOutlinedInput-root': { borderRadius: '12px' } }}
        />
        <Button
          variant="contained"
          disableElevation
          sx={{ bgcolor: '#000', textTransform: 'none', borderRadius: '10px', '&:hover': { bgcolor: '#333' } }}
        >
          Investigate
        </Button>
      </Box>
    </Drawer>
  )
}

export default AIOpsPanel
