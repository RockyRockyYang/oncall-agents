import { Drawer, Box, Avatar, Typography, List, ListItemButton, ListItemIcon, ListItemText } from '@mui/material'
import EditNoteIcon from '@mui/icons-material/EditNote'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutlined'

const DRAWER_WIDTH = 260

function Sidebar() {
  const history = ['CPU 告警排查', '数据库连接池耗尽', '如何回滚部署']
  const activeIndex = 0

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: DRAWER_WIDTH,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: DRAWER_WIDTH,
          boxSizing: 'border-box',
          bgcolor: '#fff',
          borderRight: '1px solid #ececec',
          p: 1.5,
        },
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 0.5, mb: 2 }}>
        <Avatar sx={{ width: 32, height: 32, background: 'linear-gradient(135deg, #14b8a6, #22c55e)' }}>
          <AutoAwesomeIcon sx={{ fontSize: 18 }} />
        </Avatar>
        <Typography sx={{ fontWeight: 700, color: '#1a1a1a' }}>Oncall Agent</Typography>
      </Box>

      <ListItemButton
        sx={{
          flexGrow: 0,
          borderRadius: '8px',
          color: '#1a1a1a',
          mb: 1.5,
          '&:hover': { bgcolor: '#f1f1f1' },
        }}
      >
        <ListItemIcon sx={{ minWidth: 32 }}>
          <EditNoteIcon sx={{ color: '#1a1a1a' }} />
        </ListItemIcon>
        <ListItemText primary="新对话" />
      </ListItemButton>

      <Typography variant="caption" sx={{ color: '#8e8ea0', px: 1, mb: 0.5, display: 'block' }}>
        历史对话
      </Typography>

      <List sx={{ overflowY: 'auto' }}>
        {history.map((item, i) => (
          <ListItemButton
            key={i}
            selected={i === activeIndex}
            sx={{
              borderRadius: '8px',
              color: '#1a1a1a',
              py: 0.75,
              '&:hover': { bgcolor: '#f1f1f1' },
              '&.Mui-selected': { bgcolor: '#ececec' },
              '&.Mui-selected:hover': { bgcolor: '#ececec' },
            }}
          >
            <ListItemIcon sx={{ minWidth: 32 }}>
              <ChatBubbleOutlineIcon sx={{ fontSize: 18, color: '#8e8ea0' }} />
            </ListItemIcon>
            <ListItemText primary={item} slotProps={{ primary: { noWrap: true, sx: { fontSize: 14 } } }} />
          </ListItemButton>
        ))}
      </List>
    </Drawer>
  )
}

export default Sidebar
