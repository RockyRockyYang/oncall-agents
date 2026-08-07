import { Drawer, Box, Avatar, Typography, List, ListItemButton, ListItemIcon, ListItemText, Divider } from '@mui/material'
import EditNoteIcon from '@mui/icons-material/EditNote'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutlined'
import MenuBookIcon from '@mui/icons-material/MenuBook'
import type { SessionSummary } from '../lib/session'

const DRAWER_WIDTH = 260

interface SidebarProps {
  history: SessionSummary[]
  activeSessionId: string
  onNewChat: () => void
  onSelectSession: (id: string) => void
  onOpenKnowledgeBase: () => void
}

function Sidebar({ history, activeSessionId, onNewChat, onSelectSession, onOpenKnowledgeBase }: SidebarProps) {
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
        <Typography sx={{ fontWeight: 700, color: '#1a1a1a' }}>智能 Oncall 助手</Typography>
      </Box>

      <ListItemButton
        onClick={onNewChat}
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

      {/* 会话相关的可滚动区域，flex:1 把下面的"知识库管理"入口挤到底部，
          跟这些跟具体 session 相关的操作区分开——知识库是全局操作，不属于任何一个对话 */}
      <Box sx={{ flex: 1, overflowY: 'auto' }}>
        <Typography variant="caption" sx={{ color: '#8e8ea0', px: 1, mb: 0.5, display: 'block' }}>
          历史对话
        </Typography>

        <List>
          {history.map((item) => (
            <ListItemButton
              key={item.id}
              selected={item.id === activeSessionId}
              onClick={() => onSelectSession(item.id)}
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
              <ListItemText primary={item.title} slotProps={{ primary: { noWrap: true, sx: { fontSize: 14 } } }} />
            </ListItemButton>
          ))}
        </List>
      </Box>

      <Divider sx={{ my: 1 }} />

      <ListItemButton
        onClick={onOpenKnowledgeBase}
        sx={{
          flexGrow: 0,
          borderRadius: '8px',
          color: '#1a1a1a',
          '&:hover': { bgcolor: '#f1f1f1' },
        }}
      >
        <ListItemIcon sx={{ minWidth: 32 }}>
          <MenuBookIcon sx={{ fontSize: 18, color: '#8e8ea0' }} />
        </ListItemIcon>
        <ListItemText primary="知识库管理" slotProps={{ primary: { sx: { fontSize: 14 } } }} />
      </ListItemButton>
    </Drawer>
  )
}

export default Sidebar
