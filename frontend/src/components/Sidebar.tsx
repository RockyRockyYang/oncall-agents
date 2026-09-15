import { useState } from 'react'
import { Drawer, Box, Avatar, Typography, List, ListItemButton, ListItemIcon, ListItemText, Divider, IconButton, Menu, MenuItem } from '@mui/material'
import EditNoteIcon from '@mui/icons-material/EditNote'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutlined'
import MenuBookIcon from '@mui/icons-material/MenuBook'
import MoreVertIcon from '@mui/icons-material/MoreVert'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutlined'
import type { SessionSummary } from '../lib/session'

const DRAWER_WIDTH = 260

interface SidebarProps {
  history: SessionSummary[]
  activeSessionId: string
  onNewChat: () => void
  onSelectSession: (id: string) => void
  onDeleteSession: (id: string) => void
  onOpenKnowledgeBase: () => void
}

function Sidebar({ history, activeSessionId, onNewChat, onSelectSession, onDeleteSession, onOpenKnowledgeBase }: SidebarProps) {
  // 记录当前打开的是哪一条历史记录的菜单，而不是每条各自维护一个 anchorEl ——
  // 同一时间只会有一个菜单打开，一个 state 就够了
  const [menuAnchor, setMenuAnchor] = useState<HTMLElement | null>(null)
  const [menuSessionId, setMenuSessionId] = useState<string | null>(null)

  const openMenu = (e: React.MouseEvent<HTMLElement>, id: string) => {
    e.stopPropagation() // 防止顺带触发 ListItemButton 的 onSelectSession
    setMenuAnchor(e.currentTarget)
    setMenuSessionId(id)
  }

  const closeMenu = () => {
    setMenuAnchor(null)
    setMenuSessionId(null)
  }

  const handleDelete = () => {
    if (menuSessionId && window.confirm('删除这条历史对话？此操作无法撤销。')) {
      onDeleteSession(menuSessionId)
    }
    closeMenu()
  }

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
                '&:hover .session-menu-btn': { opacity: 1 },
              }}
            >
              <ListItemIcon sx={{ minWidth: 32 }}>
                <ChatBubbleOutlineIcon sx={{ fontSize: 18, color: '#8e8ea0' }} />
              </ListItemIcon>
              <ListItemText primary={item.title} slotProps={{ primary: { noWrap: true, sx: { fontSize: 14 } } }} />
              <IconButton
                size="small"
                className="session-menu-btn"
                onClick={(e) => openMenu(e, item.id)}
                sx={{
                  opacity: 0,
                  ml: 0.5,
                  flexShrink: 0,
                  '&:hover': { bgcolor: '#e2e2e2' },
                }}
              >
                <MoreVertIcon sx={{ fontSize: 18 }} />
              </IconButton>
            </ListItemButton>
          ))}
        </List>
      </Box>

      <Menu anchorEl={menuAnchor} open={Boolean(menuAnchor)} onClose={closeMenu}>
        <MenuItem onClick={handleDelete} sx={{ color: '#d32f2f', fontSize: 14, gap: 1 }}>
          <DeleteOutlineIcon sx={{ fontSize: 18 }} />
          删除
        </MenuItem>
      </Menu>

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
