import { useRef, useState, type ChangeEvent } from 'react'
import { Dialog, DialogTitle, DialogContent, DialogActions, Button, Typography, CircularProgress, Alert } from '@mui/material'
import UploadFileIcon from '@mui/icons-material/UploadFile'

const ALLOWED_EXTENSIONS = ['.txt', '.md', '.markdown']
const MAX_FILE_SIZE = 50 * 1024 * 1024 // 50MB

interface KnowledgeBaseDialogProps {
  open: boolean
  onClose: () => void
}

function KnowledgeBaseDialog({ open, onClose }: KnowledgeBaseDialogProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<{ ok: boolean; message: string } | null>(null)

  const isAllowedFile = (name: string) => ALLOWED_EXTENSIONS.some((ext) => name.toLowerCase().endsWith(ext))

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = '' // 允许重复选择同一个文件
    if (!file) return

    if (!isAllowedFile(file.name)) {
      setResult({ ok: false, message: `只支持上传 ${ALLOWED_EXTENSIONS.join(' / ')} 格式的文件` })
      return
    }
    if (file.size > MAX_FILE_SIZE) {
      setResult({ ok: false, message: '文件大小不能超过 50MB' })
      return
    }

    setUploading(true)
    setResult(null)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail ?? '上传失败')
      setResult({ ok: true, message: `${data.source} 上传成功，已生成 ${data.chunks_inserted} 个知识片段` })
    } catch (err) {
      setResult({ ok: false, message: err instanceof Error ? err.message : '上传失败' })
    } finally {
      setUploading(false)
    }
  }

  const handleClose = () => {
    setResult(null)
    onClose()
  }

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="xs" fullWidth>
      <DialogTitle sx={{ fontWeight: 600 }}>知识库管理</DialogTitle>
      <DialogContent>
        <Typography variant="body2" sx={{ color: '#8e8ea0', mb: 2 }}>
          上传 runbook / 故障处理文档，之后所有对话都能检索到这份知识。支持 {ALLOWED_EXTENSIONS.join(' / ')}，最大 50MB。
        </Typography>

        <input
          type="file"
          accept={ALLOWED_EXTENSIONS.join(',')}
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
        <Button
          fullWidth
          variant="outlined"
          startIcon={uploading ? <CircularProgress size={16} /> : <UploadFileIcon />}
          disabled={uploading}
          onClick={() => fileInputRef.current?.click()}
          sx={{ textTransform: 'none', borderRadius: '10px', borderColor: '#e5e5e5', color: '#333', py: 1.25 }}
        >
          {uploading ? '上传中…' : '选择文件上传'}
        </Button>

        {result && (
          <Alert severity={result.ok ? 'success' : 'error'} sx={{ mt: 2 }}>
            {result.message}
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose} sx={{ textTransform: 'none' }}>
          关闭
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default KnowledgeBaseDialog
