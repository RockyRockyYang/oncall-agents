from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.vector_store import VectorStoreService

router = APIRouter()
_svc = VectorStoreService()

ALLOWED_EXTENSIONS = {"txt", "md", "markdown"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB，跟前端校验保持一致


class UploadResponse(BaseModel):
    source: str
    chunks_inserted: int


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile) -> UploadResponse:
    source = file.filename or "unknown"
    ext = source.rsplit(".", 1)[-1].lower() if "." in source else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type, allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 50MB).")

    text = content.decode("utf-8")
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    if not chunks:
        raise HTTPException(status_code=400, detail="No content to ingest.")
    _svc.ingest(chunks, source=source)
    return UploadResponse(source=source, chunks_inserted=len(chunks))
