from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.vector_store import VectorStoreService

router = APIRouter()
_svc = VectorStoreService()


class UploadResponse(BaseModel):
    source: str
    chunks_inserted: int


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile) -> UploadResponse:
    content = await file.read()
    text = content.decode("utf-8")
    source = file.filename or "unknown"
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    if not chunks:
        raise HTTPException(status_code=400, detail="No content to ingest.")
    _svc.ingest(chunks, source=source)
    return UploadResponse(source=source, chunks_inserted=len(chunks))
