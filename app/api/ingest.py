from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.vector_store import VectorStoreService

router = APIRouter()
_svc = VectorStoreService()


class IngestRequest(BaseModel):
    source: str
    text: str


class IngestResponse(BaseModel):
    source: str
    chunks_inserted: int


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    chunks = [c.strip() for c in req.text.split("\n\n") if c.strip()]
    if not chunks:
        raise HTTPException(status_code=400, detail="No content to ingest.")
    _svc.ingest(chunks, source=req.source)
    return IngestResponse(source=req.source, chunks_inserted=len(chunks))


# testing