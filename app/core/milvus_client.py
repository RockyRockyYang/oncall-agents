from pymilvus import MilvusClient as _MilvusClient
from app.config import settings

_client: _MilvusClient | None = None

def get_milVusClient() -> _MilvusClient:
    global _client
    if _client is None:
        _client = _MilvusClient(
            host=settings.milvus_host, 
            port=settings.milvus_port
        )
    return _client