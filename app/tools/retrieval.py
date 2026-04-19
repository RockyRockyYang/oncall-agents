from langchain_core.tools import tool
from app.services.vector_store import VectorStoreService

_svc = VectorStoreService()


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the on-call knowledge base for relevant information.
    """
    chunks = _svc.search(query)
    return "\n\n---\n\n".join(chunks) if chunks else "No relevant content found."
