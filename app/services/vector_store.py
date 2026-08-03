from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from loguru import logger

from app.config import settings

COLLECTION = "oncall_kb"


class VectorStoreService:
    def __init__(self, collection: str = COLLECTION):
        self.collection = collection
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )
        self.store = PGVector(
            embeddings=embeddings,
            collection_name=collection,
            connection=settings.database_url,
            use_jsonb=True,
        )
        logger.info("VectorStoreService initialized | collection={}", self.collection)

    def drop_collection(self):
        self.store.delete_collection()
        logger.info("Collection dropped | collection={}", self.collection)

    def ingest(self, chunks: list[str], source: str):
        logger.info("Ingesting chunks | source={} count={}", source, len(chunks))
        docs = [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]
        self.store.add_documents(docs)
        logger.info("Ingestion complete | source={} count={}", source, len(chunks))

    def search(self, query: str, top_k: int = 3) -> list[str]:
        logger.debug("Searching | query={!r} top_k={}", query, top_k)
        results = self.store.similarity_search(query, k=top_k)
        chunks = [doc.page_content for doc in results]
        logger.debug("Search returned {} results", len(chunks))
        return chunks


vector_store_service = VectorStoreService()
