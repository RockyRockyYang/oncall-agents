import json
import voyageai
from loguru import logger
from pymilvus import MilvusClient, DataType
from app.config import settings

COLLECTION = "oncall_kb"
VECTOR_DIM = 512  # voyage-3-lite outputs 512 dimensions


class VectorStoreService:
    def __init__(self, collection: str = COLLECTION):
        self.collection = collection
        self.db = MilvusClient(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
        )
        self.voyage = voyageai.Client(api_key=settings.voyage_api_key)  # type: ignore[attr-defined]
        logger.info("VectorStoreService initialized | collection={}", self.collection)
        self._ensure_collection()

    def _ensure_collection(self):
        if self.db.has_collection(self.collection):
            logger.debug("Collection already exists | collection={}", self.collection)
            return

        schema = self.db.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("metadata", DataType.VARCHAR, max_length=65535)

        index_params = self.db.prepare_index_params()
        index_params.add_index("vector", metric_type="COSINE")

        self.db.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Collection created | collection={}", self.collection)

    def drop_collection(self):
        self.db.drop_collection(self.collection)
        logger.info("Collection dropped | collection={}", self.collection)

    def ingest(self, chunks: list[str], source: str):
        logger.info("Ingesting chunks | source={} count={}", source, len(chunks))
        result = self.voyage.embed(chunks, model="voyage-3-lite", input_type="document")

        rows = [
            {
                "vector": result.embeddings[i],
                "content": chunks[i],
                "metadata": json.dumps({"source": source}),
            }
            for i in range(len(chunks))
        ]
        self.db.insert(collection_name=self.collection, data=rows)
        self.db.flush(self.collection)
        logger.info("Ingestion complete | source={} count={}", source, len(chunks))

    def search(self, query: str, top_k: int = 3) -> list[str]:
        logger.debug("Searching | query={!r} top_k={}", query, top_k)
        self.db.load_collection(self.collection)
        query_vector = self.voyage.embed(
            [query], model="voyage-3-lite", input_type="query"
        ).embeddings[0]

        results = self.db.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=top_k,
            output_fields=["content"],
        )
        chunks = [hit["content"] for hit in results[0]]
        logger.debug("Search returned {} results", len(chunks))
        return chunks
