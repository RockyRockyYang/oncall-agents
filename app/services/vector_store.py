import json
import voyageai
from pymilvus import MilvusClient, DataType
from app.config import settings

COLLECTION = "oncall_kb"
VECTOR_DIM = 512  # voyage-3-lite outputs 512 dimensions


class VectorStoreService:
    def __init__(self):
        self.db = MilvusClient(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
        )
        self.voyage = voyageai.Client(api_key=settings.voyage_api_key)
        self._ensure_collection()

    def _ensure_collection(self):
        if self.db.has_collection(COLLECTION):
            return

        schema = self.db.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("metadata", DataType.VARCHAR, max_length=65535)

        index_params = self.db.prepare_index_params()
        index_params.add_index("vector", metric_type="COSINE")

        self.db.create_collection(
            collection_name=COLLECTION,
            schema=schema,
            index_params=index_params,
        )

    def ingest(self, chunks: list[str], source: str):
        result = self.voyage.embed(chunks, model="voyage-3-lite", input_type="document")

        rows = [
            {
                "vector": result.embeddings[i],
                "content": chunks[i],
                "metadata": json.dumps({"source": source}),
            }
            for i in range(len(chunks))
        ]
        self.db.insert(collection_name=COLLECTION, data=rows)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        self.db.load_collection(COLLECTION)
        query_vector = self.voyage.embed(
            [query], model="voyage-3-lite", input_type="query"
        ).embeddings[0]

        results = self.db.search(
            collection_name=COLLECTION,
            data=[query_vector],
            limit=top_k,
            output_fields=["content"],
        )
        return [hit["content"] for hit in results[0]]
