import json
import voyageai
from pymilvus import MilvusClient, DataType
from app.config import settings

COLLECTION = "oncall_kb"
VECTOR_DIM = 512  # voyage-3-lite outputs 512 dimensions

voyage = voyageai.Client(api_key=settings.voyage_api_key)
db = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")

schema = db.create_schema()                 
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)                             
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)                                 
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("metadata", DataType.VARCHAR, max_length=65535) 

index_params = db.prepare_index_params()                                                          
index_params.add_index("vector", metric_type="COSINE")

if db.has_collection(COLLECTION):
    db.drop_collection(COLLECTION)

db.create_collection(
    collection_name=COLLECTION,        
    schema=schema,
    index_params=index_params,
)
print(f"Created collection: {COLLECTION}")

# --- Step 1: Read and chunk the document ---
with open("docs/high_cpu.md") as f:
    text = f.read()

chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
print(f"\nDocument split into {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"  [{i}] {chunk[:60]}...")

# --- Step 2: Embed the chunks ---
result = voyage.embed(chunks, model="voyage-3-lite", input_type="document")
vectors = result.embeddings
print(f"\nGenerated {len(vectors)} vectors of dimension {len(vectors[0])}")

# --- Step 3: Insert into Milvus ---
rows = [
    {
        "vector": vectors[i],
        "content": chunks[i],
        "metadata": json.dumps({"source": "high_cpu.md"}),
    }
    for i in range(len(chunks))
]

db.insert(collection_name=COLLECTION, data=rows)
print(f"\nInserted {len(rows)} rows into Milvus")

# --- Step 4: Search ---
query = "how do I find which process is using too much CPU?"
query_vector = voyage.embed([query], model="voyage-3-lite", input_type="query").embeddings[0]

db.load_collection(COLLECTION)

results = db.search(
    collection_name=COLLECTION,
    data=[query_vector],
    limit=2,
    output_fields=["content", "metadata"],
)

print(f"\nQuery: '{query}'")
print("\nTop results:")
for hit in results[0]:
    print(f"\n  Score: {hit['distance']:.4f}")
    print(f"  Content: {hit['entity']['content'][:200]}")
