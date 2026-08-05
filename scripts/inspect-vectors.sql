-- 看有哪些集合（collection）

SELECT *
FROM langchain_pg_collection;

-- 看存了多少条向量数据

SELECT collection_id,
       COUNT(*) as chunk_count
FROM langchain_pg_embedding
GROUP BY collection_id;

-- 看具体存了什么内容（前 5 条）

SELECT e.document,
       e.cmetadata,
       LEFT(e.embedding::text, 60) AS vector_preview
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = 'oncall_kb'
LIMIT 5;

-- 按 source 分组，看每个文档存了多少 chunk

SELECT e.cmetadata->>'source' AS source,
       COUNT(*) AS chunks
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = 'oncall_kb'
GROUP BY e.cmetadata->>'source';