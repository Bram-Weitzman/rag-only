from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import uuid4
import os
import math
import httpx
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue


# --- env ---
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # optional
COLLECTION = os.environ.get("QDRANT_COLLECTION", "isc2_toronto_v3")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
API_ENDPOINT = f"{OLLAMA_HOST}/api/embeddings"

# --- FastAPI ---
app = FastAPI(title="RAG-only API", version="1.0")
origins = [
    "http://10.20.10.3", # The origin of your web server
    "http://localhost",  # Optional: for local testing
    "http://localhost:8000", # Optional: for local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Qdrant client ---
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# --- helpers ---

# --- helpers ---
import logging # Use standard logging or loguru

async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Call Ollama embeddings endpoint once for a batch of texts."""
    # Note: For simplicity, this example sends the first text.
    # A full implementation might loop or use a different payload structure if
    # the API supports batching under a different key.
    payload = {"model": EMBED_MODEL, "prompt": texts[0]}

    async with httpx.AsyncClient(timeout=60) as http:
        try:
            r = await http.post(f"{OLLAMA_HOST}/api/embeddings", json=payload)
            r.raise_for_status() # Raise an exception for non-200 responses

            data = r.json()

            # --- DEBUGGING STEP ---
            # Log the actual response from Ollama to see its structure
            logging.info(f"Ollama response received: {data}")

            # --- THE FIX ---
            # Check for the correct singular 'embedding' key
            if "embedding" in data:
                # The function expects List[List[float]], so we wrap the result in a list
                return [data["embedding"]]
            else:
                # This is the line that was being incorrectly triggered
                raise HTTPException(502, f"Unexpected Ollama embeddings Response format: {data}")

        except httpx.RequestError as e:
            raise HTTPException(503, f"Could not connect to Ollama: {e}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(502, f"Ollama embed error: {e.response.text[:200]}")

#async def embed_texts(texts: List[str]) -> List[List[float]]:
#    """Call Ollama embeddings endpoint once for a batch of texts."""
    # Ollama embeddings API (2025): POST /api/embeddings {model, input}
#    payload = {"model": EMBED_MODEL, "input": texts}
#    async with httpx.AsyncClient(timeout=60) as http:
#        r = await http.post(f"{OLLAMA_HOST}/api/embeddings", json=payload)
#        if r.status_code != 200:
#            raise HTTPException(502, f"Ollama embed error: {r.text[:200]}")
#        data = r.json()
        # shape can be {"embeddings":[...]} or {"data":[{"embedding":[...]}...]} depending on version
#        if "embeddings" in data:
#            return data["embeddings"]
#        elif "data" in data:
#            return [d["embedding"] for d in data["data"]]
#        else:
#            raise HTTPException(502, "Unexpected Ollama embeddings response")

def ensure_collection(vector_size: int):
    exists = client.collection_exists(COLLECTION)
    if not exists:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        # optional: create payload index for filters
        client.create_payload_index(COLLECTION, field_name="doc_id", field_schema="keyword")
        client.create_payload_index(COLLECTION, field_name="source", field_schema="keyword")

def chunk_text(text: str, max_tokens: int = 400, overlap: int = 60) -> List[str]:
    # naive token-ish chunking by words; works fine for many cases
    words = text.split()
    if not words:
        return []
    step = max_tokens - overlap
    if step <= 0:
        step = max_tokens
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + max_tokens])
        if chunk:
            chunks.append(chunk)
    return chunks

# --- models ---
class IngestItem(BaseModel):
    text: str
    doc_id: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    chunk_size: int = 400
    chunk_overlap: int = 60

class IngestRequest(BaseModel):
    items: List[IngestItem]

class IngestResponse(BaseModel):
    inserted_points: int

class QueryRequest(BaseModel):
    query: str = Field(..., description="User's query")
    top_k: int = 5
    filter_by_source: Optional[str] = None
    filter_by_doc_id: Optional[str] = None

class RetrievedChunk(BaseModel):
    text: str
    score: float
    doc_id: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    chunks: List[RetrievedChunk]

# --- startup: infer vector size once via a tiny embed ---
@app.on_event("startup")
async def startup():
    vec = await embed_texts(["ping"])
    dim = len(vec[0])
    ensure_collection(dim)

# --- routes ---
@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    texts = []
    payloads = []
    ids = []
    # build chunks
    for item in req.items:
        base_doc_id = item.doc_id or str(uuid4())
        chunks = chunk_text(item.text, item.chunk_size, item.chunk_overlap)
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            payloads.append({
                "doc_id": base_doc_id,
                "source": item.source,
                "metadata": item.metadata or {},
                "chunk_ix": idx
            })
            ids.append(f"{base_doc_id}-{idx}")
    if not texts:
        return IngestResponse(inserted_points=0)

    # embed and upsert
    vectors = await embed_texts(texts)
    points = [
        PointStruct(id=ids[i], vector=vectors[i], payload={**payloads[i], "text": texts[i]})
        for i in range(len(texts))
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    return IngestResponse(inserted_points=len(points))

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    qvec = (await embed_texts([req.query]))[0]

    flt = None
    conditions = []
    if req.filter_by_source:
        conditions.append(FieldCondition(key="source", match=MatchValue(value=req.filter_by_source)))
    if req.filter_by_doc_id:
        conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=req.filter_by_doc_id)))
    if conditions:
        flt = Filter(must=conditions)

    result = client.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=max(1, min(req.top_k, 50)),
        with_payload=True,
        score_threshold=None,
        query_filter=flt
    )

    chunks = []
    for r in result:
        p = r.payload or {}
        chunks.append(RetrievedChunk(
            text=p.get("text", ""),
            score=float(r.score),
            doc_id=p.get("doc_id"),
            source=p.get("source"),
            metadata=p.get("metadata")
        ))
    return QueryResponse(chunks=chunks)

@app.get("/healthz")
async def health():
    return {"status": "ok"}
