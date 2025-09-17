from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import uuid4
import os
import httpx
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# --- Environment Variables ---
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "isc2_toronto_v3")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "tinyllama")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434").rstrip("/")

# --- FastAPI App ---
app = FastAPI(title="RAG API v3", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for simplicity in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Qdrant Client ---
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# --- Helper Functions ---
async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of texts using the Ollama API."""
    # Note: This simple version embeds one text at a time.
    # For batching, the payload structure and loop would need to be adjusted.
    try:
        async with httpx.AsyncClient(timeout=300) as http_client:
            response = await http_client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": texts[0]},
            )
            response.raise_for_status()
            data = response.json()
            if "embedding" in data:
                return [data["embedding"]]
            else:
                raise HTTPException(502, f"Unexpected Ollama embeddings format: {data}")
    except httpx.RequestError as e:
        raise HTTPException(503, f"Could not connect to Ollama for embeddings: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Ollama embed error: {e.response.text[:200]}")

# in app/main.py

async def generate_answer(context: str, question: str, current_date: str) -> str:
    """
    Sends the retrieved context and a question to the LLM to generate an answer.
    """
    # A more sophisticated prompt with a persona and stricter rules
    prompt = f"""

You are a helpful assistant for the (ISC)² Toronto Chapter. Your tone should be friendly and concise.
Base your answer ONLY on the information within the provided CONTEXT.

Here are your rules:
1. The current date is {current_date}. Use this to determine which information is relevant.
2. When asked about events, find the event that occurs next after the current date. Ignore events from the past.
3. If asked for a link and the context contains a URL, provide it. If not, do not make one up.
4. If the CONTEXT does not contain the information to answer, you MUST say "I'm sorry, but I don't have enough information about that topic."

CONTEXT:
---
{context}
---

QUESTION: {question}
"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    async with httpx.AsyncClient(timeout=300) as client:
        try:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "Error: No response from LLM.").strip()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Could not connect to Ollama: {e}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Ollama generate error: {e.response.text}")



def ensure_collection(vector_size: int):
    """Creates the Qdrant collection if it doesn't exist."""
    if not client.collection_exists(COLLECTION):
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        client.create_payload_index(COLLECTION, field_name="doc_id", field_schema="keyword")

def chunk_text(text: str, max_tokens: int = 400, overlap: int = 60) -> List[str]:
    """A simple text chunking function."""
    words = text.split()
    if not words: return []
    step = max_tokens - overlap
    if step <= 0: step = max_tokens
    chunks = [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), step)]
    return [chunk for chunk in chunks if chunk]

# --- Pydantic Models ---
class IngestItem(BaseModel):
    text: str
    doc_id: Optional[str] = None
    source: Optional[str] = None

class IngestRequest(BaseModel):
    items: List[IngestItem]

class IngestResponse(BaseModel):
    inserted_points: int

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filter_by_doc_id: Optional[str] = None

class RetrievedChunk(BaseModel):
    text: str
    score: float
    doc_id: Optional[str] = None
    source: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
#    chunks: List[RetrievedChunk] #outputs source chuncks

# --- API Endpoints ---
@app.on_event("startup")
async def startup():
    """Ensures the Qdrant collection exists on application startup."""
    vec = await embed_texts(["ping"])
    dim = len(vec[0])
    ensure_collection(dim)

@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Ingests text into the Qdrant database."""
    texts, payloads, ids = [], [], []
    for item in req.items:
        base_doc_id = item.doc_id or str(uuid4())
        chunks = chunk_text(item.text)
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            payloads.append({"doc_id": base_doc_id, "source": item.source, "chunk_ix": idx})
            ids.append(str(uuid4())) # Generate a valid UUID for each chunk
    if not texts:
        return IngestResponse(inserted_points=0)
    
    vectors = await embed_texts(texts)
    points = [PointStruct(id=ids[i], vector=vectors[i], payload={**payloads[i], "text": texts[i]}) for i in range(len(texts))]
    client.upsert(collection_name=COLLECTION, points=points)
    return IngestResponse(inserted_points=len(points))

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Performs RAG to answer a query."""
    qvec = (await embed_texts([req.query]))[0]
    
    query_filter = None
    if req.filter_by_doc_id:
        query_filter = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=req.filter_by_doc_id))])

    search_result = client.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=req.top_k,
        with_payload=True,
        query_filter=query_filter
    )
    
    context_str = "\n---\n".join([r.payload.get("text", "") for r in search_result])

    # Get the current date to pass to the LLM
    current_date = datetime.now().strftime("%B %d, %Y")
    final_answer = await generate_answer(context_str, req.query, current_date)
   
#    final_answer = await generate_answer(context_str, req.query)
    
    retrieved_chunks = [RetrievedChunk(
        text=r.payload.get("text", ""),
        score=r.score,
        doc_id=r.payload.get("doc_id"),
        source=r.payload.get("source")
    ) for r in search_result]
    
    #return QueryResponse(answer=final_answer, chunks=retrieved_chunks) #return includes chuncks / sources
    return QueryResponse(answer=final_answer)
@app.get("/healthz")
async def health():
    return {"status": "ok"}
