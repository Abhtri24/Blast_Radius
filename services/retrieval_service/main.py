"""
Retrieval/RAG Service - owns vector search only.
Run standalone: uvicorn main:app --port 8001
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import ollama

app = FastAPI(title="Retrieval Service")

EMBED_MODEL = "nomic-embed-text"
DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_client = ollama.Client(host=OLLAMA_HOST)


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    collection_name: str = "repo_chunks"


@app.post("/search")
def search(req: SearchRequest):
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(req.collection_name)
    query_emb = _client.embeddings(model=EMBED_MODEL, prompt=req.query)["embedding"]
    results = collection.query(query_embeddings=[query_emb], n_results=req.k)
    distances = results.get("distances", [[None] * len(results["documents"][0])])[0]
    chunks = [
        {
            "text": doc,
            "file": meta["file"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "distance": dist,
        }
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], distances)
    ]
    return {"chunks": chunks}


@app.get("/health")
def health():
    return {"status": "ok"}
