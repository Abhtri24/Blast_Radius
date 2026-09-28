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
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(req.collection_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection '{req.collection_name}' not found: {str(e)}")

    query_emb = None
    try:
        query_emb = _client.embeddings(model=EMBED_MODEL, prompt=req.query)["embedding"]
    except Exception as e:
        print(f"[retrieval_service] Ollama embedding failed ({e}). Using text fallback search.")

    if query_emb is not None:
        try:
            results = collection.query(query_embeddings=[query_emb], n_results=req.k)
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[0.0] * len(documents)])[0]
            chunks = [
                {
                    "text": doc,
                    "file": meta.get("file", "unknown") if isinstance(meta, dict) else "unknown",
                    "start_line": meta.get("start_line", 1) if isinstance(meta, dict) else 1,
                    "end_line": meta.get("end_line", 1) if isinstance(meta, dict) else 1,
                    "distance": dist,
                }
                for doc, meta, dist in zip(documents, metadatas, distances)
            ]
            return {"chunks": chunks}
        except Exception as e:
            print(f"[retrieval_service] Vector query failed ({e}). Falling back to text search.")

    # Fallback text matching when Ollama is unavailable or embedding fails
    try:
        all_data = collection.get(include=["documents", "metadatas"])
        all_docs = all_data.get("documents", [])
        all_metas = all_data.get("metadatas", [])

        if not all_docs:
            return {"chunks": []}

        words = [w.lower() for w in req.query.split() if len(w) > 1]
        scored = []
        for doc, meta in zip(all_docs, all_metas):
            doc_lower = doc.lower()
            score = sum(doc_lower.count(w) for w in words) if words else 0
            scored.append((score, doc, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = scored[:req.k]

        max_score = max([item[0] for item in top_k], default=1)
        chunks = [
            {
                "text": item[1],
                "file": item[2].get("file", "unknown") if isinstance(item[2], dict) else "unknown",
                "start_line": item[2].get("start_line", 1) if isinstance(item[2], dict) else 1,
                "end_line": item[2].get("end_line", 1) if isinstance(item[2], dict) else 1,
                "distance": float(round(300.0 - (200.0 * item[0] / max_score) if max_score > 0 else 350.0, 2)),
            }
            for item in top_k
        ]
        return {"chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}

