"""
Week 3 - Exercise 3: Retrieval + RAG core logic.
Shared by app.py (monolith) and later by the split-out services.
"""
import os
import chromadb
import ollama

EMBED_MODEL = "nomic-embed-text"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "repo_chunks"

# Point this at a Colab/remote Ollama instance to use its GPU instead of
# your VM's CPU, e.g.: export OLLAMA_HOST=https://<your-tunnel>.trycloudflare.com
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_client = ollama.Client(host=OLLAMA_HOST)

# Which backend generates the final answer. Flip via env var so you can
# swap Ollama <-> a hosted API (e.g. Groq) without touching any other code.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")  # "ollama" or "groq"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "codellama")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


def get_collection(collection_name="repo_chunks"):
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(collection_name)


def retrieve_context(query, k=5, collection_name="repo_chunks"):
    collection = get_collection(collection_name)
    query_emb = _client.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    results = collection.query(query_embeddings=[query_emb], n_results=k)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "file": meta["file"],
                        "start_line": meta["start_line"], "end_line": meta["end_line"]})
    return chunks


def generate_answer(question, context_chunks, model=None):
    context_text = "\n\n".join(
        f"[{c['file']} lines {c['start_line']}-{c['end_line']}]\n{c['text']}" for c in context_chunks
    )
    prompt = (
        "You are a code assistant. Use ONLY the context below to answer. "
        "If the context is insufficient, say so explicitly instead of guessing.\n\n"
        f"CONTEXT:\n{context_text}\n\nQUESTION: {question}\nANSWER:"
    )

    if LLM_BACKEND == "groq":
        import requests
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": model or GROQ_MODEL, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    else:
        resp = _client.chat(
            model=model or OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 400},   # cap output length so it finishes well under the tunnel's 120s limit
            keep_alive="30m",               # keep the model loaded in GPU memory between requests
        )
        return resp["message"]["content"]


def answer_question(question, k=5, model=None, collection_name="repo_chunks"):
    """Full pipeline: Question -> Retrieval -> Context -> LLM -> Answer."""
    chunks = retrieve_context(question, k=k, collection_name=collection_name)
    answer = generate_answer(question, chunks, model=model)
    return answer, chunks
