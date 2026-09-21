"""
Application Service / Orchestrator - the only service the user talks to.
Run standalone: uvicorn main:app --port 8000
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title="Application Service")

RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://retrieval_service:8000")
LLM_URL = os.environ.get("LLM_URL", "http://llm_service:8000")

# Guardrail: refuse to answer when the best retrieved chunk is not a close
# enough match. Distance is Chroma's L2 distance (lower = more similar).
# Threshold calibrated empirically: genuinely relevant matches observed in
# the 237-417 range, clearly out-of-scope queries observed in the 464-554
# range. 440 sits in the gap between them. See eval notes for the sample
# used to calibrate this.
DISTANCE_REFUSAL_THRESHOLD = float(os.environ.get("DISTANCE_REFUSAL_THRESHOLD", 440))

REFUSAL_MESSAGE = (
    "I don't have enough relevant information in the indexed codebase to "
    "answer this confidently. This may be outside the scope of the "
    "indexed repository, or the relevant code may not have been retrieved."
)


class AskRequest(BaseModel):
    question: str
    k: int = 5
    collection_name: str = "repo_chunks"
    model: str | None = None


@app.post("/ask")
def ask(req: AskRequest):
    search_resp = requests.post(
        f"{RETRIEVAL_URL}/search",
        json={"query": req.question, "k": req.k, "collection_name": req.collection_name},
        timeout=90,
    )
    chunks = search_resp.json()["chunks"]

    distances = [c["distance"] for c in chunks if c.get("distance") is not None]
    best_distance = min(distances) if distances else None

    if best_distance is not None and best_distance > DISTANCE_REFUSAL_THRESHOLD:
        return {
            "question": req.question,
            "answer": REFUSAL_MESSAGE,
            "context_used": chunks,
            "model": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_duration_ns": None,
            "guardrail_triggered": "low_retrieval_confidence",
            "best_distance": best_distance,
        }

    context_text = "\n\n".join(
        f"[{c['file']} lines {c['start_line']}-{c['end_line']}]\n{c['text']}" for c in chunks
    )
    prompt = (
        "You are a code assistant. Use ONLY the context below to answer. "
        "If the context is insufficient, say so explicitly.\n\n"
        f"CONTEXT:\n{context_text}\n\nQUESTION: {req.question}\nANSWER:"
    )
    gen_payload = {"prompt": prompt}
    if req.model:
        gen_payload["model"] = req.model
    gen_resp = requests.post(f"{LLM_URL}/generate", json=gen_payload, timeout=120)
    gen_data = gen_resp.json()
    return {
        "question": req.question,
        "answer": gen_data.get("answer"),
        "context_used": chunks,
        "model": gen_data.get("model"),
        "prompt_tokens": gen_data.get("prompt_tokens"),
        "completion_tokens": gen_data.get("completion_tokens"),
        "total_duration_ns": gen_data.get("total_duration_ns"),
        "guardrail_triggered": None,
        "best_distance": best_distance,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
