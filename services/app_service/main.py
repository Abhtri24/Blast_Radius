"""
Application Service / Orchestrator - the only service the user talks to.
Run standalone: uvicorn main:app --port 8000
"""
import os
import re
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

# --------------- Sourcegraph config ---------------
SOURCEGRAPH_URL = os.environ.get("SOURCEGRAPH_URL", "http://host.docker.internal:7080")
SOURCEGRAPH_TOKEN = os.environ.get("SOURCEGRAPH_TOKEN", "")
SOURCEGRAPH_REPO = os.environ.get("SOURCEGRAPH_REPO", "github.com/Abhtri24/Backend-Project")

FILENAME_PATTERN = re.compile(r"\b[\w\-]+\.(?:js|jsx|ts|tsx|py|java|go|rb|json)\b")


def extract_filenames(text):
    return list(dict.fromkeys(FILENAME_PATTERN.findall(text)))


def graph_search(term):
    query = f"repo:^{re.escape(SOURCEGRAPH_REPO)}$ {term}"
    gql = {
        "query": (
            "query($query: String!) { "
            "search(query: $query, version: V3) { "
            "results { results { __typename "
            "... on FileMatch { file { path } repository { name } "
            "lineMatches { lineNumber preview } } } } } }"
        ),
        "variables": {"query": query},
    }
    headers = {"Content-Type": "application/json"}
    if SOURCEGRAPH_TOKEN:
        headers["Authorization"] = f"token {SOURCEGRAPH_TOKEN}"
    try:
        resp = requests.post(f"{SOURCEGRAPH_URL}/.api/graphql", json=gql, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        matches = data.get("data", {}).get("search", {}).get("results", {}).get("results", [])
        hits = []
        for m in matches:
            if m.get("__typename") != "FileMatch":
                continue
            path = m["file"]["path"]
            for lm in m.get("lineMatches", []):
                hits.append({"file": path, "line": lm["lineNumber"] + 1, "preview": lm["preview"]})
        return hits
    except Exception as e:
        return [{"error": str(e)}]


# --------------- /ask (standard RAG) ---------------

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


# --------------- /ask_with_graph (RAG + Sourcegraph) ---------------

class AskGraphRequest(BaseModel):
    question: str
    k: int = 5
    collection_name: str = "repo_chunks"
    model: str | None = None


@app.post("/ask_with_graph")
def ask_with_graph(req: AskGraphRequest):
    # Step 1: Retrieve chunks from ChromaDB
    search_resp = requests.post(
        f"{RETRIEVAL_URL}/search",
        json={"query": req.question, "k": req.k, "collection_name": req.collection_name},
        timeout=90,
    )
    chunks = search_resp.json()["chunks"]

    # Step 2: Guardrail — same distance check as /ask
    distances = [c["distance"] for c in chunks if c.get("distance") is not None]
    best_distance = min(distances) if distances else None

    if best_distance is not None and best_distance > DISTANCE_REFUSAL_THRESHOLD:
        # Skip both Sourcegraph and LLM — no point querying further
        return {
            "question": req.question,
            "answer": REFUSAL_MESSAGE,
            "context_used": chunks,
            "graph_hits": [],
            "filenames_searched": [],
            "model": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_duration_ns": None,
            "guardrail_triggered": "low_retrieval_confidence",
            "best_distance": best_distance,
        }

    # Step 3: Sourcegraph structural search
    filenames = extract_filenames(req.question)
    graph_hits = []
    for fname in filenames:
        graph_hits.extend(graph_search(fname))

    # Step 4: Build prompt with both Chroma context and Sourcegraph hits
    context_text = "\n\n".join(
        f"[{c['file']} lines {c['start_line']}-{c['end_line']}]\n{c['text']}" for c in chunks
    )

    if filenames:
        if graph_hits and not any("error" in h for h in graph_hits):
            graph_text = "\n".join(
                f"- {h['file']} line {h['line']}: {h['preview'].strip()}" for h in graph_hits
            )
            graph_section = (
                f"\n\nSTRUCTURAL SEARCH (Sourcegraph) for {', '.join(filenames)}:\n"
                f"Found {len(graph_hits)} reference(s) across the repo:\n{graph_text}"
            )
        else:
            graph_section = (
                f"\n\nSTRUCTURAL SEARCH (Sourcegraph) for {', '.join(filenames)}:\n"
                f"No references found outside the file(s) where they are defined. "
                f"This strongly suggests they are unused elsewhere in the repository."
            )
    else:
        graph_section = ""

    prompt = (
        "You are a code assistant. Use ONLY the context below to answer. "
        "The STRUCTURAL SEARCH section (if present) shows real, verified "
        "cross-file references found by a code search tool -- trust it over "
        "your own assumptions about usage. If the context is insufficient, "
        "say so explicitly.\n\n"
        f"CONTEXT:\n{context_text}{graph_section}\n\nQUESTION: {req.question}\nANSWER:"
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
        "graph_hits": graph_hits,
        "filenames_searched": filenames,
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
