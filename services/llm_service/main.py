"""
LLM Service - owns talking to whichever model backend is configured.
Run standalone: uvicorn main:app --port 8002
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import requests

app = FastAPI(title="LLM Service")

BACKEND = os.environ.get("LLM_BACKEND", "ollama")           # "ollama" or "groq"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "codellama")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

_client = ollama.Client(host=OLLAMA_HOST, headers={"ngrok-skip-browser-warning": "true"})


class GenerateRequest(BaseModel):
    prompt: str
    model: str | None = None


@app.post("/generate")
def generate(req: GenerateRequest):
    if BACKEND == "groq":
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": req.model or GROQ_MODEL, "messages": [{"role": "user", "content": req.prompt}]},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return {
            "answer": data["choices"][0]["message"]["content"],
            "backend": "groq",
            "model": req.model or GROQ_MODEL,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_duration_ns": None,
        }
    else:
        resp = _client.chat(
            model=req.model or OLLAMA_MODEL,
            messages=[{"role": "user", "content": req.prompt}],
            options={"num_predict": 400},
            keep_alive="30m",
        )
        return {
            "answer": resp["message"]["content"],
            "backend": "ollama",
            "model": req.model or OLLAMA_MODEL,
            "prompt_tokens": resp.get("prompt_eval_count"),
            "completion_tokens": resp.get("eval_count"),
            "total_duration_ns": resp.get("total_duration"),
        }


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND}
