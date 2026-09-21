"""
Week 4 Ex1/Ex3 evaluation script.
Runs all questions in eval_questions.json through app_service's /ask,
once per model, one full model-block at a time (to avoid Ollama contention
from multiple loaded models). Appends results incrementally to
eval_results.jsonl so a mid-run failure doesn't lose prior progress.

Run with:
    source venv/bin/activate
    python run_eval.py
"""
import json
import time
import requests

APP_SERVICE_URL = "http://localhost:8000"
COLLECTION_NAME = "Backend-Project"
MODELS = ["codellama", "starcoder2", "deepseek-coder"]
QUESTIONS_FILE = "eval_questions.json"
RESULTS_FILE = "eval_results.jsonl"
K = 6


def load_questions():
    with open(QUESTIONS_FILE) as f:
        return json.load(f)


def load_done_keys():
    """Read existing results file (if any) to support resuming."""
    done = set()
    try:
        with open(RESULTS_FILE) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                done.add((row["id"], row["model"]))
    except FileNotFoundError:
        pass
    return done


def ask(question, model):
    start = time.time()
    resp = requests.post(
        f"{APP_SERVICE_URL}/ask",
        json={
            "question": question,
            "k": K,
            "collection_name": COLLECTION_NAME,
            "model": model,
        },
        timeout=180,
    )
    latency = time.time() - start
    resp.raise_for_status()
    return resp.json(), latency


def main():
    questions = load_questions()
    done = load_done_keys()
    total = len(questions) * len(MODELS)
    count = 0

    with open(RESULTS_FILE, "a") as out:
        for model in MODELS:
            print(f"\n=== Model: {model} ===")
            for q in questions:
                count += 1
                key = (q["id"], model)
                if key in done:
                    print(f"[{count}/{total}] {q['id']} ({model}) — already done, skipping")
                    continue

                print(f"[{count}/{total}] {q['id']} ({model}) — running...", end=" ", flush=True)
                try:
                    data, latency = ask(q["question"], model)
                except Exception as e:
                    print(f"FAILED: {e}")
                    row = {
                        "id": q["id"],
                        "category": q["category"],
                        "question": q["question"],
                        "model": model,
                        "error": str(e),
                        "latency_sec": None,
                    }
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    continue

                print(f"ok ({latency:.1f}s)")
                row = {
                    "id": q["id"],
                    "category": q["category"],
                    "question": q["question"],
                    "model": model,
                    "answer": data.get("answer"),
                    "context_used": data.get("context_used"),
                    "latency_sec": round(latency, 3),
                    "prompt_tokens": data.get("prompt_tokens"),
                    "completion_tokens": data.get("completion_tokens"),
                    "total_duration_ns": data.get("total_duration_ns"),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()

    print(f"\nDone. Results in {RESULTS_FILE}")


if __name__ == "__main__":
    main()
