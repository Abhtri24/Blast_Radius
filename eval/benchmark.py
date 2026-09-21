"""
Week 4 - Exercises 1-4: run the SAME question set through 3 local models,
logging response, latency, and CPU/memory usage so you can quantitatively
compare them. Correctness / relevance / hallucination-rate columns are left
for you to fill in by review, since those need human or LLM-judge scoring --
this script defines exactly how the automatic metrics are calculated
(so you can describe that clearly in your report).

Run overnight/unattended, e.g.:
    nohup python benchmark.py > benchmark.log 2>&1 &

Requires the models already pulled, e.g.:
    ollama pull codellama
    ollama pull starcoder2
    ollama pull qwen2.5:1.5b   # or another 3rd model of your choice
"""
import json
import time
import csv
import psutil
import ollama

MODELS = ["codellama", "starcoder2", "qwen2.5:1.5b"]   # edit to your 3 models
QUESTIONS_FILE = "eval_questions.json"
OUTPUT_CSV = "benchmark_results.csv"


def sample_resources():
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "mem_percent": psutil.virtual_memory().percent,
    }


def run_benchmark():
    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)

    rows = []
    for model in MODELS:
        for q in questions:
            before = sample_resources()
            start = time.time()
            try:
                resp = ollama.chat(model=model, messages=[{"role": "user", "content": q["question"]}])
                answer = resp["message"]["content"]
                token_count = resp.get("eval_count", None)  # tokens generated, if Ollama reports it
                error = ""
            except Exception as e:
                answer = ""
                token_count = None
                error = str(e)
            latency_s = time.time() - start
            after = sample_resources()

            rows.append({
                "model": model,
                "question_id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "answer": answer,
                "latency_seconds": round(latency_s, 2),
                "token_count": token_count,
                "cpu_percent_avg": round((before["cpu_percent"] + after["cpu_percent"]) / 2, 1),
                "mem_percent_avg": round((before["mem_percent"] + after["mem_percent"]) / 2, 1),
                "error": error,
                # Fill these in manually after reviewing the answer against the retrieved context / ground truth:
                "correctness_1to5": "",
                "relevance_1to5": "",
                "hallucinated_claims": "",     # e.g. "1/4 claims unsupported" -> 25% hallucination rate
                "test_passed": "",             # for code-generation questions only
            })
            print(f"[{model}] Q{q['id']} done in {latency_s:.1f}s")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_benchmark()
