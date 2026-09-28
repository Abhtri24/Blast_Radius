"""
Guardrail effectiveness test suite.

Tests the distance-based refusal guardrail on /ask and /ask_with_graph
with a mix of in-scope (should answer) and out-of-scope (should refuse)
questions. Outputs results to guardrail_test_results.json.

Run with:
    python test_guardrail.py
"""
import json
import time
import requests

APP_SERVICE_URL = "http://localhost:8000"
COLLECTION_NAME = "Backend-Project"
K = 6
RESULTS_FILE = "guardrail_test_results.json"

# --------------- Test set ---------------
# In-scope questions: subset from eval_questions.json (read-only, not modified)
IN_SCOPE_QUESTIONS = [
    {"id": "g_in_1",  "question": "What does the loginUser function in user.controller.js do?"},
    {"id": "g_in_2",  "question": "Which file handles user authentication middleware?"},
    {"id": "g_in_3",  "question": "Which file defines the Video schema?"},
    {"id": "g_in_4",  "question": "What does the asyncHandler utility in utils/asynchandler.js do and why is it used?"},
    {"id": "g_in_5",  "question": "Which file uploads files to Cloudinary?"},
    {"id": "g_in_6",  "question": "Which file defines the ApiError class?"},
    {"id": "g_in_7",  "question": "What does generateAccessToken and generateRefreshToken do on the User model?"},
    {"id": "g_in_8",  "question": "Which functions in user.controller.js use generateAccandRefToken?"},
    {"id": "g_in_9",  "question": "What does the multer.middleware.js file do?"},
    {"id": "g_in_10", "question": "Which files import from auth.middleware.js?"},
]

# Out-of-scope questions: general knowledge, creative, math — clearly not about any codebase
OUT_OF_SCOPE_QUESTIONS = [
    {"id": "g_out_1",  "question": "What is the capital of France?"},
    {"id": "g_out_2",  "question": "Write a poem about the ocean."},
    {"id": "g_out_3",  "question": "What is 2 + 2?"},
    {"id": "g_out_4",  "question": "Tell me a joke about programmers."},
    {"id": "g_out_5",  "question": "What year did World War II end?"},
    {"id": "g_out_6",  "question": "Explain the theory of relativity in simple terms."},
    {"id": "g_out_7",  "question": "Who won the FIFA World Cup in 2022?"},
    {"id": "g_out_8",  "question": "What is the recipe for chocolate cake?"},
    {"id": "g_out_9",  "question": "Translate 'hello world' to Spanish."},
    {"id": "g_out_10", "question": "What are the symptoms of the common cold?"},
]


def test_question(question_id, question_text, expected_behavior, endpoint="/ask"):
    """Run a single question and check guardrail behavior."""
    start = time.time()
    try:
        resp = requests.post(
            f"{APP_SERVICE_URL}{endpoint}",
            json={"question": question_text, "k": K, "collection_name": COLLECTION_NAME},
            timeout=120,
        )
        latency = time.time() - start
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {
            "id": question_id,
            "question": question_text,
            "endpoint": endpoint,
            "expected": expected_behavior,
            "error": str(e),
            "latency_sec": round(time.time() - start, 3),
        }

    guardrail_triggered = data.get("guardrail_triggered")
    best_distance = data.get("best_distance")
    actual_behavior = "should_refuse" if guardrail_triggered else "should_answer"
    correct = actual_behavior == expected_behavior

    return {
        "id": question_id,
        "question": question_text,
        "endpoint": endpoint,
        "expected": expected_behavior,
        "actual": actual_behavior,
        "guardrail_triggered": guardrail_triggered,
        "best_distance": best_distance,
        "correct": correct,
        "latency_sec": round(latency, 3),
    }


def compute_metrics(results):
    """Compute guardrail effectiveness metrics."""
    valid = [r for r in results if "error" not in r]
    total = len(valid)
    if total == 0:
        return {"error": "No valid results"}

    correct = sum(1 for r in valid if r["correct"])
    accuracy = correct / total

    # False refusal: expected should_answer but guardrail refused
    should_answer = [r for r in valid if r["expected"] == "should_answer"]
    false_refusals = sum(1 for r in should_answer if r["actual"] == "should_refuse")
    false_refusal_rate = false_refusals / len(should_answer) if should_answer else 0

    # False answer: expected should_refuse but guardrail let through
    should_refuse = [r for r in valid if r["expected"] == "should_refuse"]
    false_answers = sum(1 for r in should_refuse if r["actual"] == "should_answer")
    false_answer_rate = false_answers / len(should_refuse) if should_refuse else 0

    # Precision: of all refusals, how many were correct?
    actual_refusals = [r for r in valid if r["actual"] == "should_refuse"]
    precision = (
        sum(1 for r in actual_refusals if r["expected"] == "should_refuse") / len(actual_refusals)
        if actual_refusals else 0
    )

    # Recall: of all that should have been refused, how many were?
    recall = (
        sum(1 for r in should_refuse if r["actual"] == "should_refuse") / len(should_refuse)
        if should_refuse else 0
    )

    return {
        "total_tested": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "false_refusal_count": false_refusals,
        "false_refusal_rate": round(false_refusal_rate, 4),
        "false_answer_count": false_answers,
        "false_answer_rate": round(false_answer_rate, 4),
        "precision_on_refusals": round(precision, 4),
        "recall_on_refusals": round(recall, 4),
    }


def main():
    print("=" * 60)
    print("GUARDRAIL EFFECTIVENESS TEST")
    print("=" * 60)

    all_results = []

    # Test /ask endpoint
    print("\n--- Testing /ask endpoint ---")
    for q in IN_SCOPE_QUESTIONS:
        print(f"  [{q['id']}] {q['question'][:60]}...", end=" ", flush=True)
        result = test_question(q["id"], q["question"], "should_answer", endpoint="/ask")
        status = "✓" if result.get("correct") else "✗"
        dist = result.get("best_distance", "?")
        print(f"{status} (dist={dist})")
        all_results.append(result)

    for q in OUT_OF_SCOPE_QUESTIONS:
        print(f"  [{q['id']}] {q['question'][:60]}...", end=" ", flush=True)
        result = test_question(q["id"], q["question"], "should_refuse", endpoint="/ask")
        status = "✓" if result.get("correct") else "✗"
        dist = result.get("best_distance", "?")
        print(f"{status} (dist={dist})")
        all_results.append(result)

    # Test /ask_with_graph endpoint (same guardrail logic, should behave identically)
    print("\n--- Testing /ask_with_graph endpoint ---")
    # Use a smaller subset for the graph endpoint to keep runtime reasonable
    graph_in = IN_SCOPE_QUESTIONS[:3]
    graph_out = OUT_OF_SCOPE_QUESTIONS[:3]

    for q in graph_in:
        qid = q["id"] + "_graph"
        print(f"  [{qid}] {q['question'][:60]}...", end=" ", flush=True)
        result = test_question(qid, q["question"], "should_answer", endpoint="/ask_with_graph")
        status = "✓" if result.get("correct") else "✗"
        dist = result.get("best_distance", "?")
        print(f"{status} (dist={dist})")
        all_results.append(result)

    for q in graph_out:
        qid = q["id"] + "_graph"
        print(f"  [{qid}] {q['question'][:60]}...", end=" ", flush=True)
        result = test_question(qid, q["question"], "should_refuse", endpoint="/ask_with_graph")
        status = "✓" if result.get("correct") else "✗"
        dist = result.get("best_distance", "?")
        print(f"{status} (dist={dist})")
        all_results.append(result)

    # Compute and print metrics
    ask_results = [r for r in all_results if r["endpoint"] == "/ask"]
    graph_results = [r for r in all_results if r["endpoint"] == "/ask_with_graph"]

    ask_metrics = compute_metrics(ask_results)
    graph_metrics = compute_metrics(graph_results)
    combined_metrics = compute_metrics(all_results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n/ask endpoint ({len(ask_results)} tests):")
    for k, v in ask_metrics.items():
        print(f"  {k}: {v}")

    print(f"\n/ask_with_graph endpoint ({len(graph_results)} tests):")
    for k, v in graph_metrics.items():
        print(f"  {k}: {v}")

    print(f"\nCombined ({len(all_results)} tests):")
    for k, v in combined_metrics.items():
        print(f"  {k}: {v}")

    # Write results
    output = {
        "test_run_metadata": {
            "collection_name": COLLECTION_NAME,
            "k": K,
            "threshold": 440,
            "description": (
                "Guardrail effectiveness test: checks whether the distance-based "
                "refusal guardrail correctly refuses out-of-scope questions and "
                "correctly allows in-scope codebase questions."
            ),
        },
        "metrics": {
            "ask": ask_metrics,
            "ask_with_graph": graph_metrics,
            "combined": combined_metrics,
        },
        "results": all_results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
