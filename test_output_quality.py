"""
AI Output Quality test suite.

Systematic pass/fail tests on LLM output quality across 5 dimensions:
  1. Relevance      — does the answer address the question asked?
  2. Context support — do claims trace back to context_used?
  3. Hallucination   — does the answer reference unsupported files/functions?
  4. Format          — do code-gen answers contain code blocks?
  5. Refusal         — does the guardrail fire correctly?

Each test produces a clear PASS/FAIL per question per metric.
Results are written to output_quality_results.json.

METHODOLOGY (required by assignment):
  - Relevance: keyword overlap — extract nouns/terms from the question and
    check that at least 40% appear in the answer. Simple, deterministic,
    reproducible. Does NOT use LLM-as-judge (noted for transparency).
  - Context support: for every filename mentioned in the answer, check if
    it also appears in context_used's file list. PASS if all mentioned
    files are supported, or if the answer mentions no specific filenames.
  - Hallucination: extract all *.js, *.py, etc. filenames from the answer
    via regex; flag any that don't appear in context_used or graph_hits.
    PASS if zero unsupported filenames found.
  - Format compliance: only applied to code_generation category questions.
    Check for triple-backtick code fences. PASS if at least one code block
    is present.
  - Appropriate refusal: for out-of-scope questions, PASS if guardrail
    fired; for in-scope questions, PASS if guardrail did NOT fire.

Run with:
    python test_output_quality.py
"""
import json
import re
import time
import requests

APP_SERVICE_URL = "http://localhost:8000"
COLLECTION_NAME = "Backend-Project"
K = 6
RESULTS_FILE = "output_quality_results.json"

# Filename regex — same as app_service uses
FILENAME_RE = re.compile(r"\b[\w\-]+\.(?:js|jsx|ts|tsx|py|java|go|rb|json|css|html|md)\b")
CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")

# --------------- Test questions ---------------
# Mix of categories from eval_questions.json (read-only; we just reference them here)
IN_SCOPE_QUESTIONS = [
    {"id": "oq_1",  "category": "code_explanation",         "question": "What does the loginUser function in user.controller.js do?",
     "expected_refusal": False},
    {"id": "oq_2",  "category": "code_explanation",         "question": "What does the asyncHandler utility in utils/asynchandler.js do and why is it used?",
     "expected_refusal": False},
    {"id": "oq_3",  "category": "code_retrieval",           "question": "Which file handles user authentication middleware?",
     "expected_refusal": False},
    {"id": "oq_4",  "category": "code_retrieval",           "question": "Which file defines the Video schema?",
     "expected_refusal": False},
    {"id": "oq_5",  "category": "code_retrieval",           "question": "Which file uploads files to Cloudinary?",
     "expected_refusal": False},
    {"id": "oq_6",  "category": "dependency_understanding", "question": "Which functions in user.controller.js use generateAccandRefToken?",
     "expected_refusal": False},
    {"id": "oq_7",  "category": "dependency_understanding", "question": "What would break if ApiError.js were deleted?",
     "expected_refusal": False},
    {"id": "oq_8",  "category": "bug_analysis",             "question": "What could cause loginUser to return a 401 even with a correct password?",
     "expected_refusal": False},
    {"id": "oq_9",  "category": "code_generation",          "question": "Write a unit test for loginUser.",
     "expected_refusal": False},
    {"id": "oq_10", "category": "code_generation",          "question": "Write an Express middleware that checks whether a JWT access token is expired, similar to auth.middleware.js.",
     "expected_refusal": False},
    {"id": "oq_11", "category": "refactoring",              "question": "Suggest an improvement to how loginUser handles missing credentials.",
     "expected_refusal": False},
    {"id": "oq_12", "category": "rag_project_level",        "question": "What does this project's authentication flow look like end-to-end, from route to controller to model?",
     "expected_refusal": False},
]

OUT_OF_SCOPE_QUESTIONS = [
    {"id": "oq_oos_1", "category": "out_of_scope", "question": "What is the capital of France?",
     "expected_refusal": True},
    {"id": "oq_oos_2", "category": "out_of_scope", "question": "Write a poem about the ocean.",
     "expected_refusal": True},
    {"id": "oq_oos_3", "category": "out_of_scope", "question": "Explain quantum computing in simple terms.",
     "expected_refusal": True},
]

ALL_QUESTIONS = IN_SCOPE_QUESTIONS + OUT_OF_SCOPE_QUESTIONS


# --------------- Metric functions ---------------

def extract_question_terms(question):
    """Extract meaningful terms from a question for relevance checking."""
    # Remove common stop words and question words
    stop_words = {
        "what", "does", "the", "do", "is", "are", "how", "why", "which",
        "file", "this", "that", "and", "or", "a", "an", "in", "on", "to",
        "for", "of", "it", "be", "from", "with", "if", "there", "any",
        "could", "would", "should", "has", "have", "was", "were", "can",
        "write", "suggest", "explain", "based", "about", "look", "like",
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-.]+", question.lower())
    terms = [w for w in words if w not in stop_words and len(w) > 2]
    return terms


def check_relevance(question, answer):
    """
    PASS if at least 40% of question terms appear in the answer.
    Method: keyword overlap (deterministic, no LLM-as-judge).
    """
    terms = extract_question_terms(question)
    if not terms:
        return True, "No terms to check"
    answer_lower = answer.lower()
    hits = sum(1 for t in terms if t in answer_lower)
    ratio = hits / len(terms)
    detail = f"{hits}/{len(terms)} terms found ({ratio:.0%})"
    return ratio >= 0.4, detail


def check_context_support(answer, context_used):
    """
    PASS if every filename mentioned in the answer also appears in context_used.
    Method: extract filenames from answer, check against context file list.
    """
    answer_files = set(FILENAME_RE.findall(answer.lower()))
    if not answer_files:
        return True, "No filenames in answer"

    context_files = set()
    for c in (context_used or []):
        fname = c.get("file", "")
        # Extract just the basename for comparison
        basename = fname.rsplit("/", 1)[-1].lower() if "/" in fname else fname.lower()
        context_files.add(basename)
        # Also add the full path lowered
        context_files.add(fname.lower())

    unsupported = answer_files - context_files
    if unsupported:
        return False, f"Unsupported files in answer: {unsupported}"
    return True, f"All {len(answer_files)} file(s) found in context"


def check_hallucination(answer, context_used, graph_hits=None):
    """
    PASS if no filenames in the answer are absent from context_used + graph_hits.
    Method: regex extraction of filenames, string matching against context.
    """
    answer_files = set(FILENAME_RE.findall(answer.lower()))
    if not answer_files:
        return True, "No filenames to check"

    supported_files = set()
    for c in (context_used or []):
        fname = c.get("file", "")
        basename = fname.rsplit("/", 1)[-1].lower() if "/" in fname else fname.lower()
        supported_files.add(basename)
        supported_files.add(fname.lower())
        # Also extract filenames from the chunk text itself
        for f in FILENAME_RE.findall(c.get("text", "").lower()):
            supported_files.add(f)

    for h in (graph_hits or []):
        if "file" in h:
            fname = h["file"]
            basename = fname.rsplit("/", 1)[-1].lower() if "/" in fname else fname.lower()
            supported_files.add(basename)
            supported_files.add(fname.lower())

    # Also consider common well-known filenames that aren't hallucinations
    # (e.g. package.json is always valid to mention if it appears in context text)
    hallucinated = answer_files - supported_files
    if hallucinated:
        return False, f"Hallucinated files: {hallucinated}"
    return True, f"All {len(answer_files)} file(s) grounded in context"


def check_format_compliance(answer, category):
    """
    PASS if code_generation answers contain at least one code block.
    Only applied to code_generation category.
    Method: regex for triple-backtick fences.
    """
    if category != "code_generation":
        return True, "N/A (not a code_generation question)"
    has_code = bool(CODE_BLOCK_RE.search(answer))
    if has_code:
        return True, "Code block found"
    return False, "No code block found in code_generation answer"


def check_appropriate_refusal(guardrail_triggered, expected_refusal):
    """
    PASS if guardrail behavior matches expectation.
    """
    actually_refused = guardrail_triggered is not None
    if expected_refusal and actually_refused:
        return True, "Correctly refused"
    elif expected_refusal and not actually_refused:
        return False, "Should have refused but answered"
    elif not expected_refusal and actually_refused:
        return False, f"Should have answered but refused ({guardrail_triggered})"
    else:
        return True, "Correctly answered"


# --------------- Main ---------------

def main():
    print("=" * 60)
    print("AI OUTPUT QUALITY TEST SUITE")
    print("=" * 60)

    results = []
    summary_counts = {
        "relevance": {"pass": 0, "fail": 0, "total": 0},
        "context_support": {"pass": 0, "fail": 0, "total": 0},
        "hallucination": {"pass": 0, "fail": 0, "total": 0},
        "format_compliance": {"pass": 0, "fail": 0, "total": 0},
        "appropriate_refusal": {"pass": 0, "fail": 0, "total": 0},
    }

    for q in ALL_QUESTIONS:
        print(f"\n[{q['id']}] {q['question'][:70]}...")
        start = time.time()
        try:
            resp = requests.post(
                f"{APP_SERVICE_URL}/ask",
                json={"question": q["question"], "k": K, "collection_name": COLLECTION_NAME},
                timeout=180,
            )
            latency = time.time() - start
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": q["id"], "question": q["question"], "error": str(e),
            })
            continue

        answer = data.get("answer", "")
        context_used = data.get("context_used", [])
        graph_hits = data.get("graph_hits", [])
        guardrail = data.get("guardrail_triggered")
        best_distance = data.get("best_distance")

        # Skip quality checks if guardrail fired (answer is the refusal message)
        if guardrail:
            tests = {}
            # Only refusal check applies
            ref_pass, ref_detail = check_appropriate_refusal(guardrail, q["expected_refusal"])
            tests["appropriate_refusal"] = {"pass": ref_pass, "detail": ref_detail}
            summary_counts["appropriate_refusal"]["total"] += 1
            summary_counts["appropriate_refusal"]["pass" if ref_pass else "fail"] += 1

            # Mark other tests as N/A
            for metric in ["relevance", "context_support", "hallucination", "format_compliance"]:
                tests[metric] = {"pass": True, "detail": "N/A (guardrail fired)"}

            print(f"  Guardrail fired (dist={best_distance}) — refusal check: {'PASS' if ref_pass else 'FAIL'}")
        else:
            tests = {}

            # 1. Relevance
            rel_pass, rel_detail = check_relevance(q["question"], answer)
            tests["relevance"] = {"pass": rel_pass, "detail": rel_detail}
            summary_counts["relevance"]["total"] += 1
            summary_counts["relevance"]["pass" if rel_pass else "fail"] += 1
            print(f"  Relevance:        {'PASS' if rel_pass else 'FAIL'} — {rel_detail}")

            # 2. Context support
            ctx_pass, ctx_detail = check_context_support(answer, context_used)
            tests["context_support"] = {"pass": ctx_pass, "detail": ctx_detail}
            summary_counts["context_support"]["total"] += 1
            summary_counts["context_support"]["pass" if ctx_pass else "fail"] += 1
            print(f"  Context support:  {'PASS' if ctx_pass else 'FAIL'} — {ctx_detail}")

            # 3. Hallucination
            hal_pass, hal_detail = check_hallucination(answer, context_used, graph_hits)
            tests["hallucination"] = {"pass": hal_pass, "detail": hal_detail}
            summary_counts["hallucination"]["total"] += 1
            summary_counts["hallucination"]["pass" if hal_pass else "fail"] += 1
            print(f"  Hallucination:    {'PASS' if hal_pass else 'FAIL'} — {hal_detail}")

            # 4. Format compliance
            fmt_pass, fmt_detail = check_format_compliance(answer, q["category"])
            tests["format_compliance"] = {"pass": fmt_pass, "detail": fmt_detail}
            summary_counts["format_compliance"]["total"] += 1
            summary_counts["format_compliance"]["pass" if fmt_pass else "fail"] += 1
            print(f"  Format:           {'PASS' if fmt_pass else 'FAIL'} — {fmt_detail}")

            # 5. Appropriate refusal
            ref_pass, ref_detail = check_appropriate_refusal(guardrail, q["expected_refusal"])
            tests["appropriate_refusal"] = {"pass": ref_pass, "detail": ref_detail}
            summary_counts["appropriate_refusal"]["total"] += 1
            summary_counts["appropriate_refusal"]["pass" if ref_pass else "fail"] += 1
            print(f"  Refusal:          {'PASS' if ref_pass else 'FAIL'} — {ref_detail}")

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer_preview": answer[:200] if answer else None,
            "best_distance": best_distance,
            "guardrail_triggered": guardrail,
            "latency_sec": round(latency, 3),
            "tests": tests,
        })

    # --------------- Summary ---------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    methodology = {
        "relevance": (
            "Keyword overlap: extract meaningful terms from the question (excluding stop words), "
            "check what fraction appear in the answer text. PASS threshold: ≥40% of terms present. "
            "Deterministic, no LLM-as-judge."
        ),
        "context_support": (
            "Filename grounding: extract all filenames (*.js, *.py, etc.) from the answer via regex, "
            "verify each appears in the context_used chunk file list. PASS if all mentioned files "
            "are present in context, or if the answer mentions no specific files."
        ),
        "hallucination": (
            "Unsupported reference detection: extract filenames from the answer, check against "
            "both context_used files AND the text content of context chunks. PASS if zero "
            "filenames appear that aren't grounded in the provided context."
        ),
        "format_compliance": (
            "Code block presence: for code_generation category questions only, regex-check for "
            "triple-backtick code fences (```...```). PASS if at least one code block is found. "
            "N/A for non-code-generation questions."
        ),
        "appropriate_refusal": (
            "Guardrail correctness: for out-of-scope questions, PASS if guardrail_triggered is "
            "non-null (question was correctly refused). For in-scope questions, PASS if "
            "guardrail_triggered is null (question was correctly answered)."
        ),
    }

    for metric, counts in summary_counts.items():
        total = counts["total"]
        if total == 0:
            continue
        pass_rate = counts["pass"] / total
        print(f"\n{metric}:")
        print(f"  {counts['pass']}/{total} PASS ({pass_rate:.0%})")
        print(f"  Method: {methodology[metric]}")

    # Write output
    output = {
        "test_run_metadata": {
            "collection_name": COLLECTION_NAME,
            "k": K,
            "total_questions": len(ALL_QUESTIONS),
            "description": (
                "AI output quality test suite. Tests LLM answers across 5 dimensions: "
                "relevance, context support, hallucination, format compliance, and "
                "appropriate refusal. Each test produces a clear PASS/FAIL."
            ),
        },
        "methodology": methodology,
        "summary": {
            metric: {
                "pass": counts["pass"],
                "fail": counts["fail"],
                "total": counts["total"],
                "pass_rate": round(counts["pass"] / counts["total"], 4) if counts["total"] > 0 else None,
            }
            for metric, counts in summary_counts.items()
        },
        "results": results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
