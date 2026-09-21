"""
Week 4 - Exercise 6: probe whether your RAG system can answer questions that
need reasoning across MULTIPLE files/modules, not just single-chunk lookup.
This is intentionally a stretch test -- expect some failures, and document
them. That's the point of the exercise (Sourcegraph-style tooling is next
week's fix for exactly this gap).

Run from the project root:
    python eval/repo_understanding_test.py
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rag_core import answer_question  # noqa: E402

# Edit these to reference real files/functions in your target repo
MULTI_FILE_QUESTIONS = [
    "Which files are involved in user authentication, end to end?",
    "Which function or module calls the payment/checkout logic?",
    "What happens, across the codebase, after a user submits a registration request?",
    "If I modify the main entry-point file, which other components may be affected?",
    "Which test files exercise the authentication logic?",
]


def run():
    for i, q in enumerate(MULTI_FILE_QUESTIONS, 1):
        answer, chunks = answer_question(q, k=8)
        files_touched = sorted({c["file"] for c in chunks})
        print(f"\n{'='*80}\nQ{i}: {q}")
        print(f"Files retrieved (spread across repo = good sign): {files_touched}")
        print(f"ANSWER:\n{answer}")
        print(
            "\n-> Note by hand: did the answer actually connect multiple files, "
            "or just describe the single closest chunk?"
        )


if __name__ == "__main__":
    run()
