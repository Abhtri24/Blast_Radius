"""
Week 4 - Exercise 5: trace RETRIEVAL -> CONTEXT -> ANSWER for a handful of
questions so you can manually spot: good retrieval, bad/irrelevant retrieval,
missed info, correct answers, and hallucination despite having context.

Run from the project root (needs rag_core.py and an existing chroma_db):
    python eval/rag_trace.py
"""
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rag_core import answer_question  # noqa: E402

# Pick ~8-10 representative questions from eval_questions.json for close reading
TRACE_QUESTION_IDS = [1, 3, 5, 6, 9, 13, 15, 19]


def run_trace():
    with open(os.path.join(os.path.dirname(__file__), "eval_questions.json")) as f:
        all_questions = {q["id"]: q for q in json.load(f)}

    report_lines = []
    for qid in TRACE_QUESTION_IDS:
        q = all_questions[qid]
        answer, chunks = answer_question(q["question"], k=5)

        report_lines.append(f"\n{'='*80}\nQ{qid} [{q['category']}]: {q['question']}\n")
        report_lines.append("--- RETRIEVED CONTEXT ---")
        for c in chunks:
            report_lines.append(f"  {c['file']} (lines {c['start_line']}-{c['end_line']})")
        report_lines.append("\n--- ANSWER ---")
        report_lines.append(answer)
        report_lines.append(
            "\n--- YOUR ANALYSIS (fill in) ---\n"
            "Relevant retrieval? [ ]  Irrelevant chunk present? [ ]  "
            "Missing info? [ ]  Hallucinated despite context? [ ]\n"
        )

    output = "\n".join(report_lines)
    with open("rag_trace_report.txt", "w") as f:
        f.write(output)
    print("Wrote rag_trace_report.txt -- open it and annotate each question by hand.")


if __name__ == "__main__":
    run_trace()
