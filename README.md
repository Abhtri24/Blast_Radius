# Blast Radius — Week 3 & 4 Step-by-Step

Run everything below on your Oracle VM (Ubuntu, Ollama already installed).

## 0. One-time setup

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pull the small models you'll actually iterate with day-to-day
ollama pull nomic-embed-text
ollama pull codellama          # your baseline "required" model
ollama pull qwen2.5:1.5b       # fast, tiny, good for local dev loop
```

Pick a target repo to analyze — your own small project, or clone any small
public repo for the demo:
```bash
git clone https://github.com/<some-small-repo>.git target_repo
```

---

## WEEK 3

### Exercise 1 — Basic app talking to Ollama
Already wired up: `app.py`'s free-form question mode calls `rag_core.py`,
which calls Ollama directly. Sanity check it works before adding RAG:
```bash
python -c "import ollama; print(ollama.chat(model='codellama', messages=[{'role':'user','content':'hello'}]))"
```

### Exercise 2 — Knowledge base (chunking + embeddings)
```bash
python kb_builder.py ./target_repo
```
This walks the repo, splits files into ~40-line chunks, embeds each with
`nomic-embed-text`, and stores them in `./chroma_db`.

### Exercise 3 — Retrieval + RAG
```bash
streamlit run app.py
```
Open the URL it prints. Try "Ask a free-form question" first (e.g. "What
does this project do?") to see retrieval + generation working end to end.
Then try "Analyze diff" with `base_ref=main` and any branch/commit with
changes, to see the actual Blast Radius impact-analysis flow.

**Demo tip:** run the same question with and without RAG (comment out the
context injection in `generate_answer`) to show your class the difference —
that's literally what Exercise 3 asks you to demonstrate.

### Exercise 4 — Services + orchestration
The same logic, split into three services under `services/`:
- `retrieval_service` — owns the vector DB and embedding search
- `llm_service` — owns talking to the model (swappable backend)
- `app_service` — orchestrator; the only one the user/frontend calls

Run them individually to prove each is independent:
```bash
cd services/retrieval_service && uvicorn main:app --port 8001 &
cd services/llm_service && uvicorn main:app --port 8002 &
cd services/app_service && RETRIEVAL_URL=http://localhost:8001 LLM_URL=http://localhost:8002 uvicorn main:app --port 8000 &

curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What does this project do?"}'
```

### Exercise 5 — Dockerize
```bash
docker compose up --build
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What does this project do?"}'
```
By default this points at your host's Ollama via `host.docker.internal`
(the `extra_hosts` entry in `docker-compose.yml` makes that resolve on
Linux). To use Groq instead of local Ollama for speed, set before starting:
```bash
export LLM_BACKEND=groq
export GROQ_API_KEY=<your key from console.groq.com>
docker compose up --build
```

---

## WEEK 4

### Exercise 1-2 — Question set + models
1. Open `eval/eval_questions.json` and replace the `<FILE>`, `<FUNCTION>`,
   etc. placeholders with real names from `./target_repo`.
2. Pull your 3 comparison models if not already present:
   ```bash
   ollama pull codellama
   ollama pull starcoder2
   ollama pull qwen2.5:1.5b      # or another model of your choice
   ```

### Exercise 3-4 — Run the benchmark + analyze
```bash
cd eval
nohup python benchmark.py > benchmark.log 2>&1 &
```
This can take a while on CPU — that's expected, let it run in the
background (that's exactly the "batch it overnight" approach we discussed).
When it finishes, `benchmark_results.csv` has per-question latency, CPU%,
memory%, and token count for all 3 models, plus blank columns
(`correctness_1to5`, `hallucinated_claims`, `test_passed`) for you to fill
in by reviewing each answer. Once filled in, pivot the CSV (Excel/pandas)
by `model` to compare averages — that's your Exercise 4 analysis table.

### Exercise 5 — RAG pipeline trace
```bash
python eval/rag_trace.py
```
Produces `rag_trace_report.txt` with retrieved-context + answer for 8
representative questions, and blank checkboxes for you to annotate
(relevant/irrelevant retrieval, missed info, hallucination-despite-context).

### Exercise 6 — Repo-level understanding
```bash
python eval/repo_understanding_test.py
```
Edit the question list at the top of that file to reference real
files/functions in your repo first. Expect some weak answers — document
them, since the exercise is explicitly about finding this gap.

---

## What to say in your presentation
- Architecture diagram: `User -> app_service -> retrieval_service + llm_service -> Ollama/Groq`
- One slide showing an actual diff analyzed live (Week 3 Ex3 demo)
- One table from `benchmark_results.csv` (Week 4 Ex4)
- 2-3 annotated examples from `rag_trace_report.txt`, including at least
  one hallucination example (Week 4 Ex5)
- 1-2 examples from `repo_understanding_test.py`, including a failure case
  (Week 4 Ex6) — framed as "this is what Sourcegraph will help solve next"
