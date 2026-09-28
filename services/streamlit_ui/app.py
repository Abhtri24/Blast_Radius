"""
Blast Radius — PR Impact Analyzer UI

Four tabs:
  1. App          — orchestrated RAG (diff + chat modes) via app_service /ask
  2. Code Graph   — RAG + Sourcegraph structural search via app_service /ask_with_graph
  3. Retrieval    — raw vector search via retrieval_service /search (no LLM)
  4. LLM          — raw prompt via llm_service /generate (no retrieval)

Run with:
    streamlit run app.py
"""
import os
import json
import requests
import streamlit as st
from repo_manager import index_repo, load_registry, is_stale

# --------------- Service URLs ---------------
APP_SERVICE_URL = os.environ.get("APP_SERVICE_URL", "http://localhost:8000")
RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://localhost:8001")
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://localhost:8002")

# --------------- Groq model list ---------------
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]


# --------------- Service helpers ---------------
def ask_service(question, collection_name, k=6, model=None):
    """Call app_service /ask — standard RAG."""
    payload = {"question": question, "k": k, "collection_name": collection_name}
    if model:
        payload["model"] = model
    try:
        resp = requests.post(f"{APP_SERVICE_URL}/ask", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"App Service error: {e}")
        return {"answer": f"Service Error: {str(e)}", "context_used": [], "model": None}


def ask_with_graph_service(question, collection_name, k=6, model=None):
    """Call app_service /ask_with_graph — RAG + Sourcegraph."""
    payload = {"question": question, "k": k, "collection_name": collection_name}
    if model:
        payload["model"] = model
    try:
        resp = requests.post(f"{APP_SERVICE_URL}/ask_with_graph", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"App Service Graph error: {e}")
        return {"answer": f"Service Error: {str(e)}", "context_used": [], "graph_hits": [], "model": None}


def search_service(query, collection_name, k=5):
    """Call retrieval_service /search — vector search only."""
    try:
        resp = requests.post(
            f"{RETRIEVAL_SERVICE_URL}/search",
            json={"query": query, "k": k, "collection_name": collection_name},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("chunks", [])
    except Exception as e:
        st.error(f"Retrieval Service error: {e}")
        return []


def generate_service(prompt):
    """Call llm_service /generate — raw LLM."""
    try:
        resp = requests.post(
            f"{LLM_SERVICE_URL}/generate",
            json={"prompt": prompt},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"LLM Service error: {e}")
        return {"answer": f"Service Error: {str(e)}"}



# --------------- Custom CSS ---------------
def inject_custom_css():
    st.markdown("""
    <style>
    /* ---------- Global typography ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---------- Answer card ---------- */
    .answer-card {
        background: linear-gradient(135deg, #1e2230 0%, #252a3a 100%);
        border: 1px solid #6C63FF33;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0 1rem 0;
        line-height: 1.65;
        color: #E8E8E8;
    }
    .answer-card code { color: #A78BFA; }

    /* ---------- Guardrail badge ---------- */
    .guardrail-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .guardrail-pass {
        background: #065F4620;
        color: #34D399;
        border: 1px solid #34D39940;
    }
    .guardrail-refuse {
        background: #7C3AED20;
        color: #C4B5FD;
        border: 1px solid #C4B5FD40;
    }

    /* ---------- Metric pills ---------- */
    .metric-row {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin: 0.5rem 0 1rem 0;
    }
    .metric-pill {
        background: #1A1D2990;
        border: 1px solid #ffffff12;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.8rem;
        color: #B0B8C8;
    }
    .metric-pill strong { color: #E0E0E0; }

    /* ---------- Tab label tweaks ---------- */
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom: 2px solid #6C63FF;
    }

    /* ---------- Sidebar polish ---------- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #ffffff08;
    }

    /* ---------- Empty state ---------- */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #888;
        font-size: 1.05rem;
    }
    .empty-state .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)


# --------------- Reusable display helpers ---------------
def render_answer_card(answer):
    """Display the LLM answer in a styled card."""
    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)


def render_guardrail_badge(guardrail_triggered, best_distance=None):
    """Show a small badge indicating guardrail status."""
    if guardrail_triggered:
        dist_str = f" — best distance: {best_distance:.1f}" if best_distance else ""
        st.markdown(
            f'<span class="guardrail-badge guardrail-refuse">'
            f'⛔ Guardrail: {guardrail_triggered}{dist_str}</span>',
            unsafe_allow_html=True,
        )
    else:
        dist_str = f" — best distance: {best_distance:.1f}" if best_distance else ""
        st.markdown(
            f'<span class="guardrail-badge guardrail-pass">'
            f'✅ Guardrail: passed{dist_str}</span>',
            unsafe_allow_html=True,
        )


def render_metrics(data):
    """Show model/token metrics as pills."""
    pills = []
    if data.get("model"):
        pills.append(f"<strong>Model:</strong> {data['model']}")
    if data.get("prompt_tokens") is not None:
        pills.append(f"<strong>Prompt:</strong> {data['prompt_tokens']} tokens")
    if data.get("completion_tokens") is not None:
        pills.append(f"<strong>Completion:</strong> {data['completion_tokens']} tokens")
    if pills:
        inner = "".join(f'<span class="metric-pill">{p}</span>' for p in pills)
        st.markdown(f'<div class="metric-row">{inner}</div>', unsafe_allow_html=True)


def render_context_chunks(chunks, label="Retrieved context (ChromaDB)"):
    """Show retrieved chunks in an expander."""
    with st.expander(f"📄 {label} — {len(chunks)} chunk(s)"):
        if not chunks:
            st.caption("No chunks retrieved.")
            return
        for c in chunks:
            dist_info = f"  |  distance: {c['distance']:.1f}" if c.get("distance") is not None else ""
            st.caption(f"`{c['file']}` lines {c['start_line']}–{c['end_line']}{dist_info}")
            st.code(c["text"], language="javascript")


def render_graph_hits(graph_hits, filenames_searched):
    """Show Sourcegraph hits in a separate expander."""
    label = f"🔗 Sourcegraph hits — {len(graph_hits)} result(s)"
    if filenames_searched:
        label += f" for {', '.join(filenames_searched)}"
    with st.expander(label):
        if not graph_hits:
            st.caption("No filenames detected in the question, or no cross-file references found.")
            return
        for h in graph_hits:
            if "error" in h:
                st.error(f"Sourcegraph error: {h['error']}")
            else:
                st.caption(f"`{h['file']}` line {h['line']}")
                st.code(h.get("preview", ""), language="javascript")


def render_empty_state(icon, message):
    """Display a centered empty-state placeholder."""
    st.markdown(
        f'<div class="empty-state"><div class="icon">{icon}</div>{message}</div>',
        unsafe_allow_html=True,
    )


# --------------- Page config ---------------
st.set_page_config(page_title="Blast Radius", layout="wide", page_icon="💥")
inject_custom_css()
st.title("💥 Blast Radius")
st.caption("RAG-powered code Q&A and PR impact analysis")

# --------------- Sidebar: repo management ---------------
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.divider()

    source = st.text_input(
        "GitHub URL or local path",
        placeholder="https://github.com/org/repo",
        help="Paste a GitHub URL or absolute local path to a git repository.",
    )
    if st.button("🔄 Index / Update", disabled=not source, use_container_width=True):
        progress_bar = st.progress(0, text="Starting...")
        status_line = st.empty()

        def on_progress(i, total, chunks_done, message):
            frac = i / total if total else 0
            progress_bar.progress(frac, text=f"{i}/{total} files — {chunks_done} chunks")
            status_line.caption(message)

        with st.spinner("Cloning (if needed) and indexing..."):
            result = index_repo(source, progress_callback=on_progress)

        progress_bar.progress(1.0, text="Done")
        st.success(f"✅ Indexed **{result['chunk_count']}** chunks from `{result['repo_name']}`")
        st.session_state["active_repo"] = result["repo_name"]

    st.divider()

    registry = load_registry()
    if registry:
        repo_names = list(registry.keys())
        default_idx = (
            repo_names.index(st.session_state["active_repo"])
            if st.session_state.get("active_repo") in repo_names
            else 0
        )
        active_repo = st.selectbox("Active repo", repo_names, index=default_idx)
        st.session_state["active_repo"] = active_repo

        entry = registry[active_repo]
        st.caption(f"📂 `{entry['path']}`")
        st.caption(f"🧩 {entry['chunk_count']} chunks  •  🕑 {entry['indexed_at']}")

        stale = is_stale(active_repo)
        if stale:
            st.warning("⚠️ Repo has new commits since last index. Re-index to refresh.")
        elif stale is False:
            st.success("✅ Up to date with HEAD")
    else:
        st.info("No repos indexed yet — add one above to get started.")

    st.divider()
    st.caption("Blast Radius v2 • Groq-powered")

# --------------- Active repo state ---------------
active_repo = st.session_state.get("active_repo")
active_entry = load_registry().get(active_repo) if active_repo else None

# --------------- Tabs ---------------
tab_app, tab_graph, tab_retrieval, tab_llm = st.tabs([
    "🚀 App", "🔗 Code Graph", "🔍 Retrieval", "🤖 LLM"
])


# ======================== Tab 1: App ========================
with tab_app:
    st.caption(
        "**Orchestrated RAG** — retrieves relevant code chunks, applies a confidence guardrail, "
        "then generates an answer via the LLM. Supports diff-based impact analysis and free-form chat."
    )
    st.divider()

    if not active_entry:
        render_empty_state("📂", "Index a repository in the sidebar to get started.")
    else:
        collection_name = active_entry["collection"]
        repo_path = active_entry["path"]

        mode = st.radio("Mode", ["💬 Chat", "📊 Diff Analysis"], horizontal=True, key="app_mode")

        if mode == "📊 Diff Analysis":
            col1, col2 = st.columns(2)
            with col1:
                base_ref = st.text_input("Base ref", value="main", key="app_base_ref")
            with col2:
                compare_ref = st.text_input("Compare ref", value="HEAD", key="app_compare_ref")

            import git
            try:
                repo = git.Repo(repo_path)
                diffs = repo.git.diff(base_ref, compare_ref)
            except Exception as e:
                st.error(f"Git error: {e}")
                diffs = None

            if diffs:
                with st.expander("📝 Raw diff", expanded=False):
                    st.code(diffs, language="diff")

                changed_files = repo.git.diff(base_ref, compare_ref, "--name-only").splitlines()
                st.subheader(f"Impact analysis — {len(changed_files)} file(s) changed")

                if not st.button("▶️ Run impact analysis", key="app_run_impact"):
                    changed_files = []

                for f in changed_files:
                    with st.spinner(f"Analyzing `{f}`..."):
                        question = (
                            f"The file '{f}' was changed. Based on the codebase, what other files, "
                            f"functions, or components might be affected by this change, and is there "
                            f"any test coverage related to it? Be specific and say 'unknown' if the "
                            f"context does not show it."
                        )
                        data = ask_service(question, collection_name, k=6)

                    with st.container(border=True):
                        st.markdown(f"**📄 {f}**")
                        render_answer_card(data["answer"])
                        render_guardrail_badge(
                            data.get("guardrail_triggered"),
                            data.get("best_distance"),
                        )
                        render_context_chunks(data.get("context_used", []))
            elif diffs is not None:
                st.info("No diff found between the two refs.")

        if mode == "💬 Chat":
            model_choice = st.selectbox("Model", GROQ_MODELS, key="app_chat_model")
            q = st.text_input("Your question about the codebase", key="app_chat_q")

            if st.button("🚀 Send", key="app_chat_send", disabled=not q):
                with st.spinner(f"Thinking ({model_choice})..."):
                    data = ask_service(q, collection_name, model=model_choice)

                with st.container(border=True):
                    render_answer_card(data["answer"])
                    render_guardrail_badge(
                        data.get("guardrail_triggered"),
                        data.get("best_distance"),
                    )
                    render_metrics(data)
                render_context_chunks(data.get("context_used", []))
            else:
                render_empty_state("💬", "Ask a question to get started.")


# ======================== Tab 2: Code Graph ========================
with tab_graph:
    st.caption(
        "**RAG + Sourcegraph** — same as App tab, but also queries Sourcegraph for structural "
        "cross-file references. Best for dependency/usage questions (e.g. 'Is X used anywhere?')."
    )
    st.divider()

    if not active_entry:
        render_empty_state("📂", "Index a repository in the sidebar to get started.")
    else:
        collection_name = active_entry["collection"]

        model_choice_graph = st.selectbox("Model", GROQ_MODELS, key="graph_model")
        q_graph = st.text_input(
            "Your question (mention filenames for Sourcegraph lookup)",
            key="graph_q",
            placeholder="e.g. Is comment.model.js used anywhere in the codebase?",
        )

        if st.button("🔗 Send with Code Graph", key="graph_send", disabled=not q_graph):
            with st.spinner(f"Searching code graph + RAG ({model_choice_graph})..."):
                data = ask_with_graph_service(q_graph, collection_name, model=model_choice_graph)

            with st.container(border=True):
                render_answer_card(data["answer"])
                render_guardrail_badge(
                    data.get("guardrail_triggered"),
                    data.get("best_distance"),
                )
                render_metrics(data)

            # Side-by-side: Chroma context vs Sourcegraph hits
            col_chroma, col_sg = st.columns(2)
            with col_chroma:
                render_context_chunks(data.get("context_used", []))
            with col_sg:
                render_graph_hits(
                    data.get("graph_hits", []),
                    data.get("filenames_searched", []),
                )
        else:
            render_empty_state("🔗", "Ask a question to query both RAG and the code graph.")


# ======================== Tab 3: Retrieval ========================
with tab_retrieval:
    st.caption(
        "**Vector search only** — talks directly to retrieval_service. "
        "No LLM generation, just raw retrieved chunks with distance scores."
    )
    st.divider()

    if not active_entry:
        render_empty_state("📂", "Index a repository in the sidebar to get started.")
    else:
        collection_name = active_entry["collection"]
        query = st.text_input("Search query", key="retrieval_query")
        k = st.slider("Results (k)", min_value=1, max_value=15, value=5, key="retrieval_k")

        if st.button("🔍 Search", key="retrieval_search_btn", disabled=not query):
            with st.spinner("Searching..."):
                chunks = search_service(query, collection_name, k=k)

            st.success(f"Retrieved **{len(chunks)}** chunks")
            for c in chunks:
                with st.container(border=True):
                    dist_info = f"  •  distance: **{c['distance']:.1f}**" if c.get("distance") is not None else ""
                    st.caption(f"📄 `{c['file']}` lines {c['start_line']}–{c['end_line']}{dist_info}")
                    st.code(c["text"], language="javascript")
        else:
            render_empty_state("🔍", "Enter a query to search the vector store.")


# ======================== Tab 4: LLM ========================
with tab_llm:
    st.caption(
        "**Raw LLM** — talks directly to llm_service. "
        "No retrieval or context injection, just raw prompt → model response."
    )
    st.divider()

    prompt = st.text_area("Prompt", key="llm_prompt", height=150)
    if st.button("🤖 Generate", disabled=not prompt, key="llm_generate_btn"):
        with st.spinner("Generating..."):
            result = generate_service(prompt)

        with st.container(border=True):
            render_answer_card(result["answer"])
            render_metrics(result)
    else:
        render_empty_state("🤖", "Type a prompt and click Generate.")
