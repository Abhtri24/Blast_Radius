"""
Week 3 - Exercises 1, 2 & 3, now with easier repo management:
- paste a GitHub URL or local path, watch chunking progress live
- switch between previously-indexed repos via dropdown
- get warned if a repo's git HEAD has moved since it was last indexed

Three tabs: App (orchestrated), Retrieval/RAG (search only), LLM (generate only).

Run with:
    streamlit run app.py
"""
import os
import requests
import streamlit as st
from repo_manager import index_repo, load_registry, is_stale

APP_SERVICE_URL = os.environ.get("APP_SERVICE_URL", "http://localhost:8000")
RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://localhost:8001")
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://localhost:8002")


def ask_service(question, collection_name, k=6):
    try:
        resp = requests.post(
            f"{APP_SERVICE_URL}/ask",
            json={"question": question, "k": k, "collection_name": collection_name},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("answer", ""), data.get("context_used", [])
    except Exception as e:
        st.error(f"App Service error: {e}")
        return f"Error: {e}", []


def search_service(query, collection_name, k=5):
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
        return {"answer": f"Error: {e}"}



st.set_page_config(page_title="Blast Radius", layout="wide")
st.title("🔎 Blast Radius — PR Impact Analyzer")

# ---------------- Sidebar: repo management (shared across all tabs) ----------------
with st.sidebar:
    st.header("Knowledge base")

    source = st.text_input("GitHub URL or local path", placeholder="https://github.com/org/repo or /home/you/myrepo")
    if st.button("Index / Update this repo", disabled=not source):
        progress_bar = st.progress(0, text="Starting...")
        status_line = st.empty()

        def on_progress(i, total, chunks_done, message):
            frac = i / total if total else 0
            progress_bar.progress(frac, text=f"{i}/{total} files — {chunks_done} chunks so far")
            status_line.caption(message)

        with st.spinner("Cloning (if needed) and indexing..."):
            result = index_repo(source, progress_callback=on_progress)

        progress_bar.progress(1.0, text="Done")
        st.success(f"Indexed {result['chunk_count']} chunks from '{result['repo_name']}'")
        st.session_state["active_repo"] = result["repo_name"]

    st.divider()

    registry = load_registry()
    if registry:
        repo_names = list(registry.keys())
        default_idx = repo_names.index(st.session_state["active_repo"]) if st.session_state.get("active_repo") in repo_names else 0
        active_repo = st.selectbox("Active repo", repo_names, index=default_idx)
        st.session_state["active_repo"] = active_repo

        entry = registry[active_repo]
        st.caption(f"Path: `{entry['path']}`")
        st.caption(f"Chunks: {entry['chunk_count']} | Indexed: {entry['indexed_at']}")

        stale = is_stale(active_repo)
        if stale:
            st.warning("⚠️ This repo has new commits since it was last indexed. Click 'Index / Update' above to refresh.")
        elif stale is False:
            st.success("✅ Knowledge base is up to date with HEAD.")
    else:
        st.info("No repos indexed yet — paste a URL or path above to get started.")

active_repo = st.session_state.get("active_repo")
active_entry = load_registry().get(active_repo) if active_repo else None

tab_app, tab_retrieval, tab_llm = st.tabs(["App", "Retrieval / RAG", "LLM"])

# ---------------- Tab 1: App (orchestrated via app_service) ----------------
with tab_app:
    if not active_entry:
        st.info("👈 Index a repo in the sidebar first.")
    else:
        collection_name = active_entry["collection"]
        repo_path = active_entry["path"]

        base_ref = st.text_input("Base ref", value="main", key="app_base_ref")
        compare_ref = st.text_input("Compare ref", value="HEAD", key="app_compare_ref")
        mode = st.radio("Mode", ["diff", "chat"], horizontal=True, key="app_mode")

        if mode == "diff":
            import git
            repo = git.Repo(repo_path)
            diffs = repo.git.diff(base_ref, compare_ref)

            if diffs:
                st.subheader("Raw diff")
                st.code(diffs, language="diff")

                changed_files = repo.git.diff(base_ref, compare_ref, "--name-only").splitlines()
                st.subheader("Impact analysis per changed file")
                for f in changed_files:
                    with st.spinner(f"Analyzing {f}..."):
                        question = (
                            f"The file '{f}' was changed. Based on the codebase, what other files, "
                            f"functions, or components might be affected by this change, and is there "
                            f"any test coverage related to it? Be specific and say 'unknown' if the "
                            f"context does not show it."
                        )
                        answer, chunks = ask_service(question, collection_name, k=6)
                    st.markdown(f"**{f}**")
                    st.write(answer)
                    with st.expander("Retrieved context used"):
                        for c in chunks:
                            st.caption(f"{c['file']} (lines {c['start_line']}-{c['end_line']})")
                            st.code(c["text"])
            else:
                st.warning("No diff found between the two refs, or repo path is invalid.")

        if mode == "chat":
            q = st.text_input("Your question about the codebase", key="app_chat_q")
            if q:
                with st.spinner("Thinking..."):
                    answer, chunks = ask_service(q, collection_name)
                st.write(answer)
                with st.expander("Retrieved context used"):
                    for c in chunks:
                        st.caption(f"{c['file']} (lines {c['start_line']}-{c['end_line']})")
                        st.code(c["text"])

# ---------------- Tab 2: Retrieval / RAG (retrieval_service only, no LLM) ----------------
with tab_retrieval:
    st.caption("Talks only to retrieval_service — no LLM generation, just raw retrieved chunks.")
    if not active_entry:
        st.info("👈 Index a repo in the sidebar first.")
    else:
        collection_name = active_entry["collection"]
        query = st.text_input("Search query", key="retrieval_query")
        k = st.slider("Number of results (k)", min_value=1, max_value=15, value=5, key="retrieval_k")
        if query:
            with st.spinner("Searching..."):
                chunks = search_service(query, collection_name, k=k)
            st.write(f"{len(chunks)} chunks retrieved")
            for c in chunks:
                st.caption(f"{c['file']} (lines {c['start_line']}-{c['end_line']})")
                st.code(c["text"])

# ---------------- Tab 3: LLM (llm_service only, no retrieval) ----------------
with tab_llm:
    st.caption("Talks only to llm_service — raw prompt in, raw model response out. No retrieval or context.")
    prompt = st.text_area("Prompt", key="llm_prompt")
    if st.button("Generate", disabled=not prompt, key="llm_generate_btn"):
        with st.spinner("Generating..."):
            result = generate_service(prompt)
        st.write(result["answer"])
        st.caption(f"Backend: {result['backend']}")
