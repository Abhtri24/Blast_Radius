"""
Manages a registry of indexed repos so the UI can:
- accept a GitHub URL OR a local path, clone if needed
- chunk+embed with progress feedback
- detect when a previously-indexed repo has new commits (stale KB)
- offer a dropdown of already-indexed repos instead of re-typing paths
"""
import os
import re
import json
import time
import chromadb
import ollama
import git

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_client = ollama.Client(host=OLLAMA_HOST)

EMBED_MODEL = "nomic-embed-text"
CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".md"}
CHUNK_LINES = 40
OVERLAP_LINES = 8
DB_PATH = "./chroma_db"
CLONE_DIR = "./repos"
REGISTRY_PATH = "./repo_registry.json"


def _sanitize_collection_name(name):
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_-")
    return (name or "repo")[:60]


def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def resolve_repo_path(source):
    """If `source` looks like a URL, clone it (or reuse existing clone). Otherwise treat as a local path."""
    if source.startswith("github.com/") or source.startswith("gitlab.com/"):
        source = "https://" + source
    if source.startswith("http://") or source.startswith("https://") or source.startswith("git@"):
        repo_name = source.rstrip("/").split("/")[-1].removesuffix(".git")
        dest = os.path.join(CLONE_DIR, repo_name)
        if os.path.exists(dest):
            return dest, repo_name
        os.makedirs(CLONE_DIR, exist_ok=True)
        git.Repo.clone_from(source, dest)
        return dest, repo_name
    else:
        repo_name = os.path.basename(os.path.normpath(source))
        return source, repo_name


def get_head_sha(repo_path):
    try:
        return git.Repo(repo_path).head.commit.hexsha
    except Exception:
        return None


def iter_code_files(repo_path):
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "__pycache__", "venv", ".venv")]
        for f in files:
            if os.path.splitext(f)[1] in CODE_EXTENSIONS:
                yield os.path.join(root, f)


def chunk_file(path):
    with open(path, "r", errors="ignore") as fh:
        lines = fh.readlines()
    chunks = []
    step = CHUNK_LINES - OVERLAP_LINES
    for start in range(0, max(len(lines), 1), step):
        end = min(start + CHUNK_LINES, len(lines))
        text = "".join(lines[start:end]).strip()
        if text:
            chunks.append((text, start + 1, end))
        if end == len(lines):
            break
    return chunks


def index_repo(source, progress_callback=None):
    """
    source: a GitHub URL or a local path.
    progress_callback(current_file_i, total_files, chunks_done, message): called as work proceeds.
    Returns: dict with repo_name, collection, chunk_count, head_sha.
    """
    repo_path, repo_name = resolve_repo_path(source)
    collection_name = _sanitize_collection_name(repo_name)
    head_sha = get_head_sha(repo_path)

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    files = list(iter_code_files(repo_path))
    total_files = len(files)
    doc_id = 0
    for i, filepath in enumerate(files, 1):
        rel_path = os.path.relpath(filepath, repo_path)
        for text, start_line, end_line in chunk_file(filepath):
            emb = _client.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
            collection.add(
                ids=[str(doc_id)],
                embeddings=[emb],
                documents=[text],
                metadatas=[{"file": rel_path, "start_line": start_line, "end_line": end_line}],
            )
            doc_id += 1
        if progress_callback:
            progress_callback(i, total_files, doc_id, f"Chunked {rel_path}")

    registry = load_registry()
    registry[repo_name] = {
        "path": repo_path,
        "source": source,
        "collection": collection_name,
        "head_sha": head_sha,
        "chunk_count": doc_id,
        "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_registry(registry)

    return {"repo_name": repo_name, "path": repo_path, "collection": collection_name,
            "chunk_count": doc_id, "head_sha": head_sha}


def is_stale(repo_name):
    """True if the repo's current HEAD differs from what was last indexed."""
    registry = load_registry()
    entry = registry.get(repo_name)
    if not entry:
        return None  # unknown repo, not applicable
    current_sha = get_head_sha(entry["path"])
    return current_sha != entry["head_sha"]
