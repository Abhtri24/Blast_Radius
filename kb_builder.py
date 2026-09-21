"""
Week 3 - Exercise 2: Knowledge Base (chunking + embeddings)

Walks a target repo, splits code files into overlapping line-window chunks,
embeds each chunk with an Ollama embedding model, and stores them in a
local Chroma collection.

Usage:
    python kb_builder.py /path/to/target/repo
"""
import sys
import os
import chromadb
import ollama

# Point this at a Colab/remote Ollama instance to use its GPU instead of
# your VM's CPU, e.g.: export OLLAMA_HOST=https://<your-tunnel>.trycloudflare.com
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_client = ollama.Client(host=OLLAMA_HOST)

EMBED_MODEL = "nomic-embed-text"   # small, run: `ollama pull nomic-embed-text`
CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".md"}
CHUNK_LINES = 40
OVERLAP_LINES = 8
DB_PATH = "./chroma_db"
COLLECTION_NAME = "repo_chunks"


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


def build_knowledge_base(repo_path):
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    doc_id = 0
    total_chunks = 0
    for filepath in iter_code_files(repo_path):
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
            total_chunks += 1

    print(f"Indexed {total_chunks} chunks from {repo_path} into '{COLLECTION_NAME}' at {DB_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python kb_builder.py /path/to/target/repo")
        sys.exit(1)
    build_knowledge_base(sys.argv[1])
