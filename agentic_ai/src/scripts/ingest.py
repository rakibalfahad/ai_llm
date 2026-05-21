"""
Ingest script — loads code and documents into ChromaDB collections.

Usage (inside the rag-api container or with Docker exec):
  python3 /app/scripts/ingest.py --source /app/data/enterprise_docs --collection enterprise
  python3 /app/scripts/ingest.py --source /path/to/code --collection codebase

Collections:
  codebase   — Python, JS, TS, shell, notebooks, Markdown
  enterprise — PDFs, Markdown, text files, Word docs
"""
import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import (
    DirectoryLoader,
    NotebookLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Extension → LangChain Language mapping ────────────────────────────────────
LANG_MAP: dict[str, Language] = {
    ".py":   Language.PYTHON,
    ".js":   Language.JS,
    ".ts":   Language.TS,
    ".java": Language.JAVA,
    ".cpp":  Language.CPP,
    ".c":    Language.C,
    ".go":   Language.GO,
    ".rs":   Language.RUST,
    ".rb":   Language.RUBY,
    ".cs":   Language.CSHARP,
}

# Extensions to skip
SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico",
    ".pt", ".pth", ".bin", ".safetensors", ".gguf",
    ".zip", ".tar", ".gz", ".bz2", ".lock",
    ".git",
}


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_file(path: Path) -> list[Document]:
    """Load a single file into a list of Documents."""
    ext = path.suffix.lower()

    if ext in SKIP_EXTENSIONS:
        return []

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
            return loader.load()

        if ext == ".ipynb":
            loader = NotebookLoader(
                str(path),
                include_outputs=False,
                max_output_length=100,
                remove_newline=True,
            )
            return loader.load()

        # Default: text loader for all other formats
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
        return loader.load()

    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return []


def split_documents(
    docs: list[Document],
    file_ext: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents using language-aware splitters when possible."""
    if not docs:
        return []

    lang = LANG_MAP.get(file_ext.lower())

    if lang:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    return splitter.split_documents(docs)


def ingest(
    source_dir: str,
    collection_name: str,
    chromadb_host: str,
    chromadb_port: int,
    chroma_token: str,
    embedding_model: str,
    embedding_device: str,
    chunk_size: int,
    chunk_overlap: int,
    clear_first: bool,
) -> None:
    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error("Source directory does not exist: %s", source_dir)
        sys.exit(1)

    logger.info("Embedding model: %s (%s)", embedding_model, embedding_device)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Connecting to ChromaDB at %s:%s", chromadb_host, chromadb_port)
    client = chromadb.HttpClient(
        host=chromadb_host,
        port=chromadb_port,
        settings=ChromaSettings(
            chroma_client_auth_provider=(
                "chromadb.auth.token_authn.TokenAuthClientProvider"
            ),
            chroma_client_auth_credentials=chroma_token,
            anonymized_telemetry=False,
        ),
    )

    if clear_first:
        logger.warning("Clearing existing collection: %s", collection_name)
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    # ── Walk the source directory ─────────────────────────────────────────────
    total_chunks = 0
    total_files = 0
    skipped = 0

    for file_path in sorted(source_path.rglob("*")):
        if not file_path.is_file():
            continue
        if any(part.startswith(".") for part in file_path.parts):
            continue   # skip hidden dirs (.git, .venv, etc.)
        if "__pycache__" in file_path.parts:
            continue

        docs = load_file(file_path)
        if not docs:
            skipped += 1
            continue

        chunks = split_documents(docs, file_path.suffix, chunk_size, chunk_overlap)
        if not chunks:
            skipped += 1
            continue

        # Annotate metadata
        for chunk in chunks:
            chunk.metadata["source"] = str(file_path)
            chunk.metadata["filename"] = file_path.name
            chunk.metadata["extension"] = file_path.suffix.lower()
            chunk.metadata["collection"] = collection_name
            chunk.metadata["content_hash"] = content_hash(chunk.page_content)

        # Unique IDs based on file path + chunk content
        ids = [
            f"{collection_name}:{file_path}:{i}:{chunk.metadata['content_hash']}"
            for i, chunk in enumerate(chunks)
        ]

        try:
            vector_store.add_documents(chunks, ids=ids)
            total_chunks += len(chunks)
            total_files += 1
            logger.info("  [%d chunks] %s", len(chunks), file_path)
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", file_path, exc)

    # ── Summary ───────────────────────────────────────────────────────────────
    final_count = vector_store._collection.count()
    logger.info(
        "\nIngestion complete: %d files → %d new chunks "
        "(collection total: %d, skipped: %d)",
        total_files, total_chunks, final_count, skipped,
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest files into ChromaDB for RAG")
    parser.add_argument("--source", required=True, help="Directory to ingest")
    parser.add_argument(
        "--collection",
        default="codebase",
        choices=["codebase", "enterprise"],
        help="Target ChromaDB collection",
    )
    parser.add_argument("--chromadb-host", default=os.getenv("CHROMADB_HOST", "chromadb"))
    parser.add_argument("--chromadb-port", type=int, default=int(os.getenv("CHROMADB_PORT", "8000")))
    parser.add_argument("--chroma-token", default=os.getenv("CHROMA_TOKEN", ""))
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    )
    parser.add_argument(
        "--embedding-device",
        default=os.getenv("EMBEDDING_DEVICE", "cpu"),
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing collection before ingesting",
    )
    args = parser.parse_args()

    ingest(
        source_dir=args.source,
        collection_name=args.collection,
        chromadb_host=args.chromadb_host,
        chromadb_port=args.chromadb_port,
        chroma_token=args.chroma_token,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        clear_first=args.clear,
    )


if __name__ == "__main__":
    main()
