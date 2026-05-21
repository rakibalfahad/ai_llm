"""
RAG Pipeline — retrieves relevant code/doc chunks from ChromaDB to augment
the user's query before it is sent to the LLM.
"""
import logging
import os
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Wraps two ChromaDB collections:
      - 'codebase'   : indexed source files (.py, .ipynb, .sh, ...)
      - 'enterprise' : enterprise documents (PDFs, Markdown, text)

    Both are queried simultaneously; the most relevant chunks are returned.
    """

    def __init__(self) -> None:
        logger.info("Initializing embedding model: %s on %s",
                    settings.embedding_model, settings.embedding_device)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("Connecting to ChromaDB at %s:%s",
                    settings.chromadb_host, settings.chromadb_port)

        self._client = chromadb.HttpClient(
            host=settings.chromadb_host,
            port=settings.chromadb_port,
            settings=ChromaSettings(
                chroma_client_auth_provider=(
                    "chromadb.auth.token_authn.TokenAuthClientProvider"
                ),
                chroma_client_auth_credentials=settings.chroma_token,
                anonymized_telemetry=False,
            ),
        )

        self._code_store = Chroma(
            client=self._client,
            collection_name=settings.chroma_code_collection,
            embedding_function=self.embeddings,
        )
        self._docs_store = Chroma(
            client=self._client,
            collection_name=settings.chroma_docs_collection,
            embedding_function=self.embeddings,
        )

    def _load_system_prompt(self) -> str:
        path = settings.system_prompt_file
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return (
            "You are an expert AI coding assistant. "
            "Use the provided context to answer questions about the codebase accurately. "
            "If the context does not contain the answer, say so clearly."
        )

    def retrieve(
        self,
        query: str,
        k: int = 0,
        collection: Optional[str] = None,
    ) -> list[dict]:
        """
        Returns a list of {'content': str, 'source': str, 'score': float}.
        Queries both collections unless `collection` is specified.
        """
        k = k or settings.rag_top_k
        results: list[dict] = []

        stores = {
            "codebase": self._code_store,
            "enterprise": self._docs_store,
        }
        if collection and collection in stores:
            stores = {collection: stores[collection]}

        for coll_name, store in stores.items():
            try:
                docs_with_scores = store.similarity_search_with_relevance_scores(
                    query, k=k
                )
                for doc, score in docs_with_scores:
                    if score >= settings.rag_score_threshold:
                        results.append(
                            {
                                "content": doc.page_content,
                                "source": doc.metadata.get("source", "unknown"),
                                "collection": coll_name,
                                "score": round(score, 4),
                            }
                        )
            except Exception as exc:
                logger.warning("ChromaDB query failed for collection %s: %s",
                               coll_name, exc)

        # Sort by score descending, keep top-k across all collections
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def build_augmented_messages(
        self,
        original_messages: list[dict],
        context_docs: list[dict],
        web_results: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Prepend retrieved context to the conversation as a system message
        (or augment an existing system message).

        context_docs — chunks from ChromaDB (local codebase / enterprise docs)
        web_results  — live results from web_search.py (optional)
        """
        system_prompt = self._load_system_prompt()
        sections: list[str] = []

        if context_docs:
            formatted_chunks = []
            for i, doc in enumerate(context_docs, 1):
                formatted_chunks.append(
                    f"[{i}] Source: {doc['source']} (score: {doc['score']})\n"
                    f"{doc['content']}"
                )
            context_block = "\n\n---\n\n".join(formatted_chunks)
            sections.append(
                "<retrieved_context>\n"
                + context_block
                + "\n</retrieved_context>\n\n"
                "Use the retrieved context above to answer the user's question. "
                "Cite source file paths when referencing specific code."
            )

        if web_results:
            formatted_web = []
            for i, r in enumerate(web_results, 1):
                formatted_web.append(
                    f"[W{i}] {r['title']}\n"
                    f"URL: {r['url']}\n"
                    f"{r['content']}"
                )
            web_block = "\n\n---\n\n".join(formatted_web)
            sections.append(
                "<web_search_results>\n"
                + web_block
                + "\n</web_search_results>\n\n"
                "The web search results above are from the live internet. "
                "Cite URLs when referencing web content. "
                "Prefer the retrieved_context (local codebase) over web results when both are relevant."
            )

        if sections:
            augmented_system = system_prompt + "\n\n" + "\n\n".join(sections)
        else:
            augmented_system = system_prompt

        # Rebuild messages: replace/insert system message at position 0
        messages = [m for m in original_messages if m.get("role") != "system"]
        return [{"role": "system", "content": augmented_system}] + messages

    def get_stats(self) -> dict:
        """Returns document counts per collection."""
        stats = {}
        for name, store in [
            ("codebase", self._code_store),
            ("enterprise", self._docs_store),
        ]:
            try:
                count = store._collection.count()
                stats[name] = {"document_count": count}
            except Exception as exc:
                stats[name] = {"error": str(exc)}
        return stats
