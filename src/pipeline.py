from __future__ import annotations

from .chunking import chunk_document
from .embeddings import Embedder
from .loaders import load_documents
from .llm import LLM
from .vectorstore import VectorStore


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder | None,
        store: VectorStore,
        llm: LLM | None,
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm

    def ingest(self, directory: str) -> int:
        """load -> chunk -> embed -> store"""
        documents = load_documents(directory)
        chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(document.source, document.text)
        ]
        if not chunks:
            return 0

        embeddings = (
            self.embedder.embed_documents([chunk.text for chunk in chunks])
            if self.embedder is not None
            else None
        )
        self.store.upsert(chunks, embeddings)
        return len(chunks)

    def ask(self, question: str) -> str:
        """embed query -> retrieve -> build prompt -> generate"""
        raise NotImplementedError
