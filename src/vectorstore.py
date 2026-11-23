from __future__ import annotations

from hashlib import sha256
from typing import Any, Protocol

from .chunking import Chunk


class ChromaCollection(Protocol):
    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, str]],
        embeddings: list[list[float]] | None = None,
    ) -> None: ...

    def query(
        self,
        *,
        query_texts: list[str],
        n_results: int,
        include: list[str],
    ) -> dict[str, Any]: ...


class VectorStore:
    def __init__(self, collection: ChromaCollection):
        self.collection = collection

    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        ids = [
            sha256(f"{chunk.source}:{chunk.index}".encode()).hexdigest()
            for chunk in chunks
        ]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {"source": chunk.source, "chunk_index": str(chunk.index)}
            for chunk in chunks
        ]
        if embeddings is None:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        else:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

    def query(self, text: str, k: int) -> list[Chunk]:
        if not text.strip():
            raise ValueError("query text must not be empty")
        if k <= 0:
            raise ValueError("k must be greater than zero")

        results = self.collection.query(
            query_texts=[text],
            n_results=k,
            include=["documents", "metadatas"],
        )
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        if len(documents) != len(metadatas):
            raise ValueError("ChromaDB returned mismatched documents and metadata")

        chunks = []
        for document, metadata in zip(documents, metadatas):
            if document is None or metadata is None:
                raise ValueError("ChromaDB returned a result without document metadata")
            chunks.append(
                Chunk(
                    text=document,
                    source=metadata["source"],
                    index=int(metadata["chunk_index"]),
                )
            )
        return chunks
