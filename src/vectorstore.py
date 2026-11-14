from __future__ import annotations

from hashlib import sha256
from typing import Protocol

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


class VectorStore:
    def __init__(self, collection: ChromaCollection):
        self.collection = collection

    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        ids = [
            sha256(f"{chunk.source}:{chunk.index}".encode("utf-8")).hexdigest()
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

    def query(self, embedding: list[float], k: int) -> list[Chunk]:
        raise NotImplementedError
