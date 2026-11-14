from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import api
from src.chunking import Chunk, chunk_document
from src.loaders import load_documents
from src.pipeline import RAGPipeline
from src.vectorstore import VectorStore


def test_api_data_directories_are_under_project_root():
    project_root = Path(api.__file__).resolve().parents[1]

    assert api.PROJECT_ROOT == project_root
    assert api.DOCS_DIR == project_root / "data" / "docs"
    assert api.CHROMA_DIR == project_root / "data" / "chroma"


class RecordingStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = None

    def upsert(self, chunks, embeddings=None):
        self.chunks = chunks
        self.embeddings = embeddings


class RecordingCollection:
    def __init__(self):
        self.upsert_args = None

    def upsert(self, **kwargs):
        self.upsert_args = kwargs


def test_load_documents_reads_markdown_and_text_recursively(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "guide.md").write_text("# Guide", encoding="utf-8")
    (tmp_path / "nested" / "notes.TXT").write_text("Notes", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"not text")

    documents = load_documents(str(tmp_path))

    assert [(document.source, document.text) for document in documents] == [
        ("guide.md", "# Guide"),
        ("nested/notes.TXT", "Notes"),
    ]


def test_load_documents_rejects_missing_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_documents(str(tmp_path / "missing"))


def test_chunk_document_overlaps_adjacent_chunks():
    chunks = chunk_document("guide.md", "abcdefghij", chunk_size=6, overlap=2)

    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij"]
    assert [chunk.index for chunk in chunks] == [0, 1]


def test_pipeline_ingests_chunks_and_uses_chroma_embeddings(tmp_path: Path):
    (tmp_path / "guide.md").write_text("A document.", encoding="utf-8")
    store = RecordingStore()
    pipeline = RAGPipeline(embedder=None, store=store, llm=None)

    count = pipeline.ingest(str(tmp_path))

    assert count == 1
    assert store.chunks == [Chunk(text="A document.", source="guide.md", index=0)]
    assert store.embeddings is None


def test_vector_store_upserts_stable_ids_and_metadata():
    collection = RecordingCollection()
    store = VectorStore(collection)
    chunks = [Chunk(text="A document.", source="guide.md", index=0)]

    store.upsert(chunks)

    assert collection.upsert_args is not None
    assert collection.upsert_args["documents"] == ["A document."]
    assert collection.upsert_args["metadatas"] == [
        {"source": "guide.md", "chunk_index": "0"}
    ]
    assert len(collection.upsert_args["ids"][0]) == 64
    assert "embeddings" not in collection.upsert_args


def test_ingest_endpoint_persists_docs_and_returns_chunk_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "guide.md").write_text("A document.", encoding="utf-8")
    store = RecordingStore()
    pipeline = RAGPipeline(embedder=None, store=store, llm=None)
    monkeypatch.setattr(api, "DOCS_DIR", tmp_path)
    monkeypatch.setitem(api.app.dependency_overrides, api.get_pipeline, lambda: pipeline)

    response = TestClient(api.app).post("/ingest")

    assert response.status_code == 200
    assert response.json() == {"chunks_ingested": 1}
    assert store.chunks[0].text == "A document."


def test_ingest_endpoint_returns_not_found_for_missing_docs_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = RecordingStore()
    pipeline = RAGPipeline(embedder=None, store=store, llm=None)
    monkeypatch.setattr(api, "DOCS_DIR", tmp_path / "missing")
    monkeypatch.setitem(api.app.dependency_overrides, api.get_pipeline, lambda: pipeline)

    response = TestClient(api.app).post("/ingest")

    assert response.status_code == 404
    assert "Document directory does not exist" in response.json()["detail"]
