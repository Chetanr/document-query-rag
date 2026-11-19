from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src import api
from src.chunking import Chunk, chunk_document
from src.llm import AnthropicLLM, LLMConfigurationError
from src.loaders import load_documents
from src.pipeline import RAGPipeline
from src.vectorstore import VectorStore


def test_anthropic_llm_uses_configured_model_and_returns_text(
    monkeypatch: pytest.MonkeyPatch,
):
    class RecordingMessages:
        def __init__(self):
            self.arguments = None

        def create(self, **kwargs):
            self.arguments = kwargs
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="A generated answer.")]
            )

    messages = RecordingMessages()
    client = SimpleNamespace(messages=messages)
    monkeypatch.setattr("src.llm.Anthropic", lambda api_key: client)
    llm = AnthropicLLM(model="configured-model-id", api_key="test-key")

    answer = llm.generate(system="System", user="Question")

    assert answer == "A generated answer."
    assert messages.arguments == {
        "model": "configured-model-id",
        "max_tokens": 1024,
        "system": "System",
        "messages": [{"role": "user", "content": "Question"}],
    }


def test_anthropic_llm_requires_model_configuration():
    llm = AnthropicLLM(model=None, api_key="test-key")

    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_MODEL"):
        llm.generate(system="System", user="Question")


def test_api_data_directories_are_under_project_root():
    project_root = Path(api.__file__).resolve().parents[1]

    assert api.PROJECT_ROOT == project_root
    assert api.DOCS_DIR == project_root / "data" / "docs"
    assert api.CHROMA_DIR == project_root / "data" / "chroma"


def test_pipeline_uses_configured_anthropic_model(
    monkeypatch: pytest.MonkeyPatch,
):
    class RecordingClient:
        def get_or_create_collection(self, name):
            return RecordingCollection()

    monkeypatch.setenv("ANTHROPIC_MODEL", "configured-model-id")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("TOXICITY_THRESHOLD", "0.5")
    monkeypatch.setattr(
        api.chromadb, "PersistentClient", lambda path: RecordingClient()
    )

    pipeline = api.get_pipeline()

    assert isinstance(pipeline.llm, AnthropicLLM)
    assert pipeline.llm.model == "configured-model-id"
    assert pipeline.llm.api_key == "test-key"
    assert pipeline.top_k == 3


def test_pipeline_requires_top_k_configuration(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("TOP_K", raising=False)

    with pytest.raises(HTTPException) as error:
        api.get_pipeline()
    assert error.value.detail == "Set TOP_K in the .env file"


@pytest.mark.parametrize("top_k", ["invalid", "0", "-1"])
def test_pipeline_rejects_invalid_top_k_configuration(
    monkeypatch: pytest.MonkeyPatch,
    top_k: str,
):
    monkeypatch.setenv("TOP_K", top_k)
    monkeypatch.setenv("TOXICITY_THRESHOLD", "0.5")

    with pytest.raises(HTTPException) as error:
        api.get_pipeline()

    assert error.value.status_code == 503
    assert error.value.detail == "TOP_K must be a positive integer"


class RecordingStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.query_text = None
        self.query_k = None
        self.query_results = []

    def upsert(self, chunks, embeddings=None):
        self.chunks = chunks
        self.embeddings = embeddings

    def query(self, text, k):
        self.query_text = text
        self.query_k = k
        return self.query_results


class RecordingLLM:
    def __init__(self, answer="A grounded answer."):
        self.answer = answer
        self.system = None
        self.user = None

    def generate(self, system, user):
        self.system = system
        self.user = user
        return self.answer


class RecordingCollection:
    def __init__(self):
        self.upsert_args = None
        self.query_args = None

    def upsert(self, **kwargs):
        self.upsert_args = kwargs

    def query(self, **kwargs):
        self.query_args = kwargs
        return {
            "documents": [["A document."]],
            "metadatas": [[{"source": "guide.md", "chunk_index": "0"}]],
        }


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
    pipeline = RAGPipeline(embedder=None, store=store, llm=None, top_k=3)

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


def test_vector_store_queries_text_with_chromas_default_embedding():
    collection = RecordingCollection()
    store = VectorStore(collection)

    chunks = store.query("What does the document say?", k=3)

    assert collection.query_args == {
        "query_texts": ["What does the document say?"],
        "n_results": 3,
        "include": ["documents", "metadatas"],
    }
    assert chunks == [Chunk(text="A document.", source="guide.md", index=0)]


def test_pipeline_asks_llm_with_top_three_retrieved_chunks():
    store = RecordingStore()
    store.query_results = [Chunk(text="Relevant text.", source="guide.md", index=2)]
    llm = RecordingLLM()
    pipeline = RAGPipeline(embedder=None, store=store, llm=llm, top_k=3)

    answer = pipeline.ask("What does the document say?")

    assert answer == "A grounded answer."
    assert store.query_text == "What does the document say?"
    assert store.query_k == 3
    assert "Relevant text." in llm.user
    assert "What does the document say?" in llm.user
    assert "using only the supplied document context" in llm.system


def test_ingest_endpoint_persists_docs_and_returns_chunk_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "guide.md").write_text("A document.", encoding="utf-8")
    store = RecordingStore()
    pipeline = RAGPipeline(embedder=None, store=store, llm=None, top_k=3)
    monkeypatch.setattr(api, "DOCS_DIR", tmp_path)
    monkeypatch.setitem(
        api.app.dependency_overrides, api.get_pipeline, lambda: pipeline
    )

    response = TestClient(api.app).post("/ingest")

    assert response.status_code == 200
    assert response.json() == {"chunks_ingested": 1}
    assert store.chunks[0].text == "A document."


def test_ask_endpoint_returns_answer_from_pipeline(
    monkeypatch: pytest.MonkeyPatch,
):
    store = RecordingStore()
    store.query_results = [Chunk(text="The answer is 42.", source="guide.md", index=0)]
    pipeline = RAGPipeline(
        embedder=None,
        store=store,
        llm=RecordingLLM("The answer is 42."),
        top_k=3,
    )
    monkeypatch.setitem(
        api.app.dependency_overrides, api.get_pipeline, lambda: pipeline
    )

    response = TestClient(api.app).post(
        "/ask", json={"question": "What is the answer?"}
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "The answer is 42."}
    assert store.query_k == 3


def test_ask_endpoint_rejects_empty_question(
    monkeypatch: pytest.MonkeyPatch,
):
    pipeline = RAGPipeline(
        embedder=None, store=RecordingStore(), llm=RecordingLLM(), top_k=3
    )
    monkeypatch.setitem(
        api.app.dependency_overrides, api.get_pipeline, lambda: pipeline
    )

    response = TestClient(api.app).post("/ask", json={"question": "  "})

    assert response.status_code == 422


def test_ask_endpoint_reports_missing_anthropic_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    pipeline = RAGPipeline(
        embedder=None,
        store=RecordingStore(),
        llm=AnthropicLLM(model="claude-sonnet-5-5", api_key=None),
        top_k=3,
    )
    monkeypatch.setitem(
        api.app.dependency_overrides, api.get_pipeline, lambda: pipeline
    )

    response = TestClient(api.app).post("/ask", json={"question": "Question?"})

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_ingest_endpoint_returns_not_found_for_missing_docs_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = RecordingStore()
    pipeline = RAGPipeline(embedder=None, store=store, llm=None, top_k=3)
    monkeypatch.setattr(api, "DOCS_DIR", tmp_path / "missing")
    monkeypatch.setitem(
        api.app.dependency_overrides, api.get_pipeline, lambda: pipeline
    )

    response = TestClient(api.app).post("/ingest")

    assert response.status_code == 404
    assert "Document directory does not exist" in response.json()["detail"]
