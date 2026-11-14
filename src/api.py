from pathlib import Path

import chromadb
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from .pipeline import RAGPipeline
from .vectorstore import VectorStore


app = FastAPI(title="Demo RAG App", version="0.1.0")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"


class IngestResponse(BaseModel):
    chunks_ingested: int


def get_pipeline() -> RAGPipeline:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="documents")
    return RAGPipeline(
        embedder=None,
        store=VectorStore(collection),
        llm=None,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(pipeline: RAGPipeline = Depends(get_pipeline)) -> IngestResponse:
    """Ingest documents into the vector database."""
    try:
        chunks_ingested = pipeline.ingest(str(DOCS_DIR))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return IngestResponse(chunks_ingested=chunks_ingested)
