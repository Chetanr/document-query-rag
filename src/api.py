import logging
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, logger
from pydantic import BaseModel

from .llm import AnthropicLLM, LLMConfigurationError
from .pipeline import RAGPipeline
from .vectorstore import VectorStore

app = FastAPI(title="Demo RAG App", version="0.1.0")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
load_dotenv(PROJECT_ROOT / ".env")


class IngestResponse(BaseModel):
    chunks_ingested: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


def get_pipeline() -> RAGPipeline:
    top_k_value = os.environ.get("TOP_K")
    if top_k_value is None:
        logger.error("TOP_K environment variable not set")
        raise HTTPException(status_code=503, detail="Set TOP_K in the .env file")
    try:
        top_k = int(top_k_value)
    except ValueError as error:
        logger.error("Invalid TOP_K value. Must be a positive integer.")
        raise HTTPException(
            status_code=503,
            detail="TOP_K must be a positive integer",
        ) from error
    if top_k <= 0:
        logger.error("Invalid TOP_K value. Must be a positive integer.")
        raise HTTPException(
            status_code=503,
            detail="TOP_K must be a positive integer",
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="documents")
    return RAGPipeline(
        embedder=None,
        store=VectorStore(collection),
        llm=AnthropicLLM(
            model=os.environ.get("ANTHROPIC_MODEL"),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        ),
        top_k=top_k,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(pipeline: RAGPipeline = Depends(get_pipeline)) -> IngestResponse:
    logging.info(f"Ingesting documents from {DOCS_DIR}")
    try:
        chunks_ingested = pipeline.ingest(str(DOCS_DIR))
    except FileNotFoundError as error:
        logging.error(f"Directory not found: {DOCS_DIR}")
        raise HTTPException(status_code=404, detail=str(error)) from error
    return IngestResponse(chunks_ingested=chunks_ingested)


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> AskResponse:
    if not request.question.strip():
        logging.error("Empty question received")
        raise HTTPException(status_code=422, detail="Question must not be empty")
    try:
        return AskResponse(answer=pipeline.ask(request.question))
    except LLMConfigurationError as error:
        logging.error("LLM configuration error occurred")
        raise HTTPException(status_code=503, detail=str(error)) from error
