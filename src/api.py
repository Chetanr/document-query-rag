import logging

import chromadb
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from .config import PROJECT_ROOT, Settings, SettingsError
from .llm import AnthropicLLM, LLMConfigurationError
from .pipeline import RAGPipeline
from .utils import toxicity
from .vectorstore import VectorStore

app = FastAPI(title="Demo RAG App", version="0.1.0")
logger = logging.getLogger(__name__)
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"


class IngestResponse(BaseModel):
    chunks_ingested: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


def get_settings() -> Settings:
    try:
        return Settings.from_env()
    except SettingsError as error:
        logger.error("Invalid app settings: %s", error)
        raise HTTPException(status_code=503, detail=str(error)) from error


def get_pipeline() -> RAGPipeline:
    settings = get_settings()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="documents")
    return RAGPipeline(
        embedder=None,
        store=VectorStore(collection),
        llm=AnthropicLLM(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
        ),
        top_k=settings.top_k,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(pipeline: RAGPipeline = Depends(get_pipeline)) -> IngestResponse:
    logger.info("Ingesting documents from %s", DOCS_DIR)
    try:
        chunks_ingested = pipeline.ingest(str(DOCS_DIR))
    except FileNotFoundError as error:
        logger.error("Documents directory not found: %s", DOCS_DIR)
        raise HTTPException(status_code=404, detail=str(error)) from error
    return IngestResponse(chunks_ingested=chunks_ingested)


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> AskResponse:
    question = request.question.strip()
    if not question:
        logger.error("Empty question received")
        raise HTTPException(status_code=422, detail="Question must not be empty")

    try:
        if toxicity.check_toxicity(question, settings.toxicity_threshold):
            logger.warning("Toxic content detected in question; request rejected")
            raise HTTPException(
                status_code=422, detail="Question contains toxic content"
            )
        return AskResponse(answer=pipeline.ask(question))
    except LLMConfigurationError as error:
        logger.error("LLM configuration error occurred: %s", error)
        raise HTTPException(status_code=503, detail=str(error)) from error
