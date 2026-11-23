import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    source: str
    index: int


def chunk_document(
    source: str,
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> list[Chunk]:
    """Split a document into chunks."""
    if chunk_size <= 0:
        logger.error("Invalid chunk_size value. Must be greater than zero.")
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        logger.error("Invalid overlap value. Must be between zero and chunk_size.")
        raise ValueError("overlap must be between zero and chunk_size")

    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, source=source, index=len(chunks)))
        if end == len(text):
            break
        start = end - overlap
    return chunks
