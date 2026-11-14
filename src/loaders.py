from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    source: str
    text: str


def load_documents(directory: str) -> list[Document]:
    """Read raw documents from disk."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {root}")

    return [
        Document(
            source=path.relative_to(root).as_posix(),
            text=path.read_text(encoding="utf-8"),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    ]
