"""Document ingestion helpers for building the vector store."""

from pathlib import Path

from farm_advisor.config import RAG_DOCS_DIR


def load_document_paths() -> list[Path]:
    """Collect plain-text document paths from the local RAG docs directory."""
    return sorted(path for path in RAG_DOCS_DIR.glob("*.txt") if path.is_file())
