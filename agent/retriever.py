"""Chroma retriever setup."""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME, EMBEDDINGS_CACHE_DIR, VECTORSTORE_DIR


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the local embedding model used for retrieval."""
    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=str(EMBEDDINGS_CACHE_DIR),
        model_kwargs={"device": "cpu", "local_files_only": True},
    )


def get_vectorstore() -> Chroma:
    """Return the persistent Chroma vector store."""
    return Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=get_embeddings(),
    )
