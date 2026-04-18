"""Build a persistent ChromaDB vectorstore from local agronomy text files."""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME, EMBEDDINGS_CACHE_DIR, RAG_DOCS_DIR, VECTORSTORE_DIR


def load_documents() -> list:
    """Load all plain-text agronomy reference files."""
    documents = []
    for path in sorted(RAG_DOCS_DIR.glob("*.txt")):
        loader = TextLoader(str(path), encoding="utf-8")
        documents.extend(loader.load())
    return documents


def build_vectorstore() -> None:
    """Chunk text files, embed them locally, and persist the Chroma store."""
    documents = load_documents()
    if not documents:
        raise FileNotFoundError(f"No .txt files found in {RAG_DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=str(EMBEDDINGS_CACHE_DIR),
        model_kwargs={"device": "cpu", "local_files_only": True},
    )
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    print(f"Indexed {len(chunks)} chunks into {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vectorstore()
