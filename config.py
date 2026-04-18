"""Shared configuration constants for model and vectorstore paths."""

from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent
RAG_DOCS_DIR = PROJECT_ROOT / "rag_docs"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
EMBEDDINGS_CACHE_DIR = VECTORSTORE_DIR / "embeddings_cache"
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

RF_MODEL_PATH = MODEL_DIR / "rf_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
LABEL_ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"
TRAINING_DATA_PATH = DATA_DIR / "crop_yield.csv"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
