from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_DIR = ROOT / "dataset"
CORPUS_DIR = DATA_DIR / "rag_corpus"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

INTENT_MODEL_PATH = ROOT / "models" / "intent_classifier_distilbert"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss_index.bin"
FAISS_METADATA_PATH = VECTORSTORE_DIR / "metadata.parquet"