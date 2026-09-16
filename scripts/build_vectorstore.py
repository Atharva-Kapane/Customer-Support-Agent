"""
Build the FAISS vector store from the training corpus.
"""

from src.embeddings import build_faiss_index
from src.config import CORPUS_DIR

if __name__ == "__main__":
    corpus_path = CORPUS_DIR / "corpus_train.parquet"
    build_faiss_index(corpus_path)
    print("\n Vector store built successfully!")