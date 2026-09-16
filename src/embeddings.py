import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    CORPUS_DIR,
)


def load_embedding_model():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def build_faiss_index(
    corpus_path: Path,
    index_path: Path = FAISS_INDEX_PATH,
    metadata_path: Path = FAISS_METADATA_PATH,
    batch_size: int = 256,
):
    """
    Embeds customer_query column and builds a FAISS IndexFlatIP (cosine similarity).
    Also saves the metadata so we can map back to original rows.
    """
    print(f"Loading corpus from {corpus_path}")
    df = pd.read_parquet(corpus_path)
    print(f"Corpus size: {len(df)} conversations")

    model = load_embedding_model()

    queries = df["customer_query"].tolist()
    embeddings = []

    print("Creating embeddings...")
    for i in tqdm(range(0, len(queries), batch_size)):
        batch = queries[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings).astype("float32")

    # Normalize for cosine similarity (IndexFlatIP)
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors")

    # Save index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index → {index_path}")

    # Save metadata (everything we need at retrieval time)
    metadata_cols = [
        "conversation_id",
        "customer_query",
        "brand_replies",
        "full_conversation",
        "intent",
        "num_turns",
        "num_brand_replies",
    ]
    df[metadata_cols].to_parquet(metadata_path, index=False)
    print(f"Saved metadata → {metadata_path}")

    return index, df


if __name__ == "__main__":
    # Build using the training corpus
    corpus_path = CORPUS_DIR / "corpus_train.parquet"
    build_faiss_index(corpus_path)