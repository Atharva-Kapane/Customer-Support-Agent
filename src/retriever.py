import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

from src.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
)


class Retriever:
    def __init__(self):
        print("Loading FAISS index and metadata...")
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.metadata = pd.read_parquet(FAISS_METADATA_PATH)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Retriever ready ({self.index.ntotal} vectors)")

    def retrieve(
        self,
        query: str,
        predicted_intent: Optional[str] = None,
        top_k: int = 8,
        intent_boost: float = 0.08,      # small boost
        intent_penalty: float = 0.04,    # small penalty
    ) -> List[Dict]:
        """
        Retrieve most similar past customer queries.
        Softly boosts chunks whose intent matches the predicted intent.
        """
        # Embed query
        query_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_emb)

        # Search more than we need so we can re-rank
        search_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_emb, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            row = self.metadata.iloc[idx]
            final_score = float(score)

            # Soft intent signal
            if predicted_intent is not None:
                if row["intent"] == predicted_intent:
                    final_score += intent_boost
                else:
                    final_score -= intent_penalty

            results.append({
                "conversation_id": int(row["conversation_id"]),
                "customer_query": row["customer_query"],
                "brand_replies": row["brand_replies"],
                "full_conversation": row["full_conversation"],
                "intent": row["intent"],
                "num_turns": int(row["num_turns"]),
                "similarity": float(score),          # original cosine
                "final_score": final_score,          # after soft intent adjustment
            })

        # Re-rank by final_score and keep top_k
        results = sorted(results, key=lambda x: x["final_score"], reverse=True)[:top_k]
        return results