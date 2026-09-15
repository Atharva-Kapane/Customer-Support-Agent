import pandas as pd
from pathlib import Path


# Preparing the RAG Corpus
# Bringing tweet level dataset to conversation level 

# -------------------------------------------------
# Paths 
# -------------------------------------------------
DATA_DIR = Path("dataset/processed")
OUTPUT_DIR = Path("dataset/rag_corpus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Load data
# -------------------------------------------------
rag_train = pd.read_parquet(DATA_DIR / "rag_train_v1.parquet")
rag_val   = pd.read_parquet(DATA_DIR / "rag_val_v1.parquet")
rag_test  = pd.read_parquet(DATA_DIR / "rag_test_v1.parquet")

clf_train = pd.read_parquet(DATA_DIR / "clf_train_labeled_v1.parquet")
clf_val   = pd.read_parquet(DATA_DIR / "clf_val_labeled_v1.parquet")
clf_test  = pd.read_parquet(DATA_DIR / "clf_test_labeled_v1.parquet")

print("Loaded all files.")

# -------------------------------------------------
# Helper function
# -------------------------------------------------
def build_corpus(rag_df: pd.DataFrame, clf_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Creates one row per conversation with:
    - customer_query   (first inbound message)
    - brand_replies    (all AmazonHelp replies joined)
    - intent           (from classifier labels)
    - full_conversation (optional, for debugging)
    """
    
    # Sort so that conversation order is preserved
    rag_df = rag_df.sort_values(["conversation_id", "created_at"]).copy()
    
    records = []
    
    for conv_id, group in rag_df.groupby("conversation_id"):
        group = group.reset_index(drop=True)
        
        # First inbound message = the customer query we will embed
        inbound_msgs = group[group["inbound"] == True]
        if len(inbound_msgs) == 0:
            continue
            
        first_query = inbound_msgs.iloc[0]["text"]
        
        # All brand (AmazonHelp) replies
        brand_replies = group[group["inbound"] == False]["text"].tolist()
        brand_replies_text = " ||| ".join(brand_replies) if brand_replies else ""
        
        # Full conversation (useful for later inspection)
        full_conv = []
        for _, row in group.iterrows():
            speaker = "Customer" if row["inbound"] else "Amazon"
            full_conv.append(f"{speaker}: {row['text']}")
        full_conversation = "\n".join(full_conv)
        
        records.append({
            "conversation_id": conv_id,
            "customer_query": first_query,
            "brand_replies": brand_replies_text,
            "full_conversation": full_conversation,
            "num_turns": len(group),
            "num_brand_replies": len(brand_replies),
        })
    
    corpus = pd.DataFrame(records)
    
    # Attach intent from classifier data (soft signal)
    # We take the intent of the first message of that conversation
    intent_map = clf_df.set_index("conversation_id")["predicted_intent"].to_dict()
    corpus["intent"] = corpus["conversation_id"].map(intent_map)
    
    # Drop conversations that somehow have no intent (should be rare)
    missing = corpus["intent"].isna().sum()
    if missing > 0:
        print(f"[{split_name}] Warning: {missing} conversations missing intent label")
    
    corpus = corpus.dropna(subset=["intent"]).reset_index(drop=True)
    
    print(f"[{split_name}] Created corpus with {len(corpus)} conversations")
    return corpus


# -------------------------------------------------
# Build all three corpora
# -------------------------------------------------
corpus_train = build_corpus(rag_train, clf_train, "train")
corpus_val   = build_corpus(rag_val,   clf_val,   "val")
corpus_test  = build_corpus(rag_test,  clf_test,  "test")

# -------------------------------------------------
# Save
# -------------------------------------------------
corpus_train.to_parquet(OUTPUT_DIR / "corpus_train.parquet", index=False)
corpus_val.to_parquet(OUTPUT_DIR / "corpus_val.parquet", index=False)
corpus_test.to_parquet(OUTPUT_DIR / "corpus_test.parquet", index=False)

print("\nSaved all corpora to:", OUTPUT_DIR)
print("\nExample row:")
print(corpus_train.iloc[0][["conversation_id", "customer_query", "intent", "num_brand_replies"]].to_dict())