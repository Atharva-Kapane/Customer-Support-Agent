from src.retriever import Retriever

retriever = Retriever()

query = "My package was supposed to arrive today but it didn't. I want a refund for the shipping."
predicted_intent = "delivery_late"

results = retriever.retrieve(query, predicted_intent=predicted_intent, top_k=5)

for i, r in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Intent      : {r['intent']}")
    print(f"Similarity  : {r['similarity']:.4f}")
    print(f"Final score : {r['final_score']:.4f}")
    print(f"Query       : {r['customer_query'][:120]}...")
    print(f"Brand reply : {r['brand_replies'][:150]}...")