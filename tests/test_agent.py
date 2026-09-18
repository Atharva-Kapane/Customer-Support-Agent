from src.agent import SupportAgent
from src.intent_classifier import IntentClassifier
import json

agent = SupportAgent(intent_classifier=IntentClassifier())
out = agent.run("My package is late, I paid for one-day shipping")

print(json.dumps({k: out[k] for k in out if k != "retrieved_contexts"}, indent=2))
print("n_contexts:", len(out["retrieved_contexts"]))