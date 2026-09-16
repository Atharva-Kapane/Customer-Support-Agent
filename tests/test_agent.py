from src.agent import SupportAgent
from src.intent_classifier import IntentClassifier
import json

classifier = IntentClassifier()
agent = SupportAgent(intent_classifier=classifier)

queries = [
    "My package is late, I paid for one-day shipping",
    "I want to talk to a real human agent right now",
    "hello",
    "I was charged twice for my Prime membership"
]

for q in queries:
    print("\n" + "="*80)
    print("QUERY:", q)
    result = agent.run(q)
    print(json.dumps(result["final_decision"], indent=2))
    print("Escalation reason:", result["escalation"]["reason"])