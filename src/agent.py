import json
from datetime import datetime
from typing import Dict, Any, Optional, List

from src.retriever import Retriever
from src.escalator import decide_escalation
from src.generator import generate_reply_and_decision


def _docs_to_context_strings(retrieved_docs: Optional[List[Dict]]) -> List[str]:
    """
    Turn retriever dicts into plain text strings for evaluation / logging.
    One string per doc: customer query + brand reply.
    """
    if not retrieved_docs:
        return []

    contexts = []
    for doc in retrieved_docs:
        customer = doc.get("customer_query") or ""
        brand = doc.get("brand_replies") or ""
        # Keep it readable for the judge later
        text = f"Customer: {customer}\nAmazon: {brand}".strip()
        if text:
            contexts.append(text)
    return contexts


class SupportAgent:
    def __init__(self, intent_classifier=None):
        """
        intent_classifier: any object that has a method
        predict(text: str) -> dict with keys: label, confidence
        """
        self.retriever = Retriever()
        self.intent_classifier = intent_classifier

    def run(self, query: str) -> Dict[str, Any]:
        """
        Full offline pipeline.
        Returns a rich JSON log of every decision, plus flat fields
        for evaluation: action, reason, reply, retrieved_contexts.
        """
        log = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "intent": {},
            "retrieval": {},
            "escalation": {},
            "generation": {},
            "final_decision": {},
            # Flat fields (always present for eval)
            "action": None,
            "reason": None,
            "reply": "",
            "retrieved_contexts": [],
        }

        # -------------------------------------------------
        # 1. Intent Classification
        # -------------------------------------------------
        if self.intent_classifier is None:
            raise ValueError("Intent classifier is required")

        intent_result = self.intent_classifier.predict(query)
        predicted_intent = intent_result["label"]
        intent_confidence = float(intent_result["confidence"])

        log["intent"] = {
            "predicted": predicted_intent,
            "confidence": intent_confidence,
            "stage": "intent_classifier",
        }

        # -------------------------------------------------
        # 2. Fast Escalation Gate (before retrieval)
        # -------------------------------------------------
        early_decision = decide_escalation(
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            retrieved_docs=None,
        )

        if early_decision["should_escalate"]:
            reason = early_decision["reason"]
            log["escalation"] = {
                "should_escalate": True,
                "reason": reason,
                "decided_at": early_decision["stage"],
                "rule_based_passed": False,
            }
            log["final_decision"] = {
                "action": "escalate",
                "reply": "",
                "escalation_reason": reason,
            }
            # Flat fields
            log["action"] = "escalate"
            log["reason"] = reason
            log["reply"] = ""
            log["retrieved_contexts"] = []  # never retrieved
            return log

        # -------------------------------------------------
        # 3. Retrieval
        # -------------------------------------------------
        retrieved_docs = self.retriever.retrieve(
            query=query,
            predicted_intent=predicted_intent,
            top_k=5,
        )

        top_similarity = retrieved_docs[0]["similarity"] if retrieved_docs else 0.0
        contexts = _docs_to_context_strings(retrieved_docs)

        log["retrieval"] = {
            "top_k": len(retrieved_docs),
            "top_similarity": round(top_similarity, 4),
            "docs_used": min(3, len(retrieved_docs)),
        }
        log["retrieved_contexts"] = contexts  # set as soon as we have them

        # -------------------------------------------------
        # 4. Second Escalation Gate (after retrieval)
        # -------------------------------------------------
        retrieval_decision = decide_escalation(
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            retrieved_docs=retrieved_docs,
        )

        if retrieval_decision["should_escalate"]:
            reason = retrieval_decision["reason"]
            log["escalation"] = {
                "should_escalate": True,
                "reason": reason,
                "decided_at": retrieval_decision["stage"],
                "rule_based_passed": False,
            }
            log["final_decision"] = {
                "action": "escalate",
                "reply": "",
                "escalation_reason": reason,
            }
            log["action"] = "escalate"
            log["reason"] = reason
            log["reply"] = ""
            # retrieved_contexts already set above
            return log

        # -------------------------------------------------
        # 5. LLM Decision + Reply Generation
        # -------------------------------------------------
        llm_result = generate_reply_and_decision(
            user_query=query,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            retrieved_docs=retrieved_docs,
        )

        log["generation"] = {
            "reply": llm_result.get("reply", ""),
            "raw_llm_output": llm_result.get("raw_llm_output"),
            "llm_should_escalate": llm_result["should_escalate"],
            "llm_reason": llm_result["reason"],
        }

        log["escalation"] = {
            "should_escalate": llm_result["should_escalate"],
            "reason": llm_result["reason"],
            "decided_at": "llm",
            "rule_based_passed": True,
        }

        # -------------------------------------------------
        # 6. Final Decision
        # -------------------------------------------------
        if llm_result["should_escalate"]:
            reason = llm_result["reason"]
            log["final_decision"] = {
                "action": "escalate",
                "reply": "",
                "escalation_reason": reason,
            }
            log["action"] = "escalate"
            log["reason"] = reason
            log["reply"] = ""
        else:
            reply = llm_result.get("reply", "") or ""
            log["final_decision"] = {
                "action": "auto_reply",
                "reply": reply,
                "escalation_reason": None,
            }
            log["action"] = "auto_reply"
            log["reason"] = llm_result.get("reason", "")
            log["reply"] = reply

        # retrieved_contexts already set after retrieval
        return log