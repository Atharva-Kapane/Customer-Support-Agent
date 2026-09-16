from typing import Dict, Any, List, Optional


def decide_escalation(
    predicted_intent: str,
    intent_confidence: float,
    retrieved_docs: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Fast rule-based gate.
    Only escalates early when we are clearly unsure or the query is not useful.
    Everything else goes to the LLM for a final decision.
    """

    # 1. Gibberish / greeting / irrelevant
    if predicted_intent == "other":
        return {
            "should_escalate": True,
            "reason": "Intent classified as 'other' (greeting, gibberish, or irrelevant).",
            "stage": "intent_classifier"
        }

    # 2. Explicit request for human
    if predicted_intent == "general_agent_escalation":
        return {
            "should_escalate": True,
            "reason": "Query is critical. Customer explicitly needs human support.",
            "stage": "intent_classifier"
        }

    # 3. Very low confidence from classifier
    if intent_confidence < 0.50:
        return {
            "should_escalate": True,
            "reason": f"Low intent confidence ({intent_confidence:.2f}).",
            "stage": "intent_classifier"
        }

    # 4. Very poor retrieval quality
    if retrieved_docs is not None:
        if len(retrieved_docs) == 0 or retrieved_docs[0]["similarity"] < 0.50:
            return {
                "should_escalate": True,
                "reason": "No sufficiently similar past cases found in knowledge base.",
                "stage": "retrieval"
            }

    # Passed all fast gates → let the LLM make the final decision
    return {
        "should_escalate": False,
        "reason": "Passed rule-based checks. Sending to LLM for final decision.",
        "stage": "llm"
    }