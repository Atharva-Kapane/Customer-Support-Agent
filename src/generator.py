import os
import json
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

def get_openrouter_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

LLM_PROVIDERS = [
    # Primary - Gemini
    ("gemini", get_gemini_client, "gemini-2.5-flash"),

    # Groq
    ("groq", get_groq_client, "openai/gpt-oss-20b"),
    
    # Strong free fallback
    ("openrouter", get_openrouter_client, "google/gemma-4-31b-it:free"),
    
    # Extra free fallbacks (optional)
    ("openrouter", get_openrouter_client, "nvidia/nemotron-3.5-lightning:free"),
    ("openrouter", get_openrouter_client, "openrouter/free"),
]



def call_llm(messages: list, temperature: float = 0.3) -> str:
    errors = []
    for name, client_fn, model in LLM_PROVIDERS:
        client = client_fn()
        if client is None:
            continue
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=700,
                response_format={"type": "json_object"}   # ← force JSON
            )
            print(f"[LLM] Used provider: {name}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            errors.append(f"{name}: {str(e)}")
            continue

    raise RuntimeError("All LLM providers failed:\n" + "\n".join(errors))


def generate_reply_and_decision(
    user_query: str,
    predicted_intent: str,
    intent_confidence: float,
    retrieved_docs: List[Dict],
) -> Dict:
    """
    LLM returns a clean JSON object.
    """

    examples = []
    for i, doc in enumerate(retrieved_docs[:3], 1):
        examples.append({
            "customer": doc["customer_query"],
            "amazon_reply": doc["brand_replies"]
        })

    system_prompt = """You are an Amazon customer support agent assistant.

Your job is to decide whether the current customer message can be safely auto-replied or should be escalated to a human.

Respond ONLY with a valid JSON object using this exact schema:

{
  "should_escalate": true or false,
  "reason": "short clear reason",
  "reply": "the reply to the customer if should_escalate is false, otherwise empty string"
}

### When to AUTO-REPLY (should_escalate = false):
- Simple delivery delays / late packages
- Order tracking questions
- Basic cancellation requests
- Common Prime subscription questions
- Gift card issues that don't involve fraud
- Cases where similar past Amazon replies exist and are helpful

### When to ESCALATE (should_escalate = true):
- Explicit request for a human agent
- Billing disputes / double charges / unauthorized payments
- Account login / security issues
- Strong anger + missing critical details
- Potential fraud or legal threats
- Cases where you truly lack enough information to help

### Reply Style (when you do reply):
- Be polite, empathetic and professional
- Keep it concise
- Do not invent order IDs, tracking numbers, or specific refund promises
- You may direct the customer to contact Amazon via phone/chat if that is what similar past replies did
- Never say you are an AI

Be practical. Most common customer service messages can be handled with a good auto-reply.
"""

    user_prompt = {
        "customer_message": user_query,
        "predicted_intent": predicted_intent,
        "intent_confidence": round(intent_confidence, 3),
        "similar_past_cases": examples
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, indent=2)},
    ]

    raw = call_llm(messages)

    try:
        parsed = json.loads(raw)
    except Exception:
        # Fallback if model still doesn't return clean JSON
        parsed = {
            "should_escalate": True,
            "reason": "Failed to parse LLM JSON output. Escalating for safety.",
            "reply": ""
        }

    return {
        "should_escalate": bool(parsed.get("should_escalate", True)),
        "reason": parsed.get("reason", ""),
        "reply": parsed.get("reply", "") if not parsed.get("should_escalate") else "",
        "raw_llm_output": raw
    }