"""
Intent accuracy + early escalation-gate metrics.

Uses only:
  - IntentClassifier
  - decide_escalation(..., retrieved_docs=None)

No FAISS, no generator, no API tokens.

Usage:
  python scripts/evaluate_classification.py
  python scripts/evaluate_classification.py --limit 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.intent_classifier import IntentClassifier
from src.escalator import decide_escalation

GOLDEN_PATH = Path("data/golden_set/golden_evaluation_set.jsonl")
OUT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_golden(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--golden", type=str, default=str(GOLDEN_PATH))
    args = parser.parse_args()

    golden = load_golden(Path(args.golden))
    if args.limit:
        golden = golden[: args.limit]

    print(f"Loaded {len(golden)} examples")
    print("Loading intent classifier only (no LLM, no FAISS)...")
    clf = IntentClassifier()

    y_true_intent, y_pred_intent = [], []
    y_true_action, y_pred_action = [], []
    rows_out = []

    for i, ex in enumerate(golden, 1):
        query = ex["query"]
        print(f"[{i}/{len(golden)}] {query[:70]}...")

        # 1) Intent only
        pred = clf.predict(query)
        pred_intent = pred["label"]
        pred_conf = float(pred["confidence"])

        # 2) Early escalation gate only (no retrieval, no LLM)
        early = decide_escalation(
            predicted_intent=pred_intent,
            intent_confidence=pred_conf,
            retrieved_docs=None,
        )
        pred_action = "escalate" if early["should_escalate"] else "auto_reply"
        pred_reason = early["reason"]

        y_true_intent.append(ex["gold_intent"])
        y_pred_intent.append(pred_intent)
        y_true_action.append(ex["gold_action"])
        y_pred_action.append(pred_action)

        rows_out.append(
            {
                "id": ex.get("id"),
                "query": query,
                "gold_intent": ex["gold_intent"],
                "pred_intent": pred_intent,
                "pred_confidence": pred_conf,
                "gold_action": ex["gold_action"],
                "pred_action_early_gate": pred_action,
                "early_gate_reason": pred_reason,
                "difficulty": ex.get("difficulty"),
            }
        )

    intent_acc = accuracy_score(y_true_intent, y_pred_intent)
    intent_f1 = f1_score(y_true_intent, y_pred_intent, average="macro", zero_division=0)

    action_acc = accuracy_score(y_true_action, y_pred_action)
    action_f1 = f1_score(
        y_true_action,
        y_pred_action,
        average="macro",
        labels=["auto_reply", "escalate"],
        zero_division=0,
    )

    print("\n" + "=" * 50)
    print("INTENT (classifier only)")
    print(f"  accuracy : {intent_acc:.4f}")
    print(f"  macro-F1 : {intent_f1:.4f}")
    print(classification_report(y_true_intent, y_pred_intent, zero_division=0, digits=3))

    print("=" * 50)
    print("EARLY ESCALATION GATE vs gold_action")
    print("  (pred escalate only from rules; no RAG/LLM)")
    print(f"  accuracy : {action_acc:.4f}")
    print(f"  macro-F1 : {action_f1:.4f}")
    print(
        classification_report(
            y_true_action,
            y_pred_action,
            labels=["auto_reply", "escalate"],
            zero_division=0,
            digits=3,
        )
    )
    print("Confusion matrix [rows=gold, cols=pred] labels=[auto_reply, escalate]")
    print(confusion_matrix(y_true_action, y_pred_action, labels=["auto_reply", "escalate"]))

    pred_path = OUT_DIR / "classification_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "n": len(golden),
        "mode": "intent + early_escalation_gate only (no LLM)",
        "intent_accuracy": intent_acc,
        "intent_macro_f1": intent_f1,
        "early_gate_action_accuracy": action_acc,
        "early_gate_action_macro_f1": action_f1,
    }
    (OUT_DIR / "classification_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nSaved:")
    print(f"  {pred_path}")
    print(f"  {OUT_DIR / 'classification_summary.json'}")
    print(
        "\nNote: early-gate action accuracy is partial. "
        "Full gold_action also depends on retrieval + LLM (later, one combined run)."
    )


if __name__ == "__main__":
    main()