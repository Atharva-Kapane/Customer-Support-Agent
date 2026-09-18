"""
Trivial + simple baselines on the golden set.
Writes reports/baseline_run.jsonl and prints action metrics.

  python -m scripts.run_baselines
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.intent_classifier import IntentClassifier
from src.escalator import decide_escalation

GOLDEN = Path("data/golden_set/golden_evaluation_set.jsonl")
OUT = Path("reports/baseline_run.jsonl")

# Intent -> short canned reply (simple baseline only)
TEMPLATES = {
    "delivery_late": (
        "We're sorry your package is delayed. Please check tracking in Your Orders. "
        "If the delivery date has passed, contact us via chat or phone."
    ),
    "order_tracking_inquiry": (
        "You can track your order under Your Orders in your Amazon account. "
        "If tracking isn't updating, contact support via chat or phone."
    ),
    "order_cancellation": (
        "You can cancel eligible orders from Your Orders. If the cancel option "
        "isn't available, the item may have shipped—contact chat or phone for help."
    ),
    "delivery_missing": (
        "We're sorry the package appears missing. Please check around the delivery "
        "area and with neighbors, then contact us via chat or phone with your order details."
    ),
    "delivery_service_complaint": (
        "We're sorry about the delivery experience. Please share feedback in Your Orders "
        "or contact us via chat or phone so we can look into it."
    ),
    "gift_card_problems": (
        "For gift card issues, try redeeming again from Your Account > Gift cards. "
        "If the problem continues, contact us via chat or phone."
    ),
    "prime_subscription_issue": (
        "You can manage Prime under Account > Prime membership. "
        "If something still looks wrong, contact us via chat or phone."
    ),
    "amazon_india_support": (
        "For Amazon.in specific help, please use the Help section on Amazon.in "
        "or contact customer service for your region."
    ),
    "device_and_technical_support": (
        "Please try a restart and check our device help pages. "
        "If it still fails, contact us via chat or phone."
    ),
    "amazon_pay_fintech": (
        "For payment or wallet issues, please contact us via chat or phone "
        "so we can review the transaction securely."
    ),
    "prime_billing_complaint": (
        "We're sorry about the billing concern. Please contact us via chat or phone "
        "so we can review charges on your account."
    ),
    "account_and_login_issues": (
        "For account access issues, use account recovery on Amazon or contact support. "
        "We recommend securing your email and password."
    ),
    "general_agent_escalation": "",  # escalate
    "other": "",  # escalate
}

DEFAULT_TEMPLATE = (
    "Thanks for contacting us. Please check Your Orders or reach us via chat or phone for help."
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def trivial_predict(_query: str) -> dict:
    return {
        "pred_intent": None,
        "pred_action": "escalate",
        "pred_reply": "",
        "baseline": "trivial",
    }


def simple_predict(clf: IntentClassifier, query: str) -> dict:
    pred = clf.predict(query)
    intent = pred["label"]
    conf = float(pred["confidence"])

    early = decide_escalation(
        predicted_intent=intent,
        intent_confidence=conf,
        retrieved_docs=None,
    )
    if early["should_escalate"]:
        return {
            "pred_intent": intent,
            "pred_confidence": conf,
            "pred_action": "escalate",
            "pred_reply": "",
            "baseline": "simple",
        }

    reply = TEMPLATES.get(intent, DEFAULT_TEMPLATE)
    if not reply:
        return {
            "pred_intent": intent,
            "pred_confidence": conf,
            "pred_action": "escalate",
            "pred_reply": "",
            "baseline": "simple",
        }

    return {
        "pred_intent": intent,
        "pred_confidence": conf,
        "pred_action": "auto_reply",
        "pred_reply": reply,
        "baseline": "simple",
    }


def score_actions(y_true, y_pred, name: str):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(
        y_true, y_pred, average="macro", labels=["auto_reply", "escalate"], zero_division=0
    )
    print(f"\n=== {name} ===")
    print(f"action accuracy: {acc:.4f}")
    print(f"action macro-F1: {f1:.4f}")
    print(
        classification_report(
            y_true, y_pred, labels=["auto_reply", "escalate"], zero_division=0, digits=3
        )
    )
    print(confusion_matrix(y_true, y_pred, labels=["auto_reply", "escalate"]))
    return {"action_accuracy": acc, "action_macro_f1": f1}


def main():
    golden = load_jsonl(GOLDEN)
    clf = IntentClassifier()

    out_rows = []
    trivial_pred, simple_pred = [], []
    gold_action = []

    for ex in golden:
        q = ex["query"]
        gold_action.append(ex["gold_action"])

        t = trivial_predict(q)
        s = simple_predict(clf, q)
        trivial_pred.append(t["pred_action"])
        simple_pred.append(s["pred_action"])

        out_rows.append(
            {
                "id": ex.get("id"),
                "query": q,
                "gold_intent": ex.get("gold_intent"),
                "gold_action": ex.get("gold_action"),
                "trivial": t,
                "simple": s,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    t_m = score_actions(gold_action, trivial_pred, "TRIVIAL (always escalate)")
    s_m = score_actions(gold_action, simple_pred, "SIMPLE (intent + template)")

    summary = {
        "n": len(golden),
        "trivial": t_m,
        "simple": s_m,
    }
    Path("reports/baseline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {OUT}")
    print("Wrote reports/baseline_summary.json")


if __name__ == "__main__":
    main()