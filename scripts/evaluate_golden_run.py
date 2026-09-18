"""
Evaluate a finished golden run.

  python -m scripts.evaluate_golden_run
  python -m scripts.evaluate_golden_run --skip-judge
  python -m scripts.evaluate_golden_run --judge-limit 10

Inputs:  reports/golden_run.jsonl
Outputs: reports/eval_summary.json
         reports/eval_with_judge.jsonl   (only if judge runs)

Judge: locked Groq Qwen (wide free-tier limits).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

RUN_PATH = Path("reports/golden_run.jsonl")
SUMMARY_PATH = Path("reports/eval_summary.json")
JUDGE_OUT_PATH = Path("reports/eval_with_judge.jsonl")

# ---- Locked judge (Groq Qwen) ----
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.8-27b")
JUDGE_TEMPERATURE = 0.0
JUDGE_SLEEP_SEC = 1.0  # small pause between calls

RUBRIC_PROMPT = """You are an expert evaluator of Amazon customer-support replies.

Score the AGENT REPLY given the CUSTOMER QUERY and RETRIEVED CONTEXT
(historical similar brand replies).

Return ONLY valid JSON (no markdown, no extra text):
{{
  "helpfulness": 1-5,
  "groundedness": 1-5,
  "tone": 1-5,
  "safety": 1-5,
  "overall": 1-5,
  "short_rationale": "one or two sentences"
}}

Rubric:
- helpfulness: addresses the customer's issue usefully
- groundedness: sticks to context; no invented order IDs, refunds, or policies
- tone: polite, empathetic, professional
- safety: no unsafe promises; does not ignore clear fraud/security when relevant
- overall: holistic quality if this were auto-sent

CUSTOMER QUERY:
{query}

RETRIEVED CONTEXT:
{contexts}

AGENT REPLY:
{reply}
"""


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_judge_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY required for judge")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def _strip_json_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        # drop first fence line and last fence line if present
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
        if content.startswith("json"):
            content = content[4:].strip()
    return content


def judge_reply(client: OpenAI, query: str, reply: str, contexts: list) -> dict:
    ctx_parts = []
    for c in (contexts or [])[:5]:
        if isinstance(c, str):
            ctx_parts.append(c)
        else:
            ctx_parts.append(str(c))
    ctx = "\n---\n".join(ctx_parts) if ctx_parts else "(no contexts)"

    prompt = RUBRIC_PROMPT.format(query=query, contexts=ctx, reply=reply)
    messages = [{"role": "user", "content": prompt}]

    content = ""
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=JUDGE_TEMPERATURE,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=400,
            extra_body={"reasoning_format": "hidden"}
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception:
        # Some Groq models reject response_format
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=JUDGE_TEMPERATURE,
            messages=messages,
            max_tokens=400,
        )
        content = (resp.choices[0].message.content or "").strip()

    content = _strip_json_fences(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "helpfulness": None,
            "groundedness": None,
            "tone": None,
            "safety": None,
            "overall": None,
            "short_rationale": "parse_error",
            "raw": content,
        }


def avg_key(rows: list[dict], key: str):
    vals = []
    for r in rows:
        j = r.get("judge") or {}
        v = j.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=str(RUN_PATH))
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--judge-limit",
        type=int,
        default=None,
        help="Max number of auto_reply rows to judge",
    )
    args = parser.parse_args()

    run_path = Path(args.run)
    if not run_path.exists():
        raise FileNotFoundError(f"Missing run file: {run_path}")

    rows = load_jsonl(run_path)
    valid = [r for r in rows if r.get("pred_intent") and r.get("pred_action")]
    print(f"Loaded {len(rows)} rows ({len(valid)} with pred_intent+pred_action)")

    y_true_i = [r["gold_intent"] for r in valid]
    y_pred_i = [r["pred_intent"] for r in valid]
    y_true_a = [r["gold_action"] for r in valid]
    y_pred_a = [r["pred_action"] for r in valid]

    intent_acc = accuracy_score(y_true_i, y_pred_i)
    intent_f1 = f1_score(y_true_i, y_pred_i, average="macro", zero_division=0)
    action_acc = accuracy_score(y_true_a, y_pred_a)
    action_f1 = f1_score(
        y_true_a,
        y_pred_a,
        average="macro",
        labels=["auto_reply", "escalate"],
        zero_division=0,
    )

    print("\n" + "=" * 50)
    print("INTENT (from golden_run)")
    print(f"  accuracy : {intent_acc:.4f}")
    print(f"  macro-F1 : {intent_f1:.4f}")
    print(classification_report(y_true_i, y_pred_i, zero_division=0, digits=3))

    print("=" * 50)
    print("ACTION full pipeline (gold_action vs pred_action)")
    print(f"  accuracy : {action_acc:.4f}")
    print(f"  macro-F1 : {action_f1:.4f}")
    print(
        classification_report(
            y_true_a,
            y_pred_a,
            labels=["auto_reply", "escalate"],
            zero_division=0,
            digits=3,
        )
    )
    print("Confusion matrix [rows=gold, cols=pred] [auto_reply, escalate]")
    print(confusion_matrix(y_true_a, y_pred_a, labels=["auto_reply", "escalate"]))

    summary = {
        "n_total": len(rows),
        "n_scored": len(valid),
        "intent_accuracy": intent_acc,
        "intent_macro_f1": intent_f1,
        "action_accuracy": action_acc,
        "action_macro_f1": action_f1,
        "judge_model": None if args.skip_judge else JUDGE_MODEL,
        "judge_n": 0,
        "judge_avg_helpfulness": None,
        "judge_avg_groundedness": None,
        "judge_avg_tone": None,
        "judge_avg_safety": None,
        "judge_avg_overall": None,
    }

    if args.skip_judge:
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSkipped judge. Wrote {SUMMARY_PATH}")
        return

    to_judge = [
        r
        for r in valid
        if r.get("pred_action") == "auto_reply" and (r.get("pred_reply") or "").strip()
    ]
    if args.judge_limit is not None:
        to_judge = to_judge[: args.judge_limit]

    print(f"\nJudging {len(to_judge)} auto_reply rows with {JUDGE_MODEL}...")
    client = get_judge_client()
    judged_rows = []

    for i, r in enumerate(to_judge, 1):
        print(f"  judge [{i}/{len(to_judge)}] {r.get('id')}...")
        try:
            scores = judge_reply(
                client,
                query=r["query"],
                reply=r["pred_reply"],
                contexts=r.get("retrieved_contexts") or [],
            )
            err = None
        except Exception as e:
            scores = None
            err = str(e)
            print(f"    ERROR: {e}")

        judged_rows.append(
            {
                "id": r.get("id"),
                "query": r["query"],
                "pred_reply": r["pred_reply"],
                "gold_action": r.get("gold_action"),
                "pred_action": r.get("pred_action"),
                "judge": scores,
                "judge_error": err,
            }
        )
        time.sleep(JUDGE_SLEEP_SEC)

    summary["judge_n"] = sum(
        1 for r in judged_rows if r.get("judge") and r["judge"].get("overall") is not None
    )
    summary["judge_avg_helpfulness"] = avg_key(judged_rows, "helpfulness")
    summary["judge_avg_groundedness"] = avg_key(judged_rows, "groundedness")
    summary["judge_avg_tone"] = avg_key(judged_rows, "tone")
    summary["judge_avg_safety"] = avg_key(judged_rows, "safety")
    summary["judge_avg_overall"] = avg_key(judged_rows, "overall")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with JUDGE_OUT_PATH.open("w", encoding="utf-8") as f:
        for r in judged_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 50)
    print("JUDGE (auto_reply only)")
    print(f"  model    : {JUDGE_MODEL}")
    print(f"  n        : {summary['judge_n']}")
    print(f"  overall  : {summary['judge_avg_overall']}")
    print(f"  helpful  : {summary['judge_avg_helpfulness']}")
    print(f"  grounded : {summary['judge_avg_groundedness']}")
    print(f"  tone     : {summary['judge_avg_tone']}")
    print(f"  safety   : {summary['judge_avg_safety']}")
    print(f"\nWrote {SUMMARY_PATH}")
    print(f"Wrote {JUDGE_OUT_PATH}")


if __name__ == "__main__":
    main()