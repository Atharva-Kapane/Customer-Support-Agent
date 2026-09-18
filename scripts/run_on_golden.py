"""
One full pass over the golden set.
Saves every example to reports/golden_run.jsonl for later metrics/judge.

  set EVAL_MODE=1
  python scripts/evaluate_on_golden.py
  python scripts/evaluate_on_golden.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Lock generator before importing modules that call the LLM
os.environ["EVAL_MODE"] = os.environ.get("EVAL_MODE", "1")

from src.agent import SupportAgent
from src.intent_classifier import IntentClassifier

GOLDEN_PATH = Path("data/golden_set/golden_evaluation_set.jsonl")
OUT_PATH = Path("reports/golden_run.jsonl")
SUMMARY_PATH = Path("reports/golden_run_summary.json")


def load_jsonl(path: Path) -> list[dict]:
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
    parser.add_argument("--out", type=str, default=str(OUT_PATH))
    parser.add_argument("--resume", action="store_true",
                        help="Skip ids already present in out file")
    args = parser.parse_args()

    golden = load_jsonl(Path(args.golden))
    if args.limit:
        golden = golden[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if args.resume and out_path.exists():
        for row in load_jsonl(out_path):
            if row.get("id"):
                done_ids.add(row["id"])
        print(f"Resume: {len(done_ids)} already done")

    print(f"EVAL_MODE={os.environ.get('EVAL_MODE')}")
    print(f"Examples to run: {len(golden)}")
    print("Loading agent...")
    agent = SupportAgent(intent_classifier=IntentClassifier())

    n_ok, n_err, n_llm_action = 0, 0, 0
    mode = "a" if args.resume else "w"

    with out_path.open(mode, encoding="utf-8") as f:
        for i, ex in enumerate(golden, 1):
            eid = ex.get("id", f"row_{i}")
            if eid in done_ids:
                continue

            query = ex["query"]
            print(f"[{i}/{len(golden)}] {eid} | {query[:60]}...")

            record = {
                "id": eid,
                "query": query,
                "gold_intent": ex.get("gold_intent"),
                "gold_action": ex.get("gold_action"),
                "gold_reason": ex.get("gold_reason"),
                "difficulty": ex.get("difficulty"),
                "source": ex.get("source"),
            }

            try:
                result = agent.run(query)
                record.update(
                    {
                        "pred_intent": result["intent"]["predicted"],
                        "pred_confidence": result["intent"]["confidence"],
                        "pred_action": result["action"],
                        "pred_reason": result.get("reason"),
                        "pred_reply": result.get("reply") or "",
                        "retrieved_contexts": result.get("retrieved_contexts") or [],
                        "decided_at": (result.get("escalation") or {}).get("decided_at"),
                        "retrieval_top_similarity": (result.get("retrieval") or {}).get(
                            "top_similarity"
                        ),
                        "error": None,
                    }
                )
                if record["decided_at"] == "llm":
                    n_llm_action += 1
                n_ok += 1
            except Exception as e:
                record.update(
                    {
                        "pred_intent": None,
                        "pred_confidence": None,
                        "pred_action": None,
                        "pred_reason": None,
                        "pred_reply": "",
                        "retrieved_contexts": [],
                        "decided_at": None,
                        "retrieval_top_similarity": None,
                        "error": str(e),
                    }
                )
                n_err += 1
                print(f"  ERROR: {e}")

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    summary = {
        "n_written_this_run": n_ok + n_err,
        "n_ok": n_ok,
        "n_err": n_err,
        "n_decided_at_llm": n_llm_action,
        "eval_mode": os.environ.get("EVAL_MODE"),
        "out": str(out_path),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nDone.")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()