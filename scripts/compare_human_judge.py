"""
Compare human scores vs LLM judge on the sample.

  python -m scripts.compare_human_judge
"""

from __future__ import annotations

import json
from pathlib import Path

HUMAN_PATH = Path("reports/human_judge_sample.jsonl")
JUDGE_PATH = Path("reports/eval_with_judge.jsonl")
AXES = ["helpfulness", "groundedness", "tone", "safety", "overall"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    human_rows = {r["id"]: r["human"] for r in load_jsonl(HUMAN_PATH)}
    judge_rows = {}
    for r in load_jsonl(JUDGE_PATH):
        if r.get("id") in human_rows and r.get("judge") and r["judge"].get("overall") is not None:
            judge_rows[r["id"]] = r["judge"]

    ids = sorted(set(human_rows) & set(judge_rows))
    print(f"Paired examples: {len(ids)}")

    for axis in AXES:
        diffs = []
        exact = 0
        within1 = 0
        for i in ids:
            h = human_rows[i].get(axis)
            j = judge_rows[i].get(axis)
            if not isinstance(h, (int, float)) or not isinstance(j, (int, float)):
                continue
            d = abs(float(h) - float(j))
            diffs.append(d)
            if d == 0:
                exact += 1
            if d <= 1:
                within1 += 1
        n = len(diffs)
        if n == 0:
            continue
        mae = sum(diffs) / n
        print(
            f"{axis:14s}  n={n}  MAE={mae:.3f}  exact={exact}/{n} ({exact/n:.1%})  "
            f"within±1={within1}/{n} ({within1/n:.1%})"
        )

    print("\nPer-id overall (human vs judge):")
    for i in ids:
        h = human_rows[i].get("overall")
        j = judge_rows[i].get("overall")
        print(f"  {i}: human={h}  judge={j}  diff={abs(h-j) if h is not None and j is not None else 'NA'}")


if __name__ == "__main__":
    main()