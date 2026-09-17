# Golden Evaluation Set

**File:** `golden_evaluation_set.jsonl`  
**Size:** 180 hand-labelled examples  
**Brand:** Amazon (TWCS)

## How this set was built

1. **~98 real first-message queries** were sampled from the untouched test split
   (`clf_test_labeled_v1.parquet` / `corpus_test.parquet`). Tweets were lightly cleaned
   (newlines, t.co links, HTML entities). Anonymized author ids like `@115821` were
   rewritten as Amazon/AmazonHelp only when it improved readability; the issue content is original.
2. **~82 synthetic queries** were written to fill gaps the test split
   under-represents: greetings, explicit human requests, fraud/account-takeover,
   multi-intent, short/noisy text, adversarial messages, and calm how-to questions.

Labels were assigned using the **same policy as the offline agent**, not by blindly
copying the classifier's `predicted_intent` when it conflicted with the text.

## Fields

| field | meaning |
|---|---|
| `id` | `gold_001` … `gold_180` |
| `query` | incoming customer message (single turn) |
| `gold_intent` | one of the 14 classifier labels |
| `gold_action` | `auto_reply` or `escalate` |
| `gold_reason` | why that action is correct |
| `difficulty` | easy / medium / hard |
| `source` | `real` or `synthetic` |
| `notes` | optional labelling comments |

## Policy used for `gold_action`

**Escalate (no useful auto-reply):**
- intent `other` (greetings, gibberish, praise-only, off-topic)
- intent `general_agent_escalation` (explicit human / supervisor)
- account takeover, hacking, lockouts that are security-related
- unauthorized charges, double charges, stolen cards, missing wallet loads
- legal threats, driver-safety/OTP scams, counterfeit claims, account deletion, bereavement

**Auto-reply:**
- late delivery, tracking, simple cancel/return how-to
- gift-card add/redeem how-to (not stolen codes)
- Prime how-to / renew / trial cancel
- simple India FAQs and device troubleshooting
- missing/misdelivered packages and damaged items (point to chat/phone; do not invent refunds)

**Important:** `gold_intent` is **not** a hard filter for retrieval at runtime.
Matching intent is only a soft ranking boost.

## Mix

### Source
{
  "real": 98,
  "synthetic": 82
}

### Difficulty
{
  "easy": 113,
  "medium": 52,
  "hard": 15
}

### Action
{
  "auto_reply": 111,
  "escalate": 69
}

### Intent
{
  "other": 20,
  "delivery_late": 17,
  "order_tracking_inquiry": 15,
  "delivery_service_complaint": 15,
  "prime_subscription_issue": 14,
  "order_cancellation": 13,
  "account_and_login_issues": 13,
  "amazon_pay_fintech": 13,
  "device_and_technical_support": 12,
  "general_agent_escalation": 11,
  "delivery_missing": 10,
  "amazon_india_support": 10,
  "prime_billing_complaint": 9,
  "gift_card_problems": 8
}

## Known limitations (use in the report)

- Real tweets are noisy; a few source `predicted_intent` labels in the parquet are themselves imperfect. Gold labels follow the **text**, not the parquet label, when they disagree.
- Twitter AmazonHelp historically often replies “call/chat us” — a good auto-reply may still refuse to resolve the order in-channel.
- Multi-intent messages have one primary gold intent; `notes` flag the overlap.
