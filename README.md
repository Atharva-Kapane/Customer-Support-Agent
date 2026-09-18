# Amazon Twitter Support Agent

Single-turn support agent for **Amazon** tweets from the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset.

For each incoming customer message it:

1. Predicts an intent (14 classes, DistilBERT).
2. Retrieves similar historical AmazonHelp replies (FAISS + MiniLM).
3. Decides **auto-reply** vs **escalate to a human**, with a written reason.
4. If it auto-replies, drafts a reply grounded in those past cases.

I did **not** build a multi-turn chatbot, live order lookup, or refund execution. On Twitter, AmazonHelp often cannot see the order anyway. A good reply here is: empathise, point to self-serve / chat / phone, and refuse to invent tracking numbers or refunds.

**Principle:** if the agent does not know, it should say so by escalating. A wrong auto-reply is worse for the brand than sending a simple query to a human.

Repo: [github.com/Atharva-Kapane/Customer-Support-Agent](https://github.com/Atharva-Kapane/Customer-Support-Agent)

---

## Headline numbers (golden set, n=180)

| System | Action accuracy | Action macro-F1 |
|--------|-----------------|-----------------|
| Trivial baseline (always escalate) | 38.3% | 0.28 |
| Simple baseline (intent + canned template) | 68.9% | 0.64 |
| **Full agent (rules + RAG + LLM)** | **81.7%** | **0.81** |

| Metric | Value |
|--------|--------|
| Intent accuracy | 74.4% |
| Intent macro-F1 | 0.76 |
| LLM-as-judge overall (n=99 auto-replies) | 4.22 / 5 |
| Human vs judge overall MAE (n=20) | 0.60 (85% within ±1) |

These JSON files are already in `reports/`. You do **not** need to spend Groq tokens to reproduce the table. The 15-minute path below rebuilds the local index and lets you click around the demo.

---

## What you need

- Python 3.10+
- A [Groq](https://console.groq.com/) API key (generator + judge use `qwen/qwen3.8-27b`)
- ~2 GB disk for the classifier + MiniLM + FAISS
- GPU is optional. DistilBERT and MiniLM run on CPU.

Gemini / OpenRouter keys are optional fallbacks for the live demo only. Evaluation locks Groq.

---

## Setup (about 10–15 minutes)

### 1. Clone and venv

```powershell
git clone https://github.com/Atharva-Kapane/Customer-Support-Agent.git
cd Customer-Support-Agent
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS: `python -m venv venv && source venv/bin/activate`.

Always launch tools with `python -m ...` so you stay inside this venv (a bare `streamlit` command can pick a different Python and then `faiss` looks “missing”).

### 2. API key

```powershell
copy .env.example .env
```

Put your Groq key in `.env`:

```
GROQ_API_KEY=gsk_...
```

### 3. Intent classifier weights

Weights are not in git. Download them from Hugging Face into the path the code already expects:

```powershell
python -m scripts.download_intent_model
```

That pulls [`Atharva-Kapane/twcs-distilbert-intent-classifier`](https://huggingface.co/Atharva-Kapane/twcs-distilbert-intent-classifier) into `models/intent_classifier_distilbert`.

Equivalent:

```powershell
huggingface-cli download Atharva-Kapane/twcs-distilbert-intent-classifier --local-dir models/intent_classifier_distilbert
```

### 4. RAG corpus + FAISS index

Processed conversation splits already live under `dataset/processed/`. Build the conversation-level corpus, then embed first messages:

```powershell
python scripts/build_rag_corpus.py
python scripts/build_vectorstore.py
```

You should see `dataset/rag_corpus/corpus_train.parquet` and `dataset/vectorstore/faiss_index.bin` + `metadata.parquet`.

MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) downloads on first embed.

### 5. Smoke-test the agent (optional, uses Groq)

```powershell
python -m tests.test_agent
```

### 6. Demo UI

```powershell
python -m streamlit run app_streamlit.py
```

- **Live agent** — type a tweet-like message. Left: pipeline log. Chat: WhatsApp-style thread. Right: retrieved chunks / generation JSON.
- **Analytics** — the table above, loaded from `reports/eval_summary.json` and `reports/baseline_summary.json`.

---

## Reproduce evaluation (optional, uses tokens)

Locked generator: Groq `qwen/qwen3.8-27b` (`EVAL_MODE=1`).  
Locked judge: the same model (see the report for why that is a limitation).

```powershell
# Full agent pass over the 180 gold queries (slow, many LLM calls)
$env:EVAL_MODE="1"
python -m scripts.run_on_golden

# Intent + action metrics, then LLM-as-judge on auto-replies
python -m scripts.evaluate_golden_run

# Human scores vs judge (n=20)
python -m scripts.compare_human_judge

# Baselines (no LLM)
python -m scripts.run_baselines
```

Smoke only:

```powershell
$env:EVAL_MODE="1"
python -m scripts.run_on_golden --limit 5
python -m scripts.evaluate_golden_run --judge-limit 10
```

Outputs (large run files are gitignored; summaries are committed):

- `reports/golden_run.jsonl`
- `reports/eval_summary.json`
- `reports/eval_with_judge.jsonl`
- `reports/baseline_summary.json`

---

## How the pipeline works

```
query
  → DistilBERT intent + confidence
  → early rules (other / ask-for-human / conf < 0.50)  → escalate
  → FAISS retrieve similar first-messages (soft intent boost)
  → retrieval rule (no neighbours / similarity < 0.50) → escalate
  → LLM JSON: { should_escalate, reason, reply }
```

Intent is **not** a hard retrieval filter. A wrong class would drop the useful chunks. Matching intent only adds a small score bump.

Escalation is layered on purpose: cheap rules catch noise and “I want a human”; the LLM still sees billing / anger / missing details after retrieval.

---

## Data (short)

Brand = Amazon (`AmazonHelp`). Original Kaggle dump is multi-brand and messy. I kept English Amazon threads, rebuilt conversations with NetworkX so replies stay together, then downsampled to 15k conversations. Splits are **by conversation**, shared by the classifier and RAG so there is no tweet leaking across train/test.

Classifier trains on **first inbound messages** only (~11k train). RAG indexes those first messages and stores the historical brand replies as the text we ground on.

Intent labels for train/val/test were produced offline by local **Qwen 3 8B** (Ollama), then DistilBERT was trained as a student. That is why I did not use Banking77 — the 77 bank intents do not match Amazon ecommerce.

Golden set: `data/golden_set/golden_evaluation_set.jsonl` (180) + notes in `data/golden_set/GOLDEN_SET.md`.

---

## Layout

```
src/            agent, classifier, retriever, escalator, generator
scripts/        corpus, FAISS, golden run, eval, baselines
data/golden_set/   180 labelled eval examples
dataset/processed/ train/val/test parquets
dataset/vectorstore/  built locally (not in git)
models/         DistilBERT weights (download, not in git)
reports/        committed summaries; large jsonl gitignored
app_streamlit.py
```

---

## Cite / borrow

- Dataset: [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- DistilBERT: Sanh et al., 2019
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Index: FAISS (`IndexFlatIP` + L2-normalised vectors)
- Generator / judge: Groq, `qwen/qwen3.8-27b`
- UI: Streamlit

I used AI coding assistants while building this. The golden **actions** and the 20 human judge scores are mine.

---

## Report

See `report.pdf` (problem framing, baselines, failures, misleading headline, next week, decision log).
