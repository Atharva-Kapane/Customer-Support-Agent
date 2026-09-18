"""
Amazon Support Agent — demo + analytics.

    python -m streamlit run app_streamlit.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

SUMMARY_PATH = Path("reports/eval_summary.json")
BASELINE_PATH = Path("reports/baseline_summary.json")

st.set_page_config(
    page_title="Amazon Support Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    html, body, .stApp {
        background: #EFE8DC;
        color: #1A1510;
        font-size: 16px;
    }
    .block-container {
        max-width: 100% !important;
        padding: 0.9rem 1.2rem 1.6rem 1.2rem !important;
    }
    h1 { font-size: 1.85rem !important; color: #1A1510 !important; font-weight: 800 !important; }
    h2, h3 { color: #1A1510 !important; font-weight: 750 !important; }
    p, label, .stCaption { color: #3A3228 !important; }
    [data-testid="stHeader"] { background: transparent; }

    /* Tabs as boxes */
    [data-testid="stTabs"] [role="tablist"] {
        gap: 12px !important;
        background: #D6CBB8 !important;
        padding: 12px !important;
        border: 2px solid #2A2218 !important;
        border-radius: 14px !important;
        margin-bottom: 14px !important;
    }
    button[data-baseweb="tab"] {
        background: #2A2218 !important;
        color: #FFF8EE !important;
        border: 2px solid #2A2218 !important;
        border-radius: 10px !important;
        min-height: 48px !important;
        padding: 10px 26px !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span,
    button[data-baseweb="tab"] div {
        color: #FFF8EE !important;
        font-size: 1.12rem !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #111111 !important;
        border: 3px solid #C4A574 !important;
        box-shadow: 3px 3px 0 #C4A574 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span {
        color: #FFFFFF !important;
    }

    /* Cream inspect buttons — never black-on-black */
    .stButton > button {
        background: #FFF8EE !important;
        color: #1A1510 !important;
        border: 2px solid #8B7355 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 44px !important;
        font-size: 0.95rem !important;
    }
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #1A1510 !important;
        font-weight: 700 !important;
    }
    .stButton > button:hover {
        background: #F3E6D0 !important;
        border-color: #5C4A36 !important;
        color: #111111 !important;
    }

    /* Chat input: white field, dark text */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div {
        background: #FFFFFF !important;
        border: 2px solid #8B7355 !important;
        border-radius: 14px !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"],
    [data-testid="stChatInput"] input {
        background: #FFFFFF !important;
        color: #111111 !important;
        caret-color: #111111 !important;
        font-size: 1.02rem !important;
        -webkit-text-fill-color: #111111 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #7A6A58 !important;
        -webkit-text-fill-color: #7A6A58 !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"] button {
        background: #2A2218 !important;
        color: #FFF8EE !important;
        border: 0 !important;
    }
    [data-testid="stChatInput"] button svg {
        fill: #FFF8EE !important;
        stroke: #FFF8EE !important;
    }

    [data-testid="stMetric"] {
        background: #FFF8EE;
        border: 2px solid #C9B896;
        border-radius: 12px;
        padding: 12px 14px;
    }
    [data-testid="stMetricLabel"] { color: #5C4A36 !important; font-weight: 700 !important; }
    [data-testid="stMetricValue"] { color: #1A1510 !important; font-weight: 800 !important; }

    .chip {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 12.5px;
        background: #FFF8EE;
        border: 1px solid #D4C4A8;
        border-left: 5px solid #C4A574;
        padding: 7px 10px;
        margin: 4px 0;
        border-radius: 6px;
        color: #1A1510;
        word-break: break-word;
    }
    .chip-ok { border-left-color: #2F7A4A; }
    .chip-warn { border-left-color: #C47B4A; }
    .chip-bad { border-left-color: #B85C5C; }

    .escalate-box {
        background: #FBE8DE;
        border: 3px solid #C45A3A;
        border-radius: 14px;
        padding: 16px 18px;
        margin-top: 10px;
        color: #1A1510;
        line-height: 1.5;
    }
    .reply-box {
        background: #E4F3E9;
        border: 3px solid #2F7A4A;
        border-radius: 14px;
        padding: 16px 18px;
        margin-top: 10px;
        color: #1A1510;
        line-height: 1.5;
    }

    /* WhatsApp-style thread */
    .wa-thread {
        background: #D7C9B3;
        border: 2px solid #B9A78C;
        border-radius: 16px;
        padding: 14px 12px 16px 12px;
        min-height: 420px;
        max-height: 58vh;
        overflow-y: auto;
    }
    .wa-row { display: flex; margin: 8px 4px; }
    .wa-row.user { justify-content: flex-end; }
    .wa-row.bot { justify-content: flex-start; }
    .wa-bubble {
        max-width: 78%;
        padding: 10px 14px;
        border-radius: 14px;
        line-height: 1.45;
        font-size: 0.98rem;
        color: #111111;
        white-space: pre-wrap;
        word-break: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.12);
    }
    .wa-bubble.user {
        background: #DCF8C6;
        border-bottom-right-radius: 4px;
        color: #111111;
    }
    .wa-bubble.bot {
        background: #FFF8EE;
        border-bottom-left-radius: 4px;
        color: #111111;
    }
    .wa-bubble.esc {
        background: #F8D7C8;
        border: 1px solid #C45A3A;
        color: #111111;
    }
    .wa-meta {
        font-size: 11px;
        font-weight: 700;
        color: #5C4A36;
        margin-bottom: 4px;
    }
    .wa-typing {
        background: #FFF8EE;
        color: #5C4A36;
        font-style: italic;
        border-bottom-left-radius: 4px;
    }
    .wa-empty {
        text-align: center;
        color: #5C4A36;
        padding: 48px 12px;
        font-weight: 600;
    }

    @keyframes pulse {
        0% { opacity: 0.35; }
        50% { opacity: 1; }
        100% { opacity: 0.35; }
    }
    .dots { animation: pulse 1.1s infinite; font-weight: 800; letter-spacing: 2px; }
</style>
""",
    unsafe_allow_html=True,
)


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def ensure_agent():
    if "agent" not in st.session_state:
        from src.agent import SupportAgent
        from src.intent_classifier import IntentClassifier

        with st.spinner("Loading classifier + FAISS (first time only)…"):
            st.session_state.agent = SupportAgent(
                intent_classifier=IntentClassifier()
            )


def chip(text: str, kind: str = "") -> str:
    cls = "chip"
    if kind:
        cls += f" chip-{kind}"
    return f'<div class="{cls}">{html.escape(text)}</div>'


def render_thread(messages: list, typing: bool) -> str:
    if not messages and not typing:
        return '<div class="wa-thread"><div class="wa-empty">No messages yet.</div></div>'

    parts = ['<div class="wa-thread">']
    for msg in messages:
        role = msg.get("role", "assistant")
        raw = str(msg.get("content") or "")
        kind = msg.get("kind", "")
        if role == "user":
            parts.append(
                '<div class="wa-row user"><div class="wa-bubble user">'
                '<div class="wa-meta">You</div>'
                f"{html.escape(raw)}"
                "</div></div>"
            )
        else:
            bubble = "esc" if kind == "escalate" else "bot"
            label = "Escalated to human" if kind == "escalate" else "Amazon agent"
            parts.append(
                f'<div class="wa-row bot"><div class="wa-bubble {bubble}">'
                f'<div class="wa-meta">{label}</div>'
                f"{html.escape(raw)}"
                "</div></div>"
            )
    if typing:
        parts.append(
            '<div class="wa-row bot"><div class="wa-bubble wa-typing">'
            '<div class="wa-meta">Amazon agent</div>'
            'Agent is typing <span class="dots">● ● ●</span>'
            "</div></div>"
        )
    parts.append("</div>")
    return "".join(parts)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "event_log" not in st.session_state:
    st.session_state.event_log = []
if "inspect" not in st.session_state:
    st.session_state.inspect = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

tab_chat, tab_analytics = st.tabs(["  Live agent  ", "  Analytics  "])

# =====================================================================
# TAB 1 — Live agent
# =====================================================================
with tab_chat:
    st.title("Amazon Support Agent")
    st.caption("Intent → rules → RAG → auto-reply or escalate")

    left, center, right = st.columns([1.05, 2.5, 1.2], gap="medium")

    with left:
        st.subheader("Pipeline log")
        st.caption("One line per step")
        if not st.session_state.event_log:
            st.markdown(chip("Waiting for a customer message…"), unsafe_allow_html=True)
        for e in st.session_state.event_log[-30:]:
            st.markdown(chip(e["text"], e.get("kind", "")), unsafe_allow_html=True)

    with center:
        st.subheader("Chat")
        typing = st.session_state.pending_query is not None
        st.markdown(
            render_thread(st.session_state.messages, typing),
            unsafe_allow_html=True,
        )

        user_text = st.chat_input("Message the Amazon agent…")
        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.pending_query = user_text
            st.session_state.event_log = [
                {"text": f"IN  {user_text[:100]}", "kind": ""}
            ]
            st.rerun()

        if st.session_state.pending_query:
            query = st.session_state.pending_query
            ensure_agent()
            try:
                result = st.session_state.agent.run(query)
                st.session_state.last_result = result

                intent = result["intent"]["predicted"]
                conf = float(result["intent"]["confidence"])
                action = result["action"]
                reason = result.get("reason") or ""
                reply = result.get("reply") or ""
                decided = (result.get("escalation") or {}).get("decided_at") or "?"
                n_ctx = len(result.get("retrieved_contexts") or [])
                top_sim = (result.get("retrieval") or {}).get("top_similarity")

                st.session_state.event_log.append(
                    {
                        "text": f"INT  {intent}  conf={conf:.2f}",
                        "kind": "ok" if conf >= 0.5 else "warn",
                    }
                )
                if top_sim is not None:
                    st.session_state.event_log.append(
                        {
                            "text": f"RAG  k={n_ctx}  top_sim={top_sim:.3f}",
                            "kind": "ok",
                        }
                    )
                else:
                    st.session_state.event_log.append(
                        {"text": "RAG  skipped (early gate)", "kind": "warn"}
                    )
                st.session_state.event_log.append(
                    {
                        "text": f"DEC  {action}  @ {decided}",
                        "kind": "ok" if action == "auto_reply" else "bad",
                    }
                )
                if reason:
                    st.session_state.event_log.append(
                        {"text": f"RSN  {reason[:120]}", "kind": ""}
                    )

                if action == "auto_reply":
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "kind": "reply",
                            "content": reply or "(empty reply)",
                        }
                    )
                else:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "kind": "escalate",
                            "content": reason or "No reason provided.",
                        }
                    )
            except Exception as e:
                st.session_state.event_log.append({"text": f"ERR  {e}", "kind": "bad"})
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "kind": "escalate",
                        "content": f"Error: {e}",
                    }
                )
            st.session_state.pending_query = None
            st.rerun()

        res = st.session_state.last_result
        if res is not None and st.session_state.pending_query is None:
            action = res.get("action")
            if action == "escalate":
                intent = res.get("intent", {}) or {}
                conf = float(intent.get("confidence") or 0)
                decided_at = (res.get("escalation") or {}).get("decided_at") or "—"
                st.markdown(
                    f"""
                    <div class="escalate-box">
                      <strong>Escalation room</strong><br/>
                      This message was handed to a human.<br/><br/>
                      <strong>Reason:</strong> {html.escape(str(res.get("reason") or "—"))}<br/>
                      <strong>Decided at:</strong> {html.escape(str(decided_at))}<br/>
                      <strong>Intent:</strong> {html.escape(str(intent.get("predicted") or "—"))}
                      ({conf:.2f})
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif action == "auto_reply":
                st.markdown(
                    f"""
                    <div class="reply-box">
                      <strong>Auto-reply sent</strong><br/><br/>
                      {html.escape(str(res.get("reply") or "—"))}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with right:
        st.subheader("Inspect")
        st.caption("Open one artifact")
        b1, b2 = st.columns(2)
        if b1.button("Retrieved docs", width="stretch"):
            st.session_state.inspect = "contexts"
        if b2.button("Generation", width="stretch"):
            st.session_state.inspect = "generation"
        if st.button("Full decision log", width="stretch"):
            st.session_state.inspect = "full"
        if st.button("Clear inspect", width="stretch"):
            st.session_state.inspect = None

        st.markdown("---")
        res = st.session_state.last_result
        mode = st.session_state.inspect

        if mode is None:
            st.info("Send a message, then open a panel.")
        elif res is None:
            st.warning("No result yet.")
        elif mode == "contexts":
            st.markdown("#### Retrieved contexts")
            ctxs = res.get("retrieved_contexts") or []
            if not ctxs:
                st.write("None (early escalate or empty retrieval).")
            for i, c in enumerate(ctxs, 1):
                with st.expander(f"Chunk {i}"):
                    st.text(c if isinstance(c, str) else str(c))
        elif mode == "generation":
            st.markdown("#### Generation / escalation")
            st.json(
                {
                    "action": res.get("action"),
                    "reason": res.get("reason"),
                    "reply": res.get("reply"),
                    "escalation": res.get("escalation"),
                    "generation": res.get("generation"),
                    "retrieval": res.get("retrieval"),
                }
            )
        elif mode == "full":
            st.markdown("#### Full agent payload")
            st.json(res)

# =====================================================================
# TAB 2 — Analytics
# =====================================================================
with tab_analytics:
    st.title("Evaluation analytics")
    st.caption("Golden set · baselines · judge · human agreement")

    summary = load_json(SUMMARY_PATH)
    baseline = load_json(BASELINE_PATH)

    st.subheader("Headline metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Intent accuracy",
        f"{summary['intent_accuracy']:.1%}"
        if summary.get("intent_accuracy") is not None
        else "—",
    )
    c2.metric(
        "Intent macro-F1",
        f"{summary['intent_macro_f1']:.2f}"
        if summary.get("intent_macro_f1") is not None
        else "—",
    )
    c3.metric(
        "Full action accuracy",
        f"{summary['action_accuracy']:.1%}"
        if summary.get("action_accuracy") is not None
        else "—",
    )
    c4.metric(
        "Action macro-F1",
        f"{summary['action_macro_f1']:.2f}"
        if summary.get("action_macro_f1") is not None
        else "—",
    )

    st.subheader("Why RAG + LLM beats baselines")
    rows = []
    if baseline.get("trivial"):
        rows.append(
            {
                "System": "Trivial (always escalate)",
                "Action accuracy": baseline["trivial"]["action_accuracy"],
                "Action macro-F1": baseline["trivial"]["action_macro_f1"],
            }
        )
    if baseline.get("simple"):
        rows.append(
            {
                "System": "Simple (intent + template)",
                "Action accuracy": baseline["simple"]["action_accuracy"],
                "Action macro-F1": baseline["simple"]["action_macro_f1"],
            }
        )
    if summary.get("action_accuracy") is not None:
        rows.append(
            {
                "System": "Full agent (rules + RAG + LLM)",
                "Action accuracy": summary["action_accuracy"],
                "Action macro-F1": summary.get("action_macro_f1"),
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        table_col, chart_col = st.columns([1.15, 1], gap="large")
        with table_col:
            st.dataframe(
                df.style.format(
                    {
                        "Action accuracy": "{:.1%}",
                        "Action macro-F1": "{:.3f}",
                    }
                ),
                width="stretch",
                hide_index=True,
                height=168,
            )
        with chart_col:
            chart_df = df.set_index("System")[["Action accuracy"]]
            st.bar_chart(chart_df, color="#2F7A4A", height=260)
        st.markdown(
            """
- **Trivial (~38%)**: always escalate.
- **Simple (~69%)**: early rules + canned templates.
- **Full agent (~82%)**: retrieval + LLM decision.
"""
        )
    else:
        st.warning(
            "Missing reports/eval_summary.json or reports/baseline_summary.json."
        )

    st.subheader("LLM-as-judge (auto-replies only)")
    j1, j2, j3, j4, j5 = st.columns(5)
    j1.metric("n judged", summary.get("judge_n") or "—")
    j2.metric(
        "Overall",
        f"{summary['judge_avg_overall']:.2f}"
        if summary.get("judge_avg_overall") is not None
        else "—",
    )
    j3.metric(
        "Helpful",
        f"{summary['judge_avg_helpfulness']:.2f}"
        if summary.get("judge_avg_helpfulness") is not None
        else "—",
    )
    j4.metric(
        "Grounded",
        f"{summary['judge_avg_groundedness']:.2f}"
        if summary.get("judge_avg_groundedness") is not None
        else "—",
    )
    j5.metric(
        "Tone",
        f"{summary['judge_avg_tone']:.2f}"
        if summary.get("judge_avg_tone") is not None
        else "—",
    )
    st.caption(
        f"Judge model: `{summary.get('judge_model') or '—'}`. "
        "Tone/safety tend to saturate; overall is optimistic vs human sample."
    )

    agree, meaning = st.columns(2, gap="large")
    with agree:
        st.subheader("Human ↔ judge (n=20)")
        st.markdown(
            """
| Axis | MAE | Exact | Within ±1 |
|------|-----|-------|-----------|
| helpfulness | 0.45 | 55% | 100% |
| groundedness | 0.50 | 60% | 90% |
| tone | 0.00 | 100% | 100% |
| safety | 0.20 | 80% | 100% |
| **overall** | **0.60** | **55%** | **85%** |

Judge is reliable on tone/safety, lenient on overall for polite deflection.
"""
        )
    with meaning:
        st.subheader('What "good" means')
        st.markdown(
            """
For **Amazon on Twitter**, a good agent:

1. Labels the issue into a small intent set.
2. Auto-replies only when a **safe, grounded** message is possible.
3. Escalates billing, account, fraud, and explicit human requests — **with a reason**.

We did **not** build live order lookup, refunds, or multi-turn memory.
"""
        )
