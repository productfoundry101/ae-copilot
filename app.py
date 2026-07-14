"""Streamlit chat UI for the AE Copilot.

All intelligence lives in agent.py; this file only renders. Two audiences,
two layers: AEs see human-readable source chips and a login briefing;
engineers get raw tool calls, method cards and SQL one click away in the
expander. If Streamlit misbehaves on demo day, cli.py is the identical agent
in a terminal.

Run:  streamlit run app.py
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

# Streamlit Community Cloud provides secrets via st.secrets, not a .env file.
# Bridge them into the environment BEFORE importing modules that read config
# at import time (db.py, agent.py). Locally, with no secrets file, this is a
# harmless no-op and .env keeps working via dotenv.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

import agent  # noqa: E402
import briefing  # noqa: E402
import db  # noqa: E402

FEEDBACK_FILE = Path(__file__).parent / "feedback" / "feedback.csv"

st.set_page_config(page_title="AE Copilot", page_icon="🧭", layout="centered")


# --- helpers -----------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def get_overview(ae_email: str) -> dict:
    return briefing.book_overview(ae_email)


def source_label(call: dict) -> str:
    """Human-readable source chip, derived mechanically from the tool call
    (machine-authored provenance; the model never writes these)."""
    n, a = call["name"], call.get("args", {})
    aid = a.get("account_id", "")
    if n == "find_account":
        return f"CRM account lookup: '{a.get('query', '')}'"
    if n == "run_risk_sweep":
        return f"Signal engine: {aid}"
    if n == "scan_book_signals":
        sig = a.get("signal")
        return "Signal engine: full book scan" + (f" ({sig})" if sig else "")
    if n == "get_book_priorities":
        return "Priority ranking (top 5 view)"
    if n == "get_stats":
        return f"CRM aggregates, SQL ({a.get('scope', '?')} scope)"
    if n == "list_my_accounts":
        return "CRM: my accounts"
    if n == "list_my_open_deals":
        extra = [x for x in [a.get("opp_type"),
                             f"≤{a['closing_within_days']}d" if a.get("closing_within_days") is not None else None] if x]
        return "CRM: open deals" + (f" ({', '.join(extra)})" if extra else "")
    if n == "get_opportunities":
        return f"CRM opportunities: {aid}"
    if n == "get_contacts":
        return f"CRM contacts: {aid}"
    if n == "get_activities":
        extra = [x for x in [f"since {a['since']}" if a.get("since") else None,
                             f"'{a['contains']}'" if a.get("contains") else None] if x]
        return f"Activity log: {aid}" + (f" ({', '.join(extra)})" if extra else "")
    if n == "get_usage":
        return f"Product usage: {aid}"
    if n == "get_tickets":
        return f"Support tickets: {aid}"
    if n == "read_knowledge_doc":
        sec = a.get("section")
        return f"Doc: {a.get('name', '')}" + (f" › {sec}" if sec else "")
    if n == "list_knowledge_docs":
        return "Doc library index"
    return n


def save_feedback(ae_email: str, question: str, answer: str,
                  verdict: str, comment: str = ""):
    """Append one feedback row. Wrong-fact reports are the most valuable
    signal we collect: in production they page the team and become eval
    cases; here they land in a CSV ready for that loop."""
    FEEDBACK_FILE.parent.mkdir(exist_ok=True)
    is_new = not FEEDBACK_FILE.exists()
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "ae", "verdict", "comment",
                        "question", "answer"])
        w.writerow([datetime.now().isoformat(), ae_email, verdict,
                    comment, question, answer])


def feedback_widget(idx: int, ae_email: str, question: str, answer: str):
    c1, c2, c3 = st.columns([1, 1, 6])
    if c1.button("👍", key=f"up_{idx}", help="Useful answer"):
        save_feedback(ae_email, question, answer, "like")
        st.toast("Thanks, logged.")
    if c2.button("👎", key=f"down_{idx}", help="Not useful"):
        save_feedback(ae_email, question, answer, "dislike")
        st.toast("Logged. Sorry about that.")
    with c3.popover("⚠️ Report discrepancy"):
        st.caption("Spotted a wrong or missing fact? This goes straight "
                   "to the team and becomes a test case.")
        comment = st.text_area("What was wrong?", key=f"rep_txt_{idx}")
        if st.button("Submit report", key=f"rep_btn_{idx}"):
            save_feedback(ae_email, question, answer, "discrepancy", comment)
            st.success("Reported. Thank you, this is the feedback "
                       "that matters most.")


def render_sources(calls: list[dict]):
    """Source chips (AE layer) + raw calls, method cards and SQL (engineer
    layer). Method cards are machine-authored by the tools and rendered
    without passing through the model, so provenance cannot be misstated."""
    chips = " · ".join(
        f":red[blocked: {source_label(c)}]" if c.get("gated")
        else f":blue[{source_label(c)}]" for c in calls)
    st.markdown(f"Sources: {chips}")
    with st.expander("Method & raw calls"):
        for c in calls:
            st.code(f"{c['name']}({c['args']})", language=None)
            if c.get("method"):
                m = c["method"]
                lines = []
                if m.get("scope"):
                    lines.append(f"Scope: {m['scope']}")
                if "definition" in m:
                    lines.append(f"Definition: {m['definition']}")
                for d in m.get("definitions", []):
                    lines.append(f"Definition: {d}")
                if "truncated" in m:
                    lines.append(f"Truncated: {m['truncated']}")
                if "accounts_scanned" in m:
                    lines.append(f"Coverage: {m['accounts_scanned']} scanned, "
                                 f"{m['accounts_flagged']} flagged")
                if m.get("note"):
                    lines.append(f"Note: {m['note']}")
                if lines:
                    st.caption("  \n".join(str(x) for x in lines))
                for sql in m.get("sql", []):
                    st.code(sql, language="sql")


def start_conversation(question: str):
    """Suggestion clicks start a FRESH conversation: guarantees the answer
    comes from new tool calls, never stale context."""
    st.session_state.history = []
    st.session_state.tool_log = {}
    st.session_state.pending_question = question
    st.rerun()


# --- sidebar: identity, suggestions, environment ----------------------------

with st.sidebar:
    st.title("AE Copilot")
    aes = db.all_aes()
    labels = {e: agent.AE_NAMES.get(e, e) for e in aes}
    ae_email = st.selectbox("Signed in as", aes, format_func=lambda e: labels[e],
                            index=aes.index("lena.koehler@personio.de")
                            if "lena.koehler@personio.de" in aes else 0)
    overview = get_overview(ae_email)
    st.markdown(f"**{overview['total_accounts']} accounts** in your book")
    if st.button("New conversation"):
        st.session_state.pop("history", None)
        st.session_state.pop("tool_log", None)
        st.rerun()
    st.divider()
    st.markdown("**You can ask me things like:**")
    for i, q in enumerate(briefing.suggested_questions(overview)):
        if st.button(q, key=f"sugg_{i}", use_container_width=True):
            start_conversation(q)
    st.divider()
    st.caption("Environment")
    st.caption(f"data mode: {db.DATA_MODE}")
    st.caption(f"as-of date: {db.AS_OF}")
    st.caption(f"model: {agent.PROVIDER}")

# Reset conversation when switching identity.
if st.session_state.get("ae_email") != ae_email:
    st.session_state.ae_email = ae_email
    st.session_state.history = []
    st.session_state.tool_log = {}

history = st.session_state.setdefault("history", [])
tool_log = st.session_state.setdefault("tool_log", {})

# --- welcome briefing (deterministic, only on an empty conversation) --------

if not history:
    first_name = agent.AE_NAMES.get(ae_email, ae_email).split()[0]
    st.subheader(f"Welcome, {first_name}")
    s = overview["by_status"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Accounts", overview["total_accounts"])
    c1.caption(f"{s.get('customer', 0)} customers · {s.get('prospect', 0)} "
               f"prospects · {s.get('churned', 0)} churned")
    c2.metric("Open pipeline", briefing.eur(overview["pipeline_eur"]))
    c2.caption(f"{overview['pipeline_deals']} open deals")
    c3.metric("Renewals ≤ 60 days", overview["renewals_60d"])
    c3.caption(f"as of {db.AS_OF}")
    if overview["attention"]:
        st.markdown("**Needs attention**")
        for i, acct in enumerate(overview["attention"]):
            col_a, col_b = st.columns([2, 5])
            if col_a.button(acct["name"], key=f"attn_{i}",
                            use_container_width=True):
                start_conversation(
                    f"Give me the full risk rundown on {acct['name']}")
            col_b.caption(acct["reason"])
    st.caption("Click an account above, or ask anything below.")

# --- conversation ------------------------------------------------------------

for i, msg in enumerate(history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if i in tool_log and tool_log[i]:
                render_sources(tool_log[i])
            question = history[i - 1]["content"] if i > 0 else ""
            feedback_widget(i, ae_email, question, msg["content"])

question = st.chat_input("Ask about an account, your pipeline, or the playbook")
if not question:
    question = st.session_state.pop("pending_question", None)

if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    # Flattened tool calls from all earlier turns power the account gate.
    prior_calls = [c for i in sorted(tool_log) for c in tool_log[i]]
    with st.chat_message("assistant"):
        with st.spinner("Checking the data..."):
            result = agent.run_turn(history, ae_email,
                                    prior_tool_calls=prior_calls)
    history.append({"role": "assistant", "content": result["answer"]})
    tool_log[len(history) - 1] = result["tool_calls"]
    st.rerun()  # re-render so the new answer gets its sources + feedback row
