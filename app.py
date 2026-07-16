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
import html
import os
import re
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


# The data source(s) each tool actually reads, in AE-facing names. Used to
# tell the AE which parts of the CRM / enablement library an answer came from.
_TOOL_SOURCES = {
    "find_account": ["Accounts"],
    "list_my_accounts": ["Accounts"],
    "list_my_open_deals": ["Opportunities"],
    "get_book_priorities": ["Accounts", "Opportunities"],
    "get_stats": ["Accounts", "Opportunities"],
    "run_risk_sweep": ["Accounts", "Opportunities", "Contacts", "Usage",
                       "Support tickets", "Activities"],
    "scan_book_signals": ["Accounts", "Opportunities", "Contacts", "Usage",
                          "Support tickets", "Activities"],
    "get_opportunities": ["Opportunities"],
    "get_contacts": ["Contacts"],
    "get_activities": ["Activities"],
    "get_usage": ["Usage"],
    "get_tickets": ["Support tickets"],
    "list_knowledge_docs": [],  # just browsing the index, not a real source
}

_DOC_NAMES = {
    "sales_playbook": "Sales playbook", "icp": "ICP",
    "battlecard_workday": "Workday battlecard", "battlecard_hibob": "HiBob battlecard",
    "objection_handling": "Objection handling", "pricing_cheatsheet": "Pricing guide",
    "customer_case_studies": "Case studies",
}

TOOL_NAMES = {t["name"] for t in agent.tools.TOOLS}


def call_sources(call: dict) -> list[str]:
    """The AE-facing data source(s) a single tool call touched."""
    n, a = call["name"], call.get("args", {})
    if n == "read_knowledge_doc":
        return [_DOC_NAMES.get(a.get("name", ""), a.get("name", "Enablement doc"))]
    return list(_TOOL_SOURCES.get(n, []))


def answer_sources(calls: list[dict]) -> list[str]:
    """Deduped, ordered set of sources across every (non-blocked) call, so the
    AE sees 'Accounts, Opportunities, HiBob battlecard' rather than a list of
    tool names."""
    seen, out = set(), []
    for c in calls:
        if c.get("gated"):
            continue
        for s in call_sources(c):
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


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


# Method lines that are implementation trivia, not business meaning: hide
# these from the AE view (they stay in the raw payload / traces for engineers).
_NOISE = ("computed in sql", "by the model", "authoritative", "computed in code",
          "connector", "deterministic coded", "network-policy", "recognises it")


def _is_noise(text: str) -> bool:
    t = str(text).lower()
    return any(k in t for k in _NOISE)


def _scope_headline(scope: str) -> str:
    """The plain 'who/what' part of a scope string, before any filter clause."""
    if not scope:
        return ""
    return re.split(r",\s*(type=|closing|filtered)", scope)[0].strip().rstrip(",")


def _constraints_from_scope(scope: str) -> list[str]:
    """Pull out constraint clauses (date windows, type filters, keyword
    filters) so they can be shown as explicit 'constraints applied'."""
    if not scope:
        return []
    out = []
    for pat, label in [
        (r"closing on or before ([0-9-]+)[^,]*", "Only deals closing on or before \\1"),
        (r"type=(\w[\w ]*)", "Only \\1 deals"),
        (r"since ([0-9-]+)", "Only activity since \\1"),
        (r"containing '([^']+)'", "Only entries mentioning '\\1'"),
    ]:
        m = re.search(pat, scope)
        if m:
            out.append(re.sub(pat, label, m.group(0)))
    return out


# Normalise any inline citation the model wrote to a clean, AE-facing
# [source: X] tag. Maps internal tool names AND bare table/doc names to the
# data source an AE recognises; tool names never reach the AE.
_CITE_MAP = {
    "run_risk_sweep": "risk signals", "scan_book_signals": "risk signals",
    "signals": "risk signals", "get_book_priorities": "priority ranking",
    "get_stats": "CRM records", "get_opportunities": "opportunities",
    "get_contacts": "contacts", "get_activities": "activities",
    "get_usage": "usage", "get_tickets": "tickets",
    "find_account": "accounts", "list_my_accounts": "accounts",
    "list_my_open_deals": "opportunities",
    "opportunities": "opportunities", "activities": "activities",
    "usage": "usage", "tickets": "tickets", "contacts": "contacts",
    "accounts": "accounts",
    "sales_playbook": "playbook", "playbook": "playbook", "icp": "ICP",
    "battlecard_hibob": "HiBob battlecard", "battlecard_workday": "Workday battlecard",
    "objection_handling": "objection handling", "pricing_cheatsheet": "pricing guide",
    "customer_case_studies": "case studies",
}


def _cite_sub(m: "re.Match") -> str:
    inner = m.group(1)
    if m.string[m.end():m.end() + 1] == "(":
        return m.group(0)  # markdown link [text](url) — leave untouched
    key = re.split(r"[:\s]", inner.strip(), 1)[0].strip().lower()
    if inner.strip().lower().startswith("source:"):
        return f":grey[[{inner.strip()}]]"
    src = _CITE_MAP.get(key)
    return f":grey[[source: {src}]]" if src else m.group(0)


def format_answer(text: str) -> str:
    """Rewrite inline citations to a uniform grey [source: X] tag. Tool names
    and bare table names both map to the data source an AE recognises; no
    internal tool name ever reaches the answer."""
    return re.sub(r"\[([^\]\n]+)\]", _cite_sub, text)


# Human operation labels per tool: the step list reads as "what happened",
# with the data sources demoted to a per-step "reads:" detail.
_TOOL_OPS = {
    "find_account": "Account lookup",
    "list_my_accounts": "Fetched your account list",
    "list_my_open_deals": "Fetched open deals",
    "get_book_priorities": "Ranked book by risk (top 5 view)",
    "get_stats": "Computed counts & totals",
    "run_risk_sweep": "Risk signal sweep",
    "scan_book_signals": "Scanned every account for signals",
    "get_opportunities": "Fetched opportunities",
    "get_contacts": "Fetched contacts",
    "get_activities": "Fetched activity log",
    "get_usage": "Fetched product usage",
    "get_tickets": "Fetched support tickets",
    "list_knowledge_docs": "Browsed the document library",
    "read_knowledge_doc": "Read document",
}

_ACCOUNT_ARG_TOOLS = {"run_risk_sweep", "get_opportunities", "get_contacts",
                      "get_activities", "get_usage", "get_tickets"}


@st.cache_data(show_spinner=False)
def _acct_name(account_id: str) -> str:
    acct = db.get_account(account_id)
    return acct["COMPANY_NAME"] if acct else account_id


def _grouped_calls(calls: list[dict]) -> list[dict]:
    """Collapse repeated calls to the same tool into one step ('Risk signal
    sweep × 2: Halcyon, Cobalt'). Gated (blocked) calls group separately so
    the pause is visible as its own step."""
    groups: list[dict] = []
    index: dict = {}
    for c in calls:
        key = (c["name"], bool(c.get("gated")))
        if key in index:
            groups[index[key]]["calls"].append(c)
        else:
            index[key] = len(groups)
            groups.append({"name": c["name"], "gated": bool(c.get("gated")),
                           "calls": [c]})
    return groups


def _step_detail(name: str, group: list[dict]) -> str:
    """The step's specifics, derived from call args (never model prose)."""
    def uniq(items):
        return list(dict.fromkeys(x for x in items if x))
    if name == "find_account":
        terms = uniq(f"“{c['args'].get('query', '')}”" for c in group)
        return "searched " + ", ".join(terms) if terms else ""
    if name in _ACCOUNT_ARG_TOOLS:
        names = uniq(_acct_name(str(c["args"].get("account_id") or ""))
                     for c in group if c["args"].get("account_id"))
        return ", ".join(names)
    if name == "read_knowledge_doc":
        items = []
        for c in group:
            doc = _DOC_NAMES.get(c["args"].get("name", ""),
                                 c["args"].get("name", ""))
            sec = c["args"].get("section")
            items.append(f"{doc} › {sec}" if sec else doc)
        return "; ".join(dict.fromkeys(items))
    if name == "get_stats":
        scopes = {"mine": "your book", "company": "company-wide"}
        return ", ".join(uniq(scopes.get(c["args"].get("scope", ""), "")
                              for c in group))
    if name == "scan_book_signals":
        sigs = uniq(c["args"].get("signal") for c in group)
        return ("filter: " + ", ".join(sigs)) if sigs else "all signal types"
    return ""


def render_sources(calls: list[dict]):
    """AE view: which data sources the answer used (deduped, plain names), plus
    an expandable 'How this was calculated' panel. Panel layout: the answer's
    contract first (scope, definitions & constraints, coverage, consolidated
    across all calls), then numbered steps in execution order, each with a
    collapsed technical drill-down (raw calls + the SQL actually run).
    Everything here is machine-authored; the model never writes it."""
    sources = answer_sources(calls)
    blocked = [c for c in calls if c.get("gated")]
    chips = " · ".join(f":blue[{s}]" for s in sources) or ":grey[none]"
    line = f"Sources: {chips}"
    if blocked:
        line += "  ·  :red[⚠ asked to clarify which account]"
    st.markdown(line)

    with st.expander("How this was calculated"):
        methods = [c.get("method") or {} for c in calls if not c.get("gated")]

        # --- the answer's contract, consolidated and deduped ---------------
        acct_names = []
        for c in calls:
            if c["name"] in _ACCOUNT_ARG_TOOLS and not c.get("gated"):
                aid = str((c.get("args") or {}).get("account_id") or "")
                if aid and _acct_name(aid) not in acct_names:
                    acct_names.append(_acct_name(aid))
        scope_bits = []
        if acct_names:
            label = "account" if len(acct_names) == 1 else "accounts"
            scope_bits.append(f"{label}: {', '.join(acct_names)}")
        for m in methods:
            h = _scope_headline(m.get("scope", ""))
            # per-account scopes (contain a raw id) are covered by the line
            # above; keep only book/company-level headlines
            if h and "ACC-" not in h and h not in scope_bits:
                scope_bits.append(h)
        defs, constraints, coverage = [], [], []
        for m in methods:
            for d in (([m["definition"]] if m.get("definition") else [])
                      + list(m.get("definitions", []))):
                if not _is_noise(d) and d not in defs:
                    defs.append(d)
            for cst in _constraints_from_scope(m.get("scope", "")):
                if cst not in constraints:
                    constraints.append(cst)
            if "accounts_scanned" in m:
                cov = (f"{m['accounts_scanned']} accounts checked, "
                       f"{m['accounts_flagged']} flagged")
                if cov not in coverage:
                    coverage.append(cov)

        if scope_bits:
            st.markdown("**Scope:** " + " · ".join(scope_bits))
        if coverage:
            st.caption(" · ".join(coverage))
        if defs or constraints:
            st.markdown("**Definitions & constraints**")
            for d in defs:
                st.caption(f"• {d}")
            for cst in constraints:
                st.caption(f"• {cst}")
        st.divider()

        # --- the steps, in execution order ----------------------------------
        st.markdown("**Steps**")
        for i, g in enumerate(_grouped_calls(calls), 1):
            name, group = g["name"], g["calls"]
            if g["gated"]:
                st.markdown(f"{i}\\. ⚠ **Paused** — asked which account "
                            "you meant before touching the data")
                note = (group[0].get("method") or {}).get("note", "")
                if note:
                    st.caption(note)
                continue
            times = f" × {len(group)}" if len(group) > 1 else ""
            label = _TOOL_OPS.get(name, name) + times
            detail = _step_detail(name, group)
            st.markdown(f"{i}\\. **{label}**" + (f": {detail}" if detail else ""))
            reads = []
            for c in group:
                for s in call_sources(c):
                    if s not in reads:
                        reads.append(s)
            notes = []
            for c in group:
                note = (c.get("method") or {}).get("note", "")
                if note and not _is_noise(note) and note not in notes:
                    notes.append(note)
            sub = []
            if reads and name != "read_knowledge_doc":
                sub.append("reads: " + ", ".join(reads))
            sub.extend(notes)
            if sub:
                st.caption("  \n".join(sub))

            # technical drill-down: raw calls + the SQL actually run,
            # collapsed (HTML details; Streamlit forbids nested expanders)
            raw = [f"{c['name']}({c.get('args')})" for c in group]
            sqls = []
            for c in group:
                for q in (c.get("method") or {}).get("sql", []):
                    if q not in sqls:
                        sqls.append(q)
            body = "".join(
                f"<div style='margin:4px 0;font-family:monospace;font-size:12px;"
                f"opacity:0.55'>{html.escape(r)}</div>" for r in raw)
            body += "".join(
                f"<div style='margin:4px 0;font-family:monospace;font-size:12px;"
                f"white-space:pre-wrap;opacity:0.8'>{html.escape(q)}</div>"
                for q in sqls)
            summary = f"Technical detail: {len(raw)} call(s)"
            if sqls:
                summary += f", {len(sqls)} SQL " + ("query" if len(sqls) == 1 else "queries")
            st.markdown(
                f"<details><summary style='cursor:pointer;opacity:0.6;"
                f"font-size:13px'>{summary}</summary>"
                f"<div style='padding:6px 0'>{body}</div></details>",
                unsafe_allow_html=True)


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
    # No auth: "signing in" is picking an email from the AEs the CRM knows
    # about. ae_email is the one identity variable threaded through every
    # call below (tools, gate, digest) for the rest of this session.
    aes = db.all_aes()
    labels = {e: agent.AE_NAMES.get(e, e) for e in aes}
    ae_email = st.selectbox("Signed in as", aes, format_func=lambda e: labels[e],
                            index=aes.index("lena.koehler@personio.de")
                            if "lena.koehler@personio.de" in aes else 0)
    # Cached 10 min (see get_overview above): cheap on repeated reruns,
    # since Streamlit re-executes this whole script on every interaction.
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

# setdefault: session_state persists across reruns, but only after the first
# rerun where these keys exist. history = chat turns; tool_log = {history
# index -> tool calls that produced that answer}, used by render_sources().
history = st.session_state.setdefault("history", [])
tool_log = st.session_state.setdefault("tool_log", {})

# --- welcome briefing (deterministic, only on an empty conversation) --------
# Shown once per fresh conversation. Every number here comes from SQL/signals
# (briefing.book_overview), never the model: fast, free, identical on reload.

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
        if msg["role"] == "assistant":
            st.markdown(format_answer(msg["content"]))
            if i in tool_log and tool_log[i]:
                render_sources(tool_log[i])
            question = history[i - 1]["content"] if i > 0 else ""
            feedback_widget(i, ae_email, question, msg["content"])
        else:
            st.markdown(msg["content"])

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
