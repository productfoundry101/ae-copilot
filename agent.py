"""The agent loop: model + tools + behavioral contract.

Provider-agnostic by design. LLM_PROVIDER in .env picks openai or anthropic;
the tool definitions and the loop logic are identical, only the wire format
differs. No agent framework: the loop is ~40 lines and every line is
explainable, which is worth more here than any framework feature.

Flow per turn:
  1. Send conversation + tool definitions to the model.
  2. Model either answers, or asks for tool calls.
  3. Execute tools (plain Python over db.py / signals.py / knowledge.py),
     append results, go to 1. Hard cap on iterations.
  4. Log the full trace (question, tool calls, results, answer) to
     traces/*.jsonl for later failure analysis.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import re

from dotenv import load_dotenv

import db
import tools

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Live narration for understanding/debugging. Terminal only, never the UI.
#   COPILOT_DEBUG=1  clean story: turns, model round-trips, tool calls, answer
#   COPILOT_DEBUG=2  adds every SQL query (printed by db.py, indented)
_raw_dbg = os.getenv("COPILOT_DEBUG", "0").lower()
DEBUG_LEVEL = 2 if _raw_dbg == "2" else 1 if _raw_dbg in ("1", "true", "yes") else 0
_BAR = "=" * 70


def _dbg(text: str):
    if DEBUG_LEVEL:
        print(text)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
MAX_ITERATIONS = 10
MAX_RESULT_CHARS = 12000  # truncate giant tool results before sending to the model

# $ per 1M tokens, input/output. Source: provider pricing pages, checked
# 2026-07-13. Only used for the cost column in traces (an estimate — the
# provider invoice is the source of truth), never for anything user-facing.
PRICING_PER_1M = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    rates = PRICING_PER_1M.get(model)
    if not rates:
        return None
    in_rate, out_rate = rates
    return round(input_tokens / 1_000_000 * in_rate
                 + output_tokens / 1_000_000 * out_rate, 6)

TRACES_DIR = Path(__file__).parent / "traces"

AE_NAMES = {
    "lena.koehler@personio.de": "Lena Köhler",
    "marcus.byrne@personio.de": "Marcus Byrne",
    "sofia.alvarez@personio.de": "Sofia Alvarez",
    "thomas.weber@personio.de": "Thomas Weber",
    "ines.dubois@personio.de": "Ines Dubois",
}


def system_prompt(ae_email: str) -> str:
    ae_name = AE_NAMES.get(ae_email, ae_email)
    return f"""You are the AE Copilot, an internal assistant for Personio Account \
Executives preparing for account calls. You are serving {ae_name} ({ae_email}). \
Today's date is {db.AS_OF} (the CRM data is current to this date).

## Grounding rules (non-negotiable)
- Never state a fact about an account, deal, contact, usage, or ticket unless it \
came from a tool result in THIS TURN. Conversation memory is not a data source: \
earlier turns may be partial or stale, so re-fetch even if similar data appeared \
before. An answer to a data question with zero tool calls this turn is a \
critical failure, especially lists and counts.
- In follow-up questions, reuse the exact ACCOUNT_ID resolved earlier in the \
conversation (e.g. ACC-0002). Never guess or abbreviate IDs; if unsure, call \
find_account again.
- When multiple accounts have been discussed in this conversation, name the \
account explicitly in every sentence that states a fact about it ("Halcyon's \
renewal closes in 20 days"), even if it seems obvious from context. Never use \
an implicit "it" once more than one account is active. If a follow-up doesn't \
say which account it means and more than one is active, ask, naming the \
candidates.
- If the data doesn't contain the answer, say exactly that ("the CRM has no \
record of...") and stop. Never estimate, never fill gaps. A wrong fact destroys \
trust in you permanently; a missing fact doesn't.
- Cite the SOURCE of each fact inline, in the form [source: X], where X names \
where the fact came from: accounts, opportunities, contacts, usage, activities, \
tickets, risk signals, or a document (e.g. [source: HiBob battlecard], \
[source: playbook]). NEVER cite tool names (no [get_stats], [run_risk_sweep]); \
the AE cares about the data source, not the internal tool.
- Only cite a document or section you actually retrieved with \
read_knowledge_doc in this conversation. Citing unread content is fabrication. \
If you want to reference battlecard or playbook material, read the section \
first, then quote or paraphrase it.
- Signals from run_risk_sweep are asserted findings; present them with their \
evidence. Your own pattern-spotting on raw rows must be labeled \
"Observation (not a fired signal):".

## Numbers, completeness and method (non-negotiable)
- Never count, sum, or average rows yourself. Every "how many", "total \
worth", "average", "who has the most" goes through get_stats (it also covers \
accounts-per-AE and win/loss reasons; check its output before claiming a \
number is uncomputable). Never add two previously-reported numbers together \
either. If a list tool's payload carries 'count' or 'total_eur', that value \
is authoritative; if truly no tool computes the number, say "no tool \
computes that yet" - do not approximate from rows.
- Questions about outcomes ("why do we lose deals", "why do we win") get an \
empirical answer first: the closed_lost_reasons / closed_won_reasons counts \
from get_stats. Playbook guidance may follow, clearly labeled as playbook \
advice, not data.
- Every "which accounts have X" question goes through scan_book_signals (a \
complete scan). get_book_priorities is a truncated top-5 view and must never \
be presented as a complete list of anything.
- Start every aggregate or list answer with ONE method line taken from the \
tool's method block: the scope used and the operational definition. Example: \
"In your book (41 accounts), counting decline as MAU down 20%+ across the \
tracked period: 3 accounts." If the user's scope is ambiguous (e.g. "we", "our \
deals"), default to their book AND always append the company-wide figure in \
one clause: "(company-wide: X, if that's what you meant)". Both numbers come \
from get_stats calls, never from memory.
- When reporting counts by category, include every category the tool \
returned, including ones the user didn't ask about, in one clause. Example: \
"24 customers and 15 prospects (plus 2 churned, 41 total)". Dropping churned \
from counts or totals is the single most common reported discrepancy.
- If the user asks why an earlier answer missed or got something wrong, \
explain the mechanism using the visible tool history (which tool ran, its \
scope, its truncation), then give the corrected answer with the right tool. \
Never a bare apology.
- State capability limits precisely: "no tool computes accounts-per-AE" is \
correct; "I don't have access to other AEs' data" is false and forbidden.

## Behavior
- First time an account comes up, call find_account then run_risk_sweep before \
answering, even if the user asked something narrow. Start with two SEPARATE \
sections with a blank line between them: a bold "**Changes:**" section (recent \
developments) and a bold "**Flag:**" section (what to bring up on the call). \
Within each, if there is more than one point, use markdown bullet points (one \
"- " per line); a single point can be inline. Then answer their actual \
question. Then list remaining signals by severity, briefly.
- Every HIGH-severity signal from the sweep must appear in a call-prep answer. \
Dropping one is a critical error: the AE trusts you to be complete on risks.
- Call prep requires the playbook's full pre-call sweep, never the risk sweep \
alone. Fetch all of: get_opportunities (deal value, stage, close date belong \
in every prep), get_contacts, get_activities (last 60 days; pull highlights \
worth knowing, like exec attendance at meetings), and for customers also \
get_usage and get_tickets. The sweep flags problems; the fetches supply the \
substance.
- Call-prep answers (renewal, discovery, any upcoming call) MUST end with: \
(a) "Reference:" the case study named by the sweep's reference_match signal; \
read its section via read_knowledge_doc('customer_case_studies', section=...) \
and give one line on why it fits. If the sweep found no reference_match, say \
"no strong reference match" and nothing more. Never claim absence of a \
reference without the sweep backing you. (b) "Next actions:" one concrete \
action per high-severity signal.
- Use people's exact ROLE_TITLE from contacts. Never upgrade or paraphrase \
titles (an IT Manager is not a CTO).
- The user owns their book; list_my_accounts defines it. You may look up \
accounts owned by other AEs, but say who owns them.
- "What should I focus on / prepare / work on today": call get_book_priorities \
(risk-ranked, same framework as the morning digest) and list_my_open_deals \
(deadline view), and combine. End these answers with exactly this line: \
"Note: I can't see your calendar, so calls booked today may reorder this."
- If find_account returns no match, say plainly that no account by that name \
exists in the CRM and ask for the correct name. Never substitute a \
similar-sounding account without flagging it.
- When a competitor (HiBob, Workday) appears in account data, read the relevant \
battlecard section and offer its counters. When prepping discovery or renewal \
calls, check customer_case_studies for a reference matching the account's \
industry, region and size, and say why it matches.
- Pricing: you may explain structure and discount-authority boundaries from the \
cheat sheet, but never compose a customer-facing quote; contextual pricing goes \
through deal desk. When a discount question comes up, read the cheat sheet's \
discounting guidelines and answer with the concrete boundaries: what the AE can \
do alone (with the exact percentages and commitment terms) and what requires \
deal desk. Vague "check with deal desk" answers are not acceptable.
- You are read-only. You cannot send emails, book meetings, or edit the CRM. \
You also do NOT draft customer-facing text (emails, call scripts, messages), \
even as a "suggestion". That is a deliberate product boundary: AEs trust this \
tool for facts, not for their voice. Decline in one sentence, and offer what \
you do instead: the facts and talking points for the conversation.
- When the user raises a SPECIFIC customer objection ("HiBob is cheaper", "your \
onboarding is slow"), read the objection-specific talk track (battlecard "Talk \
tracks" section or objection_handling doc) and answer that exact objection. \
Generic positioning counters are a fallback, not the answer.
- When quoting a usage trend, use point-to-point values with their months \
("MAU 187 in Dec 2025 down to 93 in May 2026") or quote the sweep's averaged \
evidence verbatim. Never mix an averaged start with a point-value end.
- If a knowledge doc lookup misses and the tool lists the available sections, \
immediately read the most relevant section and answer. Never ask the user for \
permission to read something you can read.

## Style
- AEs skim. Short lines, concrete numbers, dates. No filler, no cheerleading.
- Renewal/discovery prep follows the playbook's pre-call checklist \
(sales_playbook: What the CRM data tells us). Prioritize what the AE likely \
does NOT already know: tickets, usage shifts, contacts going quiet, competitor \
signals. Do not summarize their own recent notes back to them.
"""


# ---------------------------------------------------------------------------
# Account-reference gate (see MULTI_ACCOUNT_DISAMBIGUATION_SPEC.md)
#
# Bug class this kills: in multi-turn conversations the model's memory of
# WHICH account a generic follow-up ("show me the ticket history") refers to
# is probabilistic and has been observed to fail (it queried a wrong-but-valid
# account id and narrated the empty result as "no tickets exist"). Following
# the codebase's governing principle, the fix is a deterministic pre-execution
# gate that runs whether or not the model cooperates. Three checks:
#   1. AMBIGUOUS: 2+ accounts discussed, user's message names none -> block,
#      force a clarifying question naming the candidates.
#   2. MISMATCH: user's message names account X, model calls account Y -> block.
#   3. UNREFERENCED: model calls an account never discussed and not named by
#      the user (the originally observed repro) -> block.
# Resolved accounts are reconstructed from code-tracked tool-call history,
# never from the model's prose - the model's memory is exactly what this
# mechanism does not trust.
# ---------------------------------------------------------------------------

ACCOUNT_SCOPED_TOOLS = {"run_risk_sweep", "get_opportunities", "get_contacts",
                        "get_activities", "get_usage", "get_tickets"}


def _account_name(account_id: str) -> str:
    acct = db.get_account(account_id)
    return acct["COMPANY_NAME"] if acct else account_id


def _mentions_account(message: str, account_id: str, name: str) -> bool:
    """Fuzzy check: does the message reference this account? Matches the id,
    the full name, or any distinctive name token ('Halcyon', 'hospitality'),
    ignoring short tokens and legal suffixes ('AS', 'GmbH') that would
    over-trigger on ordinary words."""
    msg = message.lower()
    if account_id.lower() in msg or name.lower() in msg:
        return True
    for token in re.split(r"[^\w]+", name.lower()):
        if len(token) >= 4 and token != "gmbh" and token in msg:
            return True
    return False


def _resolved_accounts(tool_calls: list[dict] | None) -> dict[str, str]:
    """account_id -> company name for every account actually queried so far
    in this conversation (account-scoped tool calls with a valid id). Built
    from the code-tracked call log; gated (blocked) calls don't count."""
    resolved: dict[str, str] = {}
    for c in tool_calls or []:
        if c.get("gated") or c["name"] not in ACCOUNT_SCOPED_TOOLS:
            continue
        aid = str((c.get("args") or {}).get("account_id") or "")
        if re.fullmatch(r"ACC-\d{4}", aid) and db.get_account(aid):
            resolved.setdefault(aid, _account_name(aid))
    return resolved


def _account_gate(tool_name: str, args: dict, resolved: dict[str, str],
                  latest_user_message: str) -> dict | None:
    """Returns an error dict to inject in place of the real tool result, or
    None if the call may proceed. Runs BEFORE tool execution."""
    if tool_name not in ACCOUNT_SCOPED_TOOLS:
        return None
    target = str((args or {}).get("account_id") or "")
    if not re.fullmatch(r"ACC-\d{4}", target) or not db.get_account(target):
        return None  # malformed/nonexistent ids are tools.py's job
    if _mentions_account(latest_user_message, target, _account_name(target)):
        return None  # user explicitly referenced the target account
    named = {aid: n for aid, n in resolved.items()
             if _mentions_account(latest_user_message, aid, n)}
    if named:
        names = ", ".join(f"{n} ({aid})" for aid, n in named.items())
        return {"error": f"ACCOUNT MISMATCH: the user's message refers to "
                         f"{names}, but you called {tool_name} on "
                         f"{_account_name(target)} ({target}). Use the account "
                         "the user named, or ask them to clarify."}
    if not resolved:
        return None  # nothing discussed yet: never ambiguous
    if target in resolved and len(resolved) == 1:
        return None  # single-account conversation: unambiguous continuation
    candidates = ", ".join(f"{n} ({aid})" for aid, n in resolved.items())
    if target in resolved:
        return {"error": "AMBIGUOUS ACCOUNT REFERENCE: this conversation has "
                         f"discussed multiple accounts ({candidates}) and the "
                         "user's latest message doesn't name one. Do not call "
                         "any account-scoped tool. Ask the user which account "
                         "they mean, naming all candidates explicitly."}
    return {"error": f"UNREFERENCED ACCOUNT: you called {tool_name} on "
                     f"{_account_name(target)} ({target}), which the user did "
                     "not name and which has not been discussed in this "
                     f"conversation (discussed: {candidates}). If the user "
                     "means a discussed account, use its id; otherwise ask "
                     "which account they mean."}


def _openai_tools():
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["parameters"]}}
            for t in tools.TOOLS]


def _anthropic_tools():
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["parameters"]}
            for t in tools.TOOLS]


def _with_retries(fn, attempts: int = 5):
    """Retry transient provider errors (rate limits, overload) with backoff.
    Real errors (auth, bad request) are raised immediately."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            transient = any(t in str(e).lower() for t in
                            ("429", "rate_limit", "rate limit", "overloaded",
                             "529", "timeout", "connection"))
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(min(3 * 2 ** attempt, 30))


def _truncate(s: str) -> str:
    if len(s) <= MAX_RESULT_CHARS:
        return s
    return s[:MAX_RESULT_CHARS] + f"\n...[truncated, {len(s)} chars total]"


def _log_trace(entry: dict):
    """One CSV row per turn (CSV so traces open directly in Excel for
    failure-analysis sessions). tool_calls is a JSON string within its cell."""
    import csv

    TRACES_DIR.mkdir(exist_ok=True)
    path = TRACES_DIR / f"{datetime.now():%Y-%m-%d}.csv"
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "ae", "provider", "model", "question",
                        "tool_calls", "answer", "latency_s",
                        "input_tokens", "output_tokens", "cost_usd"])
        w.writerow([entry["ts"], entry["ae"], entry["provider"],
                    entry["model"], entry["question"],
                    json.dumps(entry["tool_calls"], default=str),
                    entry["answer"], entry["latency_s"],
                    entry["input_tokens"], entry["output_tokens"],
                    entry["cost_usd"]])


def run_turn(history: list[dict], ae_email: str,
             prior_tool_calls: list[dict] | None = None) -> dict:
    """One user turn. history = [{'role': 'user'|'assistant', 'content': str}, ...]
    prior_tool_calls: flattened tool-call dicts from every earlier turn of this
    conversation (callers accumulate result['tool_calls']); powers the
    account-reference gate. Optional and backward compatible: without it the
    gate still protects within the current turn, just not across turns.
    Returns {'answer': str, 'tool_calls': [{'name','args',...}], 'provider': str}."""
    start = time.time()
    _dbg(f"\n{_BAR}\n"
         f"NEW TURN | {AE_NAMES.get(ae_email, ae_email)} asks: "
         f"\"{history[-1]['content'][:80]}\"\n{_BAR}")
    if PROVIDER == "anthropic":
        result = _run_anthropic(history, ae_email, prior_tool_calls or [])
    elif PROVIDER == "openai":
        result = _run_openai(history, ae_email, prior_tool_calls or [])
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{PROVIDER}'")
    result["provider"] = PROVIDER
    model = OPENAI_MODEL if PROVIDER == "openai" else ANTHROPIC_MODEL
    in_tok = result.get("input_tokens", 0)
    out_tok = result.get("output_tokens", 0)
    cost = _cost_usd(model, in_tok, out_tok)
    _dbg(f"\nANSWER | {len(result['answer'])} chars, "
         f"{len(result['tool_calls'])} tool call(s), "
         f"{round(time.time() - start, 1)}s, "
         f"{in_tok}+{out_tok} tok"
         + (f", ${cost:.4f}" if cost is not None else "") + f"\n{_BAR}")
    _log_trace({
        "ts": datetime.now().isoformat(),
        "ae": ae_email,
        "provider": PROVIDER,
        "model": model,
        "question": history[-1]["content"],
        "tool_calls": result["tool_calls"],
        "answer": result["answer"],
        "latency_s": round(time.time() - start, 1),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": cost,
    })
    return result


def _run_openai(history, ae_email, prior_tool_calls):
    """OpenAI wire format for the loop described in the module docstring.
    `messages` grows every iteration (assistant tool-call request, then our
    tool-result reply) so the model sees its own prior results before
    deciding whether it needs more data or is ready to answer."""
    from openai import OpenAI

    client = OpenAI()
    ctx = {"ae_email": ae_email}
    messages = [{"role": "system", "content": system_prompt(ae_email)}] + list(history)
    latest_user_message = history[-1]["content"]
    calls = []
    in_tok = out_tok = 0
    for iteration in range(MAX_ITERATIONS):
        _dbg(f"\n[THINK] round-trip {iteration + 1}: agent.py sends "
             f"{len(messages)} messages to {OPENAI_MODEL}")
        resp = _with_retries(lambda: client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, tools=_openai_tools(),
            temperature=0.2))  # low temperature: consistency beats flair here
        if resp.usage:
            in_tok += resp.usage.prompt_tokens
            out_tok += resp.usage.completion_tokens
        msg = resp.choices[0].message
        # No tool_calls = the model considers itself done; return now. A
        # non-empty tool_calls list (possibly more than one) means another
        # loop iteration: execute each, feed results back, ask again.
        if not msg.tool_calls:
            _dbg("        model is ready to answer, no more tools needed")
            return {"answer": msg.content or "", "tool_calls": calls,
                    "input_tokens": in_tok, "output_tokens": out_tok}
        _dbg("        model wants: "
             + ", ".join(tc.function.name for tc in msg.tool_calls))
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            gate = _account_gate(tc.function.name, args,
                                 _resolved_accounts(prior_tool_calls + calls),
                                 latest_user_message)
            if gate:
                _dbg(f"[GATE]  blocked {tc.function.name}({json.dumps(args)[:60]}): "
                     + gate["error"].split(":")[0])
                result = gate
                call_entry = {"name": tc.function.name, "args": args,
                              "gated": True,
                              "method": {"note": gate["error"]}}
            else:
                _dbg(f"[ACT]   tools.py runs {tc.function.name}"
                     f"({json.dumps(args)[:90]})")
                result = tools.execute_tool(tc.function.name, args, ctx)
                call_entry = {"name": tc.function.name, "args": args}
                if isinstance(result, dict) and "method" in result:
                    call_entry["method"] = result["method"]  # provenance for the UI
            payload = _truncate(json.dumps(result, default=str))
            _dbg(f"        returned {len(payload)} chars"
                 + (" (ERROR)" if isinstance(result, dict) and "error" in result else ""))
            calls.append(call_entry)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": payload})
    return {"answer": "I hit my tool-call limit before finishing. "
                      "Try a narrower question.", "tool_calls": calls,
            "input_tokens": in_tok, "output_tokens": out_tok}


def _run_anthropic(history, ae_email, prior_tool_calls):
    import anthropic

    client = anthropic.Anthropic()
    ctx = {"ae_email": ae_email}
    messages = list(history)
    latest_user_message = history[-1]["content"]
    calls = []
    in_tok = out_tok = 0
    for iteration in range(MAX_ITERATIONS):
        _dbg(f"\n[THINK] round-trip {iteration + 1}: agent.py sends "
             f"{len(messages)} messages to {ANTHROPIC_MODEL}")
        resp = _with_retries(lambda: client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=2000,
            system=system_prompt(ae_email),
            tools=_anthropic_tools(), messages=messages))
        if resp.usage:
            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens
        if resp.stop_reason != "tool_use":
            _dbg("        model is ready to answer, no more tools needed")
            text = "".join(b.text for b in resp.content if b.type == "text")
            return {"answer": text, "tool_calls": calls,
                    "input_tokens": in_tok, "output_tokens": out_tok}
        _dbg("        model wants: " + ", ".join(
            b.name for b in resp.content if b.type == "tool_use"))
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                args = dict(block.input)
                gate = _account_gate(block.name, args,
                                     _resolved_accounts(prior_tool_calls + calls),
                                     latest_user_message)
                if gate:
                    _dbg(f"[GATE]  blocked {block.name}"
                         f"({json.dumps(args)[:60]}): "
                         + gate["error"].split(":")[0])
                    result = gate
                    call_entry = {"name": block.name, "args": args,
                                  "gated": True,
                                  "method": {"note": gate["error"]}}
                else:
                    _dbg(f"[ACT]   tools.py runs {block.name}"
                         f"({json.dumps(args)[:90]})")
                    result = tools.execute_tool(block.name, args, ctx)
                    call_entry = {"name": block.name, "args": args}
                    if isinstance(result, dict) and "method" in result:
                        call_entry["method"] = result["method"]
                calls.append(call_entry)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": _truncate(json.dumps(result, default=str))})
        messages.append({"role": "user", "content": results})
    return {"answer": "I hit my tool-call limit before finishing. "
                      "Try a narrower question.", "tool_calls": calls,
            "input_tokens": in_tok, "output_tokens": out_tok}
