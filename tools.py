"""The agent's hands: every function the model is allowed to call.

Each tool is defined once (name, description, JSON-schema parameters,
python function) and converted to OpenAI or Anthropic format by agent.py.
Tools are thin wrappers over db.py and signals.py; none of them contain
model-generated SQL. The model composes these tools, it never bypasses them.
"""

from __future__ import annotations

import json
import re
from datetime import date

import db
import knowledge
import signals


def _bad_account_id(account_id) -> dict | None:
    """Reject malformed or nonexistent account ids with an instructive error.
    Without this, a hallucinated id ('001') or a company name passed as an id
    returns an empty result set, which the model then narrates as 'no data
    exists': a confident falsehood. Bad input must be an error, never an
    empty success."""
    if not re.fullmatch(r"ACC-\d{4}", str(account_id or "")):
        return {"error": f"'{account_id}' is not a valid ACCOUNT_ID "
                         "(format: ACC-0002). Call find_account with the "
                         "company name first and use the ACCOUNT_ID it returns."}
    if not db.get_account(account_id):
        return {"error": f"No account exists with id {account_id}. Call "
                         "find_account to resolve the correct id."}
    return None


def _owner_note(owner: str, ctx: dict) -> str | None:
    if owner and owner != ctx["ae_email"]:
        return f"NOTE: this account is owned by {owner}, not the current user."
    return None


def _jsonable(obj):
    """Dates and decimals from databases aren't JSON-native; stringify them."""
    return json.loads(json.dumps(obj, default=str))


def _enriched_contacts(account_id: str) -> list[dict]:
    """Contacts with engagement computed from the activity log, because the
    CRM's LAST_INTERACTION field goes stale in this dataset."""
    contacts = db.contacts_for(account_id)
    activity_dates: dict = {}
    for a in db.activities_for(account_id, limit=500):
        cid = a["CONTACT_ID"]
        if cid and (cid not in activity_dates
                    or str(a["ACTIVITY_DATE"]) > str(activity_dates[cid])):
            activity_dates[cid] = a["ACTIVITY_DATE"]
    for c in contacts:
        days = signals._last_touch(c, activity_dates)
        c["DAYS_SINCE_LAST_TOUCH"] = days
        c["ENGAGED_LAST_60D"] = days is not None and days <= signals.RECENT_DAYS
    return contacts


# ---------------------------------------------------------------------------
# Tool registry: (json schema, function) pairs
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "find_account",
        "description": (
            "Resolve a company name (full or partial) or account ID to "
            "account records. Always call this first when the user mentions "
            "an account, to get its ACCOUNT_ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "Company name fragment or account ID"}},
            "required": ["query"],
        },
    },
    {
        "name": "list_my_accounts",
        "description": (
            "BROWSE the individual accounts in the current AE's book (id, "
            "name, status, segment, region, ARR). For 'how many' questions "
            "use get_stats; if you used this tool anyway, report its 'count' "
            "field, never count the rows yourself."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "list_my_open_deals",
        "description": (
            "The individual open opportunities in the current AE's book "
            "(account, stage, amount, close date, type). Filters run in SQL: "
            "use opp_type and closing_within_days instead of filtering rows "
            "yourself ('my renewals closing in 60 days' = opp_type=Renewal, "
            "closing_within_days=60; doing date math on rows drops edge "
            "cases). For totals use get_stats; the payload's 'count' and "
            "'total_eur' are authoritative."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "opp_type": {"type": "string",
                             "enum": ["Renewal", "Expansion", "New Business"],
                             "description": "Optional deal-type filter"},
                "closing_within_days": {"type": "integer",
                                        "description": "Optional: only deals "
                                        "closing within N days of today"},
            },
        },
    },
    {
        "name": "get_book_priorities",
        "description": (
            "TOP-5 risk-ranked priority list for the current AE's book (same "
            "framework as the morning digest). TRUNCATED VIEW: use only for "
            "'what should I focus on today' prioritization. NEVER use it to "
            "answer 'which accounts have X' (use scan_book_signals for that: "
            "it is complete)."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "scan_book_signals",
        "description": (
            "COMPLETE scan of every account in the current AE's book against "
            "the signal rules, no truncation. THE tool for set-membership "
            "questions: 'which customers are declining in usage', 'which "
            "accounts have no Economic Buyer', 'where is churn risk'. "
            "Optionally filter to one signal type: " +
            ", ".join(sorted(
                ["usage_drop", "single_threaded", "no_economic_buyer",
                 "eb_gone_quiet", "no_technical_contact", "competitor_mention",
                 "renewal_window", "overdue_close_date", "stalled_deal",
                 "quiet_account", "open_ticket", "recent_p1",
                 "usage_data_quality", "duplicate_open_opps"]))
        ),
        "parameters": {
            "type": "object",
            "properties": {"signal": {
                "type": "string",
                "description": "Optional: restrict to one signal type"}},
        },
    },
    {
        "name": "get_stats",
        "description": (
            "Deterministic counts and sums, computed in SQL: total accounts, "
            "accounts by status/segment, accounts per AE (answers 'who has "
            "the most accounts'), open pipeline value and deal counts by "
            "type, and closed-lost/closed-won reason counts (answers 'why do "
            "we lose deals' empirically). THE tool for every 'how many', "
            "'total worth', 'who has the most', 'why do we win/lose' "
            "question. Never count or sum rows yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {"scope": {
                "type": "string", "enum": ["mine", "company"],
                "description": "mine = current AE's book; company = all accounts"}},
            "required": ["scope"],
        },
    },
    {
        "name": "run_risk_sweep",
        "description": (
            "Run the deterministic signal engine on one account: usage trend, "
            "renewal windows, overdue close dates, stalled deals, "
            "single-threading, missing buyer personas, tickets, competitor "
            "mentions, quiet periods. Returns signals with severity, evidence "
            "and the playbook rule behind each. ALWAYS run this the first "
            "time an account comes up in a conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
    {
        "name": "get_opportunities",
        "description": "All opportunities (deals) for an account: stage, amount, close date, type, days in stage.",
        "parameters": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
    {
        "name": "get_contacts",
        "description": (
            "Contacts for an account with persona type (Economic Buyer, "
            "Champion, Technical, Influencer, User) and engagement computed "
            "from the activity log (trust DAYS_SINCE_LAST_TOUCH, not the raw "
            "LAST_INTERACTION field, which goes stale)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
    {
        "name": "get_activities",
        "description": (
            "Interaction log for an account (emails, calls, meetings, notes), "
            "newest first. Filter with 'since' (YYYY-MM-DD) or 'contains' "
            "(keyword in subject/summary)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "since": {"type": "string", "description": "YYYY-MM-DD"},
                "contains": {"type": "string"},
                "limit": {"type": "integer", "default": 25},
            },
            "required": ["account_id"],
        },
    },
    {
        "name": "get_usage",
        "description": (
            "Monthly product usage for a customer account: MAU, logins, "
            "payroll runs, module adoption. Negative values are data "
            "corruption; ignore them in trends and say so."
        ),
        "parameters": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
    {
        "name": "get_tickets",
        "description": "Support tickets for an account with priority (P1 worst) and status.",
        "parameters": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
    {
        "name": "list_knowledge_docs",
        "description": (
            "Table of contents of the sales enablement library (playbook, "
            "ICP, battlecards, objection handling, pricing guardrails, case "
            "studies) with section headings."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "read_knowledge_doc",
        "description": (
            "Read one enablement doc, or a single section of it. Prefer "
            "reading just the relevant section. Cite doc and section when "
            "using this content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string",
                         "description": "One of: " + ", ".join(knowledge.DOCS)},
                "section": {"type": "string",
                            "description": "Optional section heading (fuzzy matched)"},
            },
            "required": ["name"],
        },
    },
]


def execute_tool(name: str, args: dict, ctx: dict) -> dict | list:
    """Run one tool call. ctx carries the AE identity set at login."""
    try:
        if name == "find_account":
            rows = db.find_accounts(args["query"])
            if not rows:
                return {"result": f"No account matches '{args['query']}'"}
            for r in rows:
                note = _owner_note(r.get("OWNER_AE"), ctx)
                if note:
                    r["OWNERSHIP_NOTE"] = note
            return _jsonable(rows)
        if name == "list_my_accounts":
            rows = db.accounts_for_ae(ctx["ae_email"])
            return _jsonable({
                "method": {"scope": f"all accounts owned by {ctx['ae_email']}",
                           "note": "count is authoritative (computed in code); "
                                   "includes churned accounts"},
                "count": len(rows),
                "accounts": rows,
            })
        if name == "list_my_open_deals":
            opp_type = args.get("opp_type")
            within = args.get("closing_within_days")
            rows = db.open_opps_for_ae(ctx["ae_email"], opp_type=opp_type,
                                       closing_within_days=within)
            scope = f"open deals owned by {ctx['ae_email']}"
            if opp_type:
                scope += f", type={opp_type}"
            if within is not None:
                scope += (f", closing on or before {db.closing_cutoff(within)} "
                          f"({within} days from {db.AS_OF}), filtered in SQL")
            return _jsonable({
                "method": {"scope": scope,
                           "definitions": ["open deal = stage in Discovery, "
                                           "Qualification, Demo, Proposal or "
                                           "Negotiation"],
                           "note": "count and total_eur are authoritative "
                                   "(computed in code); this list is complete "
                                   "for the stated filters"},
                "count": len(rows),
                "total_eur": sum(r["AMOUNT_EUR"] or 0 for r in rows),
                "deals": rows,
            })
        if name == "get_book_priorities":
            ranked = signals.rank_book(ctx["ae_email"])
            n_book = len(db.accounts_for_ae(ctx["ae_email"]))
            return _jsonable({
                "method": {
                    "scope": f"top {len(ranked)} of {n_book} accounts in your "
                             "book, ranked by high-severity signals, then "
                             "medium, then nearest close date",
                    "truncated": True,
                    "note": "This is a prioritization view, NOT a complete "
                            "list of accounts with any given problem.",
                },
                "priorities": [{
                    "rank": i + 1,
                    "account": r["sweep"]["account"],
                    "high_signals": [s["headline"] for s in r["sweep"]["signals"]
                                     if s["severity"] == "high"],
                    "medium_signals": [s["headline"] for s in r["sweep"]["signals"]
                                       if s["severity"] == "medium"],
                    "next_close_date": r["next_close_date"],
                } for i, r in enumerate(ranked)],
            })
        if name == "scan_book_signals":
            return _jsonable(signals.scan_book(ctx["ae_email"],
                                               args.get("signal")))
        if name == "get_stats":
            return _jsonable(db.stats(args["scope"], ctx["ae_email"]))
        if name in ("run_risk_sweep", "get_opportunities", "get_contacts",
                    "get_activities", "get_usage", "get_tickets"):
            bad = _bad_account_id(args.get("account_id"))
            if bad:
                return bad
        aid = args.get("account_id")
        if name == "run_risk_sweep":
            sweep = signals.run_sweep(aid)
            sweep["method"] = {
                "scope": f"single account {aid}, all rules evaluated",
                "definitions": ["signals are deterministic coded rules; each "
                                "carries its evidence and rule_source"],
                "note": _owner_note(sweep["account"]["owner"], ctx),
            }
            return _jsonable(sweep)
        if name == "get_opportunities":
            rows = db.opportunities_for(aid)
            return _jsonable({"method": {"scope": f"all opportunities on {aid}"},
                              "count": len(rows), "rows": rows})
        if name == "get_contacts":
            rows = _enriched_contacts(aid)
            return _jsonable({
                "method": {"scope": f"all contacts on {aid}",
                           "definitions": ["DAYS_SINCE_LAST_TOUCH is computed "
                                           "from the activity log; the raw "
                                           "LAST_INTERACTION field goes stale"]},
                "count": len(rows), "rows": rows})
        if name == "get_activities":
            rows = db.activities_for(aid, since=args.get("since"),
                                     contains=args.get("contains"),
                                     limit=args.get("limit", 25))
            return _jsonable({
                "method": {"scope": f"activities on {aid}"
                           + (f" since {args['since']}" if args.get("since") else "")
                           + (f" containing '{args['contains']}'" if args.get("contains") else "")},
                "count": len(rows), "rows": rows})
        if name == "get_usage":
            rows = db.usage_for(aid)
            return _jsonable({
                "method": {"scope": f"monthly usage rows for {aid}",
                           "definitions": ["quote trends point-to-point with "
                                           "month names; negative values are "
                                           "corrupt and must be excluded and "
                                           "flagged"]},
                "count": len(rows), "rows": rows})
        if name == "get_tickets":
            rows = db.tickets_for(aid)
            return _jsonable({"method": {"scope": f"all support tickets for {aid}"},
                              "count": len(rows), "rows": rows})
        if name == "list_knowledge_docs":
            return knowledge.list_docs()
        if name == "read_knowledge_doc":
            return knowledge.read_doc(args["name"], args.get("section"))
        return {"error": f"Unknown tool {name}"}
    except Exception as e:  # surface errors to the model so it can recover
        return {"error": f"{type(e).__name__}: {e}"}
