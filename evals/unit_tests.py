"""Deterministic unit tests for the tool layer. No LLM, no API key, runs in
seconds. These test the guarantees the model relies on: bad input must error
(never return an empty success), computed numbers must be exact, complete
scans must be complete.

Usage:  python evals/unit_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tools  # noqa: E402

CTX = {"ae_email": "lena.koehler@personio.de"}
failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  {status}  {name}" + (f"  ({detail})" if detail and not condition else ""))
    if not condition:
        failures.append(name)


# --- bad input must be an error, never an empty success --------------------
r = tools.execute_tool("get_tickets", {"account_id": "Halcyon Hospitality Group"}, CTX)
check("name-as-id returns error", isinstance(r, dict) and "error" in r, str(r)[:80])

r = tools.execute_tool("get_contacts", {"account_id": "001"}, CTX)
check("hallucinated id returns error", isinstance(r, dict) and "error" in r, str(r)[:80])

r = tools.execute_tool("run_risk_sweep", {"account_id": "ACC-9999"}, CTX)
check("nonexistent id returns error", isinstance(r, dict) and "error" in r, str(r)[:80])

# --- valid input still works ------------------------------------------------
r = tools.execute_tool("get_tickets", {"account_id": "ACC-0002"}, CTX)
check("Halcyon has 6 tickets", r.get("count") == 6, f"count={r.get('count')}")

r = tools.execute_tool("get_contacts", {"account_id": "ACC-0002"}, CTX)
check("Halcyon contacts include engagement fields",
      r.get("count", 0) > 0 and "DAYS_SINCE_LAST_TOUCH" in r["rows"][0])

# --- aggregates are exact ----------------------------------------------------
s = tools.execute_tool("get_stats", {"scope": "mine"}, CTX)
check("book total accounts = 41", s["total_accounts"] == 41)
check("pipeline total = 3,715,493", s["open_pipeline"]["total_eur"] == 3715493.0)
check("avg deal = 86,406.81", s["open_pipeline"]["avg_deal_eur"] == 86406.81,
      str(s["open_pipeline"]["avg_deal_eur"]))

s = tools.execute_tool("get_stats", {"scope": "company"}, CTX)
check("company total accounts = 75", s["total_accounts"] == 75)
check("top owner is Lena/41",
      s["accounts_by_owner"][0]["OWNER_AE"] == "lena.koehler@personio.de"
      and s["accounts_by_owner"][0]["N"] == 41)
check("company loss reasons: incumbent=4",
      any(r["WON_LOST_REASON"] == "Lost to incumbent" and r["N"] == 4
          for r in s["closed_lost_reasons"]))

# --- SQL-side filters, not model-side ---------------------------------------
r = tools.execute_tool("list_my_open_deals",
                       {"opp_type": "Renewal", "closing_within_days": 60}, CTX)
check("renewals within 60 days = 11", r["count"] == 11, f"count={r['count']}")
check("window stated in method", "2026-07-31" in r["method"]["scope"])

# --- complete scans stay complete and untruncated ----------------------------
r = tools.execute_tool("scan_book_signals", {"signal": "usage_drop"}, CTX)
names = {x["name"] for x in r["results"]}
check("usage decline = exactly 3 accounts",
      names == {"Cobalt Manufacturing B.V.", "Gale Studios ApS",
                "Halcyon Hospitality Group"}, str(names))

r = tools.execute_tool("scan_book_signals", {"signal": "no_economic_buyer"}, CTX)
check("no-EB scan finds 11 accounts", r["method"]["accounts_flagged"] == 11,
      str(r["method"]["accounts_flagged"]))

import json  # noqa: E402
r = tools.execute_tool("scan_book_signals", {}, CTX)
check("full scan fits untruncated", len(json.dumps(r, default=str)) < 12000)

# --- ownership notes ---------------------------------------------------------
r = tools.execute_tool("find_account", {"query": "Vector Construction"}, CTX)
check("ownership note on other AE's account",
      "OWNERSHIP_NOTE" in r[0] and "thomas" in r[0]["OWNERSHIP_NOTE"])

# --- account-reference gate (MULTI_ACCOUNT_DISAMBIGUATION_SPEC) --------------
import agent  # noqa: E402

TWO = {"ACC-0002": "Halcyon Hospitality Group", "ACC-0003": "Fjord Logistics AS"}
ONE = {"ACC-0002": "Halcyon Hospitality Group"}

g = agent._account_gate("get_tickets", {"account_id": "ACC-0002"}, TWO,
                        "show me the ticket history")
check("gate: ambiguous with 2 accounts -> blocked",
      g is not None and "AMBIGUOUS" in g["error"], str(g)[:80])

g = agent._account_gate("get_tickets", {"account_id": "ACC-0002"}, TWO,
                        "show me Halcyon's tickets")
check("gate: named target passes through", g is None, str(g)[:80])

g = agent._account_gate("get_tickets", {"account_id": "ACC-0002"}, TWO,
                        "what about the hospitality one?")
check("gate: fuzzy name token passes through", g is None, str(g)[:80])

g = agent._account_gate("get_tickets", {"account_id": "ACC-0003"}, TWO,
                        "show me Halcyon's tickets")
check("gate: mismatch (named X, called Y) -> blocked",
      g is not None and "MISMATCH" in g["error"], str(g)[:80])

g = agent._account_gate("get_tickets", {"account_id": "ACC-0002"}, ONE,
                        "show me the ticket history")
check("gate: single-account continuation passes", g is None, str(g)[:80])

# the originally observed repro: 1 account discussed, generic follow-up,
# model queries a different valid-but-never-discussed account
g = agent._account_gate("get_tickets", {"account_id": "ACC-0004"}, ONE,
                        "show me the ticket history")
check("gate: undiscussed unnamed target -> blocked (original repro)",
      g is not None and "UNREFERENCED" in g["error"], str(g)[:80])

g = agent._account_gate("get_tickets", {"account_id": "ACC-0002"}, {},
                        "show me the ticket history")
check("gate: empty conversation never gated", g is None, str(g)[:80])

g = agent._account_gate("get_stats", {"scope": "mine"}, TWO,
                        "show me the numbers")
check("gate: non-account-scoped tools untouched", g is None, str(g)[:80])

resolved = agent._resolved_accounts([
    {"name": "find_account", "args": {"query": "Halcyon"}},
    {"name": "run_risk_sweep", "args": {"account_id": "ACC-0002"}},
    {"name": "get_tickets", "args": {"account_id": "ACC-0004"}, "gated": True},
    {"name": "get_usage", "args": {"account_id": "not-an-id"}},
])
check("resolved-accounts: scoped calls only, gated and invalid excluded",
      resolved == {"ACC-0002": "Halcyon Hospitality Group"}, str(resolved))

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("all unit tests passed")
