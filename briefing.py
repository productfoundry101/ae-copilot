"""Login briefing and tailored suggestions, fully deterministic.

This is the signal engine's third delivery surface (after the chat sweep and
the cron digest): a welcome overview rendered from code on login, zero LLM
calls, so it is fast, free, and identical on every load. Content discipline:
three numbers, three names, nothing else. The overview answers "where does my
attention go?" and hands everything deeper to the chat.
"""

from __future__ import annotations

import db
import signals


def _short_reason(sweep: dict) -> str:
    """One machine-authored line per attention account: top two signals."""
    highs = [s for s in sweep["signals"] if s["severity"] == "high"]
    meds = [s for s in sweep["signals"] if s["severity"] == "medium"]
    picks = (highs + meds)[:2]
    return "; ".join(p["headline"] for p in picks)


def book_overview(ae_email: str) -> dict:
    stats = db.stats("mine", ae_email)
    by_status = {r["STATUS"]: r["N"] for r in stats["accounts_by_status"]}
    renewals_60d = db.open_opps_for_ae(ae_email, opp_type="Renewal",
                                       closing_within_days=60)
    ranked = signals.rank_book(ae_email, max_accounts=3)
    return {
        "total_accounts": stats["total_accounts"],
        "by_status": by_status,
        "pipeline_eur": stats["open_pipeline"]["total_eur"],
        "pipeline_deals": stats["open_pipeline"]["total_deals"],
        "renewals_60d": len(renewals_60d),
        "attention": [{
            "name": r["sweep"]["account"]["name"],
            "id": r["sweep"]["account"]["id"],
            "reason": _short_reason(r["sweep"]),
        } for r in ranked],
    }


def suggested_questions(overview: dict) -> list[str]:
    """Four clickable starters, tailored to the signed-in AE's actual book.
    Account names come from the deterministic ranking, never from a guess."""
    qs = ["Which of my accounts are at highest risk of churn?"]
    if overview["attention"]:
        qs.append("How should I prepare for my next call with "
                  f"{overview['attention'][0]['name']}?")
    qs.append("What should be my top priorities today?")
    if overview["renewals_60d"]:
        qs.append("Which of my renewals close in the next 60 days?")
    else:
        qs.append("What's my open pipeline worth right now?")
    return qs


def eur(v: float) -> str:
    return f"€{v/1_000_000:.2f}M" if v >= 1_000_000 else f"€{v:,.0f}"
