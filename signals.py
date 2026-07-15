"""Signal engine: deterministic risk and opportunity detection.

This is the "tell me what I'd miss" capability. Every rule here is lifted
from the sales playbook or the AE discovery interviews, cited in comments.
Rules are plain Python over CRM data, not LLM judgment, on purpose:
a signal either fires with evidence or it doesn't, so the same account
always produces the same sweep. The LLM's job is to explain and
prioritise signals, never to invent them.

Used by two consumers:
  1. The chat agent runs run_sweep() proactively whenever an account
     comes up in conversation.
  2. The morning digest (digest.py) runs it across an AE's whole book
     on a schedule.
"""

from __future__ import annotations

from datetime import date, timedelta

import db
import knowledge

# --- Thresholds, each traceable to a source document -----------------------

USAGE_DROP_WARN = -20      # % MAU change that counts as a decline
USAGE_DROP_SEVERE = -40    # severe decline
STALL_DAYS = 30            # playbook: momentum signal is "days in stage"
QUIET_DAYS = 21            # no touchpoints on an account with open deals
RENEWAL_WINDOW_MM = 90     # playbook 6: start renewal motion 90 days out (MM)
RENEWAL_WINDOW_ENT = 120   # playbook 6: 120 days out for ENT
RECENT_DAYS = 60           # playbook 8: "recent activities (last 60 days)"
P1_LOOKBACK = 90           # a P1 incident colours the relationship well past 60 days
MULTITHREAD_MIN = 3        # playbook 5: "3+ stakeholders by end of demo"

COMPETITORS = ["hibob", "workday"]

OPEN_STAGES = {"Discovery", "Qualification", "Demo", "Proposal", "Negotiation"}


def _as_of() -> date:
    return date.fromisoformat(db.AS_OF)


def _days(d, ref=None):
    """Days between a date-ish value and the as-of date (positive = past)."""
    if d is None:
        return None
    if isinstance(d, str):
        d = date.fromisoformat(d[:10])
    if hasattr(d, "date") and not isinstance(d, date):
        d = d.date()
    return ((ref or _as_of()) - d).days


def _sig(kind, severity, headline, evidence, source):
    """One signal: what fired, how bad, the proof, and where the rule comes from."""
    return {
        "signal": kind,
        "severity": severity,          # high | medium | info
        "headline": headline,
        "evidence": evidence,          # concrete facts with dates and numbers
        "rule_source": source,         # which doc justifies this rule
    }


# --- Individual checks ------------------------------------------------------

def check_usage_trend(account_id):
    """Playbook 8: 'product usage trend' is required pre-call reading.
    Interview (Lena): 'the thing the support tickets are saying that the
    activity log isn't' - usage decline is the quiet churn signal."""
    rows = db.usage_for(account_id)
    clean = [r for r in rows
             if (r["MONTHLY_ACTIVE_USERS"] or 0) >= 0 and (r["LOGINS"] or 0) >= 0]
    bad = [r for r in rows if r not in clean]
    out = []
    if bad:
        months = ", ".join(str(r["MONTH"])[:7] for r in bad)
        out.append(_sig(
            "usage_data_quality", "info",
            "Some usage rows look corrupted and were excluded from trend math",
            f"Negative MAU/login values in months: {months}",
            "data quality check (negative counts are impossible)",
        ))
    if len(clean) >= 4:
        early = sum(r["MONTHLY_ACTIVE_USERS"] for r in clean[:2]) / 2
        late = sum(r["MONTHLY_ACTIVE_USERS"] for r in clean[-2:]) / 2
        pct = round((late - early) / max(early, 1) * 100)
        if pct <= USAGE_DROP_WARN:
            sev = "high" if pct <= USAGE_DROP_SEVERE else "medium"
            out.append(_sig(
                "usage_drop", sev,
                f"Product usage is down {abs(pct)}% over the tracked period",
                f"Avg MAU {early:.0f} in first two months vs {late:.0f} in last two "
                f"({clean[0]['MONTH']} to {clean[-1]['MONTH']})",
                "playbook s8: check product usage trend before any customer call",
            ))
    return out


def check_renewals(account_id, segment):
    """Playbook 6: renewal motion starts 90 days out (MM) / 120 (ENT).
    Late starts are how we get squeezed on price."""
    out = []
    window = RENEWAL_WINDOW_ENT if segment == "ENT" else RENEWAL_WINDOW_MM
    for o in db.opportunities_for(account_id):
        if o["STAGE"] not in OPEN_STAGES:
            continue
        days_past = _days(o["CLOSE_DATE"])
        if days_past is None:
            continue
        if o["TYPE"] == "Renewal" and -window <= -days_past:
            pass  # handled below via generic checks
        # if/elif: an already-overdue close date is the more urgent problem,
        # so it wins over "renewal window approaching" for the same deal.
        # DAYS_IN_STAGE below is a separate, non-exclusive check: a deal can
        # be both overdue/in-window AND stalled at the same time.
        if days_past > 0:
            out.append(_sig(
                "overdue_close_date", "high" if o["TYPE"] == "Renewal" else "medium",
                f"{o['TYPE']} '{o['NAME']}' close date passed {days_past} days ago, "
                f"still in {o['STAGE']}",
                f"Close date {o['CLOSE_DATE']}, amount EUR {o['AMOUNT_EUR']:,.0f}",
                "playbook s7: 'optimism in CRM' failure mode; forecast hygiene",
            ))
        elif o["TYPE"] == "Renewal" and days_past >= -window:
            out.append(_sig(
                "renewal_window", "medium",
                f"Renewal '{o['NAME']}' closes in {-days_past} days and is in "
                f"{o['STAGE']}",
                f"Close date {o['CLOSE_DATE']}, amount EUR {o['AMOUNT_EUR']:,.0f}. "
                f"Playbook says the renewal motion should be well underway.",
                f"playbook s6: start renewal motion {window} days out",
            ))
        if (o["DAYS_IN_STAGE"] or 0) > STALL_DAYS:
            out.append(_sig(
                "stalled_deal", "medium",
                f"'{o['NAME']}' has sat in {o['STAGE']} for {o['DAYS_IN_STAGE']} days",
                f"Amount EUR {o['AMOUNT_EUR']:,.0f}, close date {o['CLOSE_DATE']}",
                "playbook s3: days-in-stage is the momentum signal",
            ))
    # duplicate open opps of the same type = CRM hygiene issue
    open_opps = [o for o in db.opportunities_for(account_id) if o["STAGE"] in OPEN_STAGES]
    by_type = {}
    for o in open_opps:
        by_type.setdefault(o["TYPE"], []).append(o)
    for typ, opps in by_type.items():
        if len(opps) > 1:
            names = "; ".join(f"{o['NAME']} ({o['STAGE']})" for o in opps)
            out.append(_sig(
                "duplicate_open_opps", "info",
                f"{len(opps)} open {typ} opportunities on this account, possibly duplicates",
                names,
                "CRM hygiene check",
            ))
    return out


def _last_touch(contact, activity_dates):
    """A contact's true last touch: the freshest of the CRM's denormalized
    LAST_INTERACTION field and what the activity log actually shows.
    The two disagree in this dataset (the field goes stale), and the
    activity log is the source of truth."""
    candidates = []
    if contact["LAST_INTERACTION"]:
        candidates.append(_days(contact["LAST_INTERACTION"]))
    if contact["CONTACT_ID"] in activity_dates:
        candidates.append(_days(activity_dates[contact["CONTACT_ID"]]))
    candidates = [c for c in candidates if c is not None]
    # _days() convention: smaller number = more recent (fewer days have
    # passed). min() here means "trust whichever source says most recent."
    return min(candidates) if candidates else None


def check_engagement(account_id):
    """Playbook 5: multi-threading. 3+ engaged stakeholders; missing
    persona coverage is the signal Ines wished she'd had ('you have no
    IT/engineering contact at this account')."""
    out = []
    contacts = db.contacts_for(account_id)
    open_opps = [o for o in db.opportunities_for(account_id) if o["STAGE"] in OPEN_STAGES]
    if not open_opps:
        return out
    # Most recent activity date per contact, from the event log.
    activity_dates: dict = {}
    for a in db.activities_for(account_id, limit=500):
        cid = a["CONTACT_ID"]
        if cid and (cid not in activity_dates
                    or str(a["ACTIVITY_DATE"]) > str(activity_dates[cid])):
            activity_dates[cid] = a["ACTIVITY_DATE"]
    for c in contacts:
        c["_days_since_touch"] = _last_touch(c, activity_dates)
    engaged = [c for c in contacts if c["_days_since_touch"] is not None
               and c["_days_since_touch"] <= RECENT_DAYS]
    if len(engaged) < MULTITHREAD_MIN:
        names = ", ".join(f"{c['FULL_NAME']} ({c['PERSONA_TYPE']})" for c in engaged) or "nobody"
        out.append(_sig(
            "single_threaded", "high" if len(engaged) <= 1 else "medium",
            f"Only {len(engaged)} stakeholder(s) engaged in the last {RECENT_DAYS} days "
            f"with open deals in play",
            f"Recently engaged: {names}. Playbook wants {MULTITHREAD_MIN}+.",
            "playbook s5: single-threading is the biggest avoidable cause of stalls",
        ))
    personas = {c["PERSONA_TYPE"] for c in contacts}
    if "Economic Buyer" not in personas:
        out.append(_sig(
            "no_economic_buyer", "high",
            "No Economic Buyer contact exists on this account",
            "MEDDIC: pain must be felt at EB level to unlock budget",
            "playbook s2: MEDDIC qualification",
        ))
    else:
        eb_engaged = [c for c in engaged if c["PERSONA_TYPE"] == "Economic Buyer"]
        if not eb_engaged:
            ebs = [c for c in contacts if c["PERSONA_TYPE"] == "Economic Buyer"]
            out.append(_sig(
                "eb_gone_quiet", "medium",
                f"Economic Buyer exists but has not been engaged in {RECENT_DAYS} days",
                ", ".join(f"{c['FULL_NAME']}, last touch {c['_days_since_touch']} days ago"
                          for c in ebs),
                "playbook s5: get to the EB by end of proposal",
            ))
    if "Technical" not in personas:
        out.append(_sig(
            "no_technical_contact", "medium",
            "No technical stakeholder on the account",
            "Ines lost a deal to internal build because no IT contact was ever mapped",
            "AE interviews (Ines); ICP persona briefs",
        ))
    return out


def check_tickets(account_id):
    """Playbook 8: 'open and recent support tickets, especially P1s and
    unresolved P2s' before any customer call."""
    out = []
    for t in db.tickets_for(account_id):
        age = _days(t["CREATED_DATE"])
        if t["STATUS"] != "Resolved":
            sev = "high" if t["PRIORITY"] in ("P1", "P2") else "medium"
            out.append(_sig(
                "open_ticket", sev,
                f"Unresolved {t['PRIORITY']} ticket: {t['SUBJECT']}",
                f"Status {t['STATUS']}, opened {t['CREATED_DATE']} ({age} days ago)",
                "playbook s8: open tickets are required pre-call reading",
            ))
        elif t["PRIORITY"] == "P1" and age is not None and age <= P1_LOOKBACK:
            out.append(_sig(
                "recent_p1", "medium",
                f"P1 ticket in the last {P1_LOOKBACK} days (now resolved): {t['SUBJECT']}",
                f"Opened {t['CREATED_DATE']}, resolved {t['RESOLVED_DATE']}",
                "playbook s8: recent P1s colour the customer conversation",
            ))
    return out


def check_competitors(account_id):
    """Interview (Marcus): 'if somebody could just tell me this customer is
    also evaluating HiBob and here is the relevant page from the battlecard'."""
    out = []
    seen = set()
    for a in db.activities_for(account_id, limit=500):
        text = f"{a['SUBJECT']} {a['SUMMARY']}".lower()
        for comp in COMPETITORS:
            if comp in text and comp not in seen:
                # ignore routine sends of our own battlecard
                if "battlecard" in text and "mention" not in text:
                    continue
                seen.add(comp)
                out.append(_sig(
                    "competitor_mention", "high",
                    f"{comp.title()} came up in account activity",
                    f"{a['ACTIVITY_DATE']}: {a['SUMMARY']}",
                    f"battlecard: {comp} (knowledge/0{3 if comp=='workday' else 4}_battlecard_{comp}.md)",
                ))
    return out


def check_quiet(account_id):
    """Interview (Thomas): 'here's what changed since last time you looked'.
    The inverse also matters: nothing happening while deals are open."""
    open_opps = [o for o in db.opportunities_for(account_id) if o["STAGE"] in OPEN_STAGES]
    if not open_opps:
        return []
    acts = db.activities_for(account_id, limit=1)
    if not acts:
        return [_sig("quiet_account", "medium",
                     "No activity ever logged on this account despite open deals",
                     f"{len(open_opps)} open opportunity(ies)",
                     "playbook s8: recent activities are required pre-call reading")]
    gap = _days(acts[0]["ACTIVITY_DATE"])
    if gap is not None and gap > QUIET_DAYS:
        return [_sig(
            "quiet_account", "medium",
            f"No touchpoints in {gap} days while deals are open",
            f"Last activity {acts[0]['ACTIVITY_DATE']}: {acts[0]['SUMMARY'][:80]}",
            "playbook s8 + interview (Lena): the quietly slipping deal",
        )]
    return []


def check_reference_match(acct, competitor_signals):
    """Interview (Sofia): 'who else have we sold to that looks like this
    customer' is the question she always asks a colleague. Deterministic
    matching on industry, region, size band, and competitor-displacement tags,
    so the agent never has to guess whether a reference exists."""
    account_emp = acct.get("EMPLOYEE_COUNT") or 0
    competitors = {s["headline"].split()[0].lower() for s in competitor_signals}
    scored = []
    for cs in knowledge.case_studies():
        if cs["is_lost_deal"]:
            continue
        score, why = 0, []
        if cs["industry"].lower() == (acct.get("INDUSTRY") or "").lower():
            score += 2; why.append("same industry")
        if cs["region"].lower() == (acct.get("REGION") or "").lower():
            score += 1; why.append("same region")
        if account_emp and 0.4 <= cs["employees"] / max(account_emp, 1) <= 2.5:
            score += 1; why.append("similar size")
        tags = cs["use_as_reference_for"].lower()
        for comp in competitors:
            if comp in tags:
                score += 2; why.append(f"{comp} displacement reference")
        if score >= 2:
            scored.append((score, cs, why))
    if not scored:
        return []
    scored.sort(key=lambda x: -x[0])
    score, best, why = scored[0]
    return [_sig(
        "reference_match", "info",
        f"Matched customer reference: {best['name']} ({best['region']}, "
        f"{best['industry']}, {best['employees']} employees)",
        f"Match on: {', '.join(why)}. Doc section: '{best['section_heading']}'. "
        f"Tagged for: {best['use_as_reference_for']}",
        "customer_case_studies + interview (Sofia): 'who else looks like this customer'",
    )]


# --- The sweep ---------------------------------------------------------------

SEVERITY_ORDER = {"high": 0, "medium": 1, "info": 2}


def run_sweep(account_id: str) -> dict:
    """Run every check against one account. Returns the account header
    plus all signals, sorted most severe first."""
    acct = db.get_account(account_id)
    if not acct:
        return {"error": f"No account found with id {account_id}"}
    signals = []
    signals += check_usage_trend(account_id)
    signals += check_renewals(account_id, acct["SEGMENT"])
    signals += check_engagement(account_id)
    signals += check_tickets(account_id)
    competitor_signals = check_competitors(account_id)
    signals += competitor_signals
    signals += check_quiet(account_id)
    signals += check_reference_match(acct, competitor_signals)
    signals.sort(key=lambda s: SEVERITY_ORDER[s["severity"]])
    return {
        "account": {
            "id": acct["ACCOUNT_ID"], "name": acct["COMPANY_NAME"],
            "status": acct["STATUS"], "segment": acct["SEGMENT"],
            "region": acct["REGION"], "industry": acct["INDUSTRY"],
            "arr_eur": acct["ARR_EUR"], "owner": acct["OWNER_AE"],
        },
        "as_of": db.AS_OF,
        "signals": signals,
    }


SIGNAL_GLOSSARY = {
    "usage_drop": f"avg MAU in first two tracked months vs last two, down "
                  f"{abs(USAGE_DROP_WARN)}%+ (severe at {abs(USAGE_DROP_SEVERE)}%+); "
                  "corrupted negative rows excluded [playbook s8]",
    "single_threaded": f"fewer than {MULTITHREAD_MIN} stakeholders engaged in the "
                       f"last {RECENT_DAYS} days while deals are open, computed "
                       "from the activity log [playbook s5]",
    "no_economic_buyer": "no contact with persona 'Economic Buyer' exists on the "
                         "account [playbook s2, MEDDIC]",
    "eb_gone_quiet": f"an EB exists but had no touch in {RECENT_DAYS} days [playbook s5]",
    "no_technical_contact": "no Technical-persona contact on the account [AE interviews]",
    "competitor_mention": "HiBob or Workday named in activity subject/summary "
                          "(routine battlecard sends excluded) [battlecards]",
    "renewal_window": f"open renewal closing within {RENEWAL_WINDOW_MM} days "
                      f"(MM) / {RENEWAL_WINDOW_ENT} (ENT) [playbook s6]",
    "overdue_close_date": "open deal whose close date has passed [playbook s7]",
    "stalled_deal": f"open deal sitting in one stage over {STALL_DAYS} days [playbook s3]",
    "quiet_account": f"no touchpoints in {QUIET_DAYS}+ days with deals open [playbook s8]",
    "open_ticket": "any unresolved support ticket [playbook s8]",
    "recent_p1": f"P1 ticket in the last {P1_LOOKBACK} days, even if resolved [playbook s8]",
    "usage_data_quality": "impossible values (negative MAU/logins) in usage rows",
    "duplicate_open_opps": "more than one open opportunity of the same type",
}


def scan_book(ae_email: str, signal: str | None = None) -> dict:
    """Exhaustive scan: every account in the AE's book, every rule, no
    truncation. This answers 'which accounts have X' with guaranteed recall;
    the model must never answer set-membership questions from a top-N view.
    The 'method' block is machine-authored provenance."""
    accounts = db.accounts_for_ae(ae_email)
    results = []
    for acct in accounts:
        sweep = run_sweep(acct["ACCOUNT_ID"])
        sigs = [s for s in sweep["signals"] if s["signal"] != "reference_match"]
        if signal:
            sigs = [s for s in sigs if s["signal"] == signal]
        if not sigs:
            continue
        entry = {
            "account_id": acct["ACCOUNT_ID"],
            "name": acct["COMPANY_NAME"],
            "status": acct["STATUS"],
            "segment": acct["SEGMENT"],
        }
        if signal:
            # filtered scan: full detail, the user asked about this signal
            entry["signals"] = [{"severity": s["severity"],
                                 "headline": s["headline"],
                                 "evidence": s["evidence"]} for s in sigs]
        else:
            # full scan: compact signal names only, so the complete book fits
            # untruncated; drill into an account with run_risk_sweep
            entry["high"] = [s["signal"] for s in sigs if s["severity"] == "high"]
            entry["medium"] = [s["signal"] for s in sigs if s["severity"] == "medium"]
        results.append(entry)
    return {
        "method": {
            "scope": f"all {len(accounts)} accounts in the book of {ae_email}, "
                     "each checked individually",
            "definition": (SIGNAL_GLOSSARY.get(signal, f"unknown signal '{signal}'")
                           if signal else "every rule in the signal engine"),
            "truncated": False,
            "accounts_scanned": len(accounts),
            "accounts_flagged": len(results),
        },
        "results": results,
    }


def rank_book(ae_email: str, max_accounts: int = 5) -> list[dict]:
    """THE prioritization framework, shared by the chat agent and the digest
    so 'what should I focus on' has one answer everywhere.

    Ranking: number of high-severity signals first, then medium, then the
    nearest open close date as an urgency tiebreaker."""
    ranked = []
    for acct in db.accounts_for_ae(ae_email):
        sweep = run_sweep(acct["ACCOUNT_ID"])
        sigs = [s for s in sweep["signals"] if s["signal"] != "reference_match"]
        highs = [s for s in sigs if s["severity"] == "high"]
        meds = [s for s in sigs if s["severity"] == "medium"]
        if not (highs or meds):
            continue
        opps = [o for o in db.opportunities_for(acct["ACCOUNT_ID"])
                if o["STAGE"] in OPEN_STAGES and o["CLOSE_DATE"]]
        next_close = min((str(o["CLOSE_DATE"]) for o in opps), default="9999")
        ranked.append({"sweep": sweep, "highs": len(highs), "meds": len(meds),
                       "next_close_date": next_close})
    ranked.sort(key=lambda r: (-r["highs"], -r["meds"], r["next_close_date"]))
    return ranked[:max_accounts]


if __name__ == "__main__":
    import json
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else "Halcyon"
    matches = db.find_accounts(text)
    if not matches:
        print(f"No account matches '{text}'")
        sys.exit(1)
    print(json.dumps(run_sweep(matches[0]["ACCOUNT_ID"]), indent=2, default=str))
