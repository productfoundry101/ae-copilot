"""Proactive morning digest: the 'tell me what I'd miss' capability,
unprompted.

Runs the SAME signal engine as the chat assistant across an AE's entire
book, ranks accounts by severity, and composes a short morning brief.
Chat and digest are two delivery modes of one brain; nothing is duplicated.

Delivery is a DRY RUN by design: the email is rendered to the terminal and
saved to digests/, and the send step prints what it would do. Wiring SMTP
or a Slack webhook proves nothing about the product; the composed brief
proves everything. See README for the cron schedule that would run this
every weekday morning.

Usage:
  python digest.py --ae lena.koehler@personio.de     one AE
  python digest.py --all                             every AE
  python digest.py --ae ... --no-llm                 skip the LLM summary
                                                     (pure rule output)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import db
import signals
from agent import AE_NAMES, PROVIDER

DIGESTS_DIR = Path(__file__).parent / "digests"
SEV_RANK = {"high": 0, "medium": 1, "info": 2}
MAX_ACCOUNTS = 5  # a digest an AE will actually read, not a report


def sweep_book(ae_email: str) -> list[dict]:
    """Rank the AE's book with the shared prioritization framework in
    signals.rank_book, the same one the chat agent uses. One framework,
    one answer, in every surface."""
    return signals.rank_book(ae_email, max_accounts=MAX_ACCOUNTS)


def render_rule_based(ae_email: str, ranked: list[dict]) -> str:
    """Deterministic digest body straight from the signal engine."""
    lines = [f"Subject: Your book this morning - {len(ranked)} accounts need attention",
             "", f"Good morning {AE_NAMES.get(ae_email, ae_email).split()[0]},", ""]
    for r in ranked:
        a = r["sweep"]["account"]
        lines.append(f"## {a['name']} ({a['segment']}, {a['status']})")
        for s in r["sweep"]["signals"]:
            if s["severity"] in ("high", "medium"):
                lines.append(f"- [{s['severity']}] {s['headline']}")
                lines.append(f"    evidence: {s['evidence']}")
        lines.append("")
    lines.append(f"Generated {datetime.now():%Y-%m-%d %H:%M} from CRM data as of {db.AS_OF}.")
    return "\n".join(lines)


def polish_with_llm(ae_email: str, rule_text: str) -> str:
    """Optional: have the model rewrite the rule output as a tight brief.
    The model may compress and prioritise but gets ONLY fired signals as
    input, so it cannot introduce new facts."""
    prompt = (
        "Rewrite this raw signal report as a morning email for the AE. Keep "
        "every fact and number exactly as given, add nothing. Max 200 words. "
        "Lead with the single most urgent account and why. Keep the Subject "
        "line. Sign off as 'AE Copilot', never a placeholder.\n\n" + rule_text
    )
    if PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(model=__import__("agent").ANTHROPIC_MODEL,
                                      max_tokens=800,
                                      messages=[{"role": "user", "content": prompt}])
        return resp.content[0].text
    else:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=__import__("agent").OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content


def send_email(to: str, body: str):
    """Deliberate stub. In production this is a Slack DM via webhook (AEs
    live in Slack, not email). Left unimplemented in the prototype: the
    composed brief is the product, delivery is plumbing."""
    print(f"\n[DRY RUN] Would deliver this digest to {to} "
          f"(prod: Slack DM via webhook).\n")


def run_for(ae_email: str, use_llm: bool = True):
    ranked = sweep_book(ae_email)
    if not ranked:
        print(f"{ae_email}: book is clean this morning, no digest sent.")
        return
    body = render_rule_based(ae_email, ranked)
    if use_llm:
        try:
            body = polish_with_llm(ae_email, body)
        except Exception as e:
            print(f"(LLM polish unavailable: {e}; sending rule-based version)")
    DIGESTS_DIR.mkdir(exist_ok=True)
    out = DIGESTS_DIR / f"{datetime.now():%Y-%m-%d}_{ae_email.split('@')[0]}.md"
    out.write_text(body)
    print("=" * 70)
    print(body)
    print("=" * 70)
    print(f"Saved to {out}")
    send_email(ae_email, body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", help="AE email")
    ap.add_argument("--all", action="store_true", help="run for every AE")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip LLM polish, pure rule-based output")
    args = ap.parse_args()
    targets = db.all_aes() if args.all else [args.ae] if args.ae else []
    if not targets:
        ap.error("pass --ae EMAIL or --all")
    for email in targets:
        run_for(email, use_llm=not args.no_llm)
