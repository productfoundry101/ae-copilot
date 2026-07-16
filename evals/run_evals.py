"""Run the eval suite against the live agent.

Usage:
  python evals/run_evals.py                 all cases, once each
  python evals/run_evals.py nps             only cases whose id contains 'nps'
  python evals/run_evals.py --runs 3        reliability@3: every case 3 times,
                                            pass-rate reported per case

A case is only counted reliable when it passes EVERY run: single-run green
proves correctness-today, repeated runs measure stability under sampling.
Cases at 2/3 are flaky — their answer path depends on model judgment
somewhere it shouldn't, which is a diagnosis, not just a score.

Every prompt or rule change should be followed by a full run. Failures on
grounding traps are critical: an invented fact is worse than any missing
feature. Results are saved to evals/results_*.json (provider and run count
in the filename) so runs can be compared across providers and over time.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent  # noqa: E402
from cases import CASES  # noqa: E402

JUDGE_PROMPT = """You are grading one answer from an AI sales assistant against a binary rubric.

Rubric: {rubric}

User question: {question}

Assistant's answer:
---
{answer}
---

Grade strictly against the rubric only. Reply with exactly one line:
PASS
or
FAIL: <one-sentence reason>"""


def judge_answer(rubric: str, question: str, answer: str) -> tuple[bool, str]:
    """Tier 2: semantic grading by the OTHER provider (no self-grading).
    OpenAI judge calls run at temperature 0; Anthropic calls omit
    temperature entirely (newer Claude models reject the param). Falls back
    to the same provider only if the other errors, and that fallback is
    loud (printed + tagged) so a self-graded run can never pass silently as
    a cross-provider one."""
    prompt = JUDGE_PROMPT.format(rubric=rubric, question=question, answer=answer)
    order = (["anthropic", "openai"] if agent.PROVIDER == "openai"
             else ["openai", "anthropic"])
    last_err = None
    for i, provider in enumerate(order):
        try:
            if provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic()
                # No temperature param: newer Claude models (e.g.
                # claude-sonnet-5) reject it outright ("temperature is
                # deprecated for this model"), which was silently causing
                # every judge call here to fail and fall back to self-grading.
                resp = agent._with_retries(lambda: client.messages.create(
                    model=agent.ANTHROPIC_MODEL, max_tokens=100,
                    messages=[{"role": "user", "content": prompt}]))
                # content[0] isn't reliably the answer: some Claude models
                # emit a ThinkingBlock first, then the TextBlock. Filter by
                # type instead of assuming position (same pattern agent.py
                # already uses for the main loop).
                verdict = "".join(
                    b.text for b in resp.content if b.type == "text"
                ).strip()
            else:
                from openai import OpenAI
                client = OpenAI()
                resp = agent._with_retries(lambda: client.chat.completions.create(
                    model=agent.OPENAI_MODEL, temperature=0,
                    messages=[{"role": "user", "content": prompt}]))
                verdict = resp.choices[0].message.content.strip()
            self_graded = provider == agent.PROVIDER
            tag = f"[judge={provider}" + (" SELF-GRADED" if self_graded else "") + "]"
            if self_graded:
                print(f"        WARNING: judge fell back to the agent's own "
                      f"provider ({provider}); cross-provider judge "
                      f"({order[0]}) failed with: {last_err}")
            return verdict.upper().startswith("PASS"), f"{tag} {verdict}"
        except Exception as e:
            print(f"        judge provider '{provider}' failed: {e!r}")
            last_err = e
    return False, f"judge unavailable: {last_err}"


def check_case(case: dict) -> dict:
    """Run a case: single question, or a scripted multi-turn conversation
    via 'turns'. All checks apply to the FINAL turn's answer and tool calls
    (multi-turn failures are things like answering follow-ups from memory
    or passing garbage account ids deep into a conversation)."""
    turns = case.get("turns") or [case["question"]]
    history: list[dict] = []
    prior_calls: list[dict] = []  # accumulated across turns, powers the gate
    t0 = time.time()
    result = None
    try:
        for q in turns:
            history.append({"role": "user", "content": q})
            result = agent.run_turn(history, case["ae"],
                                    prior_tool_calls=prior_calls)
            history.append({"role": "assistant", "content": result["answer"]})
            prior_calls.extend(result["tool_calls"])
    except Exception as e:
        return {"id": case["id"], "pass": False,
                "failures": [f"agent crashed: {e}"], "latency_s": 0}
    answer = result["answer"]
    # gated (blocked) calls never executed, so they don't count as "called"
    called = {c["name"] for c in result["tool_calls"] if not c.get("gated")}
    failures = []
    for tool in case["must_call"]:
        if tool not in called:
            failures.append(f"never called required tool: {tool}")
    for tool in case.get("forbidden_tool_calls", []):
        if tool in called:
            failures.append(f"called forbidden tool on this turn: {tool}")
    for group in case["required_any"]:
        if not any(term.lower() in answer.lower() for term in group):
            failures.append(f"answer missing all of: {group}")
    for pattern in case["forbidden_regex"]:
        if re.search(pattern, answer, re.IGNORECASE):
            failures.append(f"matched forbidden pattern: {pattern}")
    judge_note = None
    if case.get("judge_rubric"):
        ok, judge_note = judge_answer(case["judge_rubric"], case["question"], answer)
        if not ok:
            failures.append(f"judge: {judge_note}")
    return {"id": case["id"], "pass": not failures, "failures": failures,
            "latency_s": round(time.time() - t0, 1),
            "tools": sorted(called), "judge": judge_note, "answer": answer}


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", default="",
                    help="only run cases whose id contains this substring")
    ap.add_argument("--runs", type=int, default=1,
                    help="repetitions per case (reliability@N)")
    args = ap.parse_args()
    cases = [c for c in CASES if args.filter in c["id"]]
    pause = float(os.getenv("EVAL_PAUSE_S", "15"))
    print(f"Running {len(cases)} eval case(s) x {args.runs} run(s) "
          f"with provider={agent.PROVIDER}\n")
    results = []
    first = True
    for case in cases:
        runs = []
        for _ in range(args.runs):
            if not first:
                time.sleep(pause)  # stay under tokens-per-minute rate limits
            first = False
            runs.append(check_case(case))
        passes = sum(r["pass"] for r in runs)
        if args.runs == 1:
            mark = "PASS " if passes else "FAIL "
        elif passes == args.runs:
            mark = f"PASS {passes}/{args.runs}"
        elif passes == 0:
            mark = f"FAIL {passes}/{args.runs}"
        else:
            mark = f"FLAKY {passes}/{args.runs}"
        lats = ", ".join(f"{r['latency_s']}s" for r in runs)
        print(f"  {mark:9} {case['id']}  ({lats})")
        seen = set()
        for r in runs:
            for f in r["failures"]:
                if f not in seen:
                    seen.add(f)
                    print(f"        - {f}")
        results.append({"id": case["id"], "passes": passes,
                        "total": args.runs, "runs": runs})
    reliable = sum(1 for r in results if r["passes"] == r["total"])
    flaky = sum(1 for r in results if 0 < r["passes"] < r["total"])
    failing = sum(1 for r in results if r["passes"] == 0)
    line = f"\n{reliable}/{len(results)} cases pass every run"
    if args.runs > 1:
        line += f" (reliability@{args.runs}) · {flaky} flaky · {failing} failing"
    print(line)
    self_graded = sum(1 for res in results for r in res["runs"]
                      if r.get("judge") and "SELF-GRADED" in r["judge"])
    if self_graded:
        print(f"WARNING: {self_graded} run(s) were judged by the agent's "
              f"own provider (cross-provider judge unavailable) — treat "
              f"their pass/fail with less confidence")
    out = (Path(__file__).parent /
           f"results_{datetime.now():%Y%m%d_%H%M%S}_{agent.PROVIDER}_x{args.runs}.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"Details saved to {out}")
    if reliable < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
