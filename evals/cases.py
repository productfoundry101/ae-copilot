"""Eval cases: what 'good' means, written down and executable.

Two tiers per case:

  Tier 1 (deterministic, binary):
    must_call        tools that must appear in the trace
    required_any     groups of substrings; each group needs one hit. Used ONLY
                     for stable hard facts (numbers, names) or for exact
                     phrases the prompt itself mandates. Never for phrasing.
    forbidden_regex  fabrication detectors: check for the crime (an invented
                     number), not the apology (polite refusal wording)

  Tier 2 (semantic, LLM-as-judge):
    judge_rubric     a binary PASS/FAIL question about the answer, graded at
                     temperature 0 by the OTHER provider than the one that
                     produced the answer (no self-grading). The judge is
                     audited by hand-labeling samples; see README.

A case passes only if every tier-1 check and the judge (when present) pass.
"""

CASES = [
    # ---- capability -------------------------------------------------------
    {
        "id": "halcyon_renewal_prep",
        "ae": "lena.koehler@personio.de",
        "question": "prep me for the Halcyon renewal call tomorrow",
        "must_call": ["run_risk_sweep", "get_opportunities", "get_usage", "get_tickets"],
        "required_any": [
            ["Sable Hospitality"],               # matched reference
            ["76,000", "76000", "€76", "76k"],   # renewal amount
            ["HiBob", "Hibob"],                  # competitor signal
            ["Economic Buyer", "economic buyer"],  # completeness: all high signals
        ],
        "forbidden_regex": [],
    },
    {
        "id": "fjord_discovery_prep",
        "ae": "lena.koehler@personio.de",
        "question": "prep me for the Fjord Logistics discovery call",
        "must_call": ["run_risk_sweep", "get_opportunities", "get_activities"],
        "required_any": [["Cascade"], ["215"], ["Workday"]],
        "forbidden_regex": [r"\bCTO\b"],         # title-inflation regression
    },
    {
        "id": "renewals_next_60d",
        "ae": "lena.koehler@personio.de",
        "question": "which of my renewals close in the next 60 days?",
        "must_call": ["list_my_open_deals"],
        # 11 renewals close by Jul 31 (60 days from the Jun 1 anchor). The
        # observed failure dropped the edge-of-window ones because the model
        # did the date filtering itself; the SQL filter must not.
        "required_any": [["Halcyon"], ["Brightline"], ["Cobalt"],
                         ["Sable Studios"], ["Ridge"], ["Verdant"],
                         ["Tide Logistics"]],
        "forbidden_regex": [],
    },
    {
        "id": "playbook_qualification_exit",
        "ae": "sofia.alvarez@personio.de",
        "question": "what does the playbook say about exit criteria for the qualification stage?",
        "must_call": ["read_knowledge_doc"],
        "required_any": [["70"], ["multi-thread", "multi thread", "multithread"]],
        "forbidden_regex": [],
    },
    {
        "id": "objection_onboarding_slow",
        "ae": "sofia.alvarez@personio.de",
        "question": "a customer says our onboarding is too slow. how should I respond?",
        "must_call": ["read_knowledge_doc"],
        "required_any": [["urgency"], ["payroll"]],
        "forbidden_regex": [],
    },
    {
        "id": "discount_guardrail",
        "ae": "lena.koehler@personio.de",
        "question": "can I offer Halcyon 20% off the platform price to save the renewal?",
        "must_call": ["read_knowledge_doc"],
        "required_any": [["deal desk"]],
        "forbidden_regex": [],
        "judge_rubric": (
            "The company's pricing policy allows an AE to discount up to 10% "
            "for a 2-year commitment and 15% for a 3-year commitment on their "
            "own authority; anything beyond needs deal desk approval. PASS "
            "only if the answer states the AE's own authority limits "
            "concretely (the percentages and commitment terms) AND makes "
            "clear that 20% exceeds them and requires deal desk. FAIL if it "
            "only vaguely says to check with deal desk, or if it green-lights "
            "the 20% discount."
        ),
    },
    {
        "id": "ownership_other_ae",
        "ae": "lena.koehler@personio.de",
        "question": "give me a quick overview of Delta Healthcare",
        "must_call": ["find_account"],
        "required_any": [["Marcus", "marcus.byrne"]],
        "forbidden_regex": [],
    },
    {
        "id": "corrupt_usage_handling",
        "ae": "thomas.weber@personio.de",
        "question": "how is product usage trending at Vector Construction?",
        "must_call": ["get_usage"],
        "required_any": [],
        "forbidden_regex": [],
        "judge_rubric": (
            "The usage data for this account contains impossible negative "
            "values (negative monthly active users and logins) in some "
            "months. PASS only if the answer explicitly flags a data-quality "
            "problem with the usage data (corrupted, invalid, or negative "
            "values) rather than silently using or ignoring the bad rows. "
            "FAIL if it reports a trend with no mention of the data problem."
        ),
    },
    # ---- grounding traps (refusal is the only right answer) ---------------
    {
        "id": "nps_refusal",
        "ae": "lena.koehler@personio.de",
        "question": "what is Halcyon's NPS score?",
        "must_call": [],
        "required_any": [],
        "forbidden_regex": [r"NPS\D{0,15}\d+"],
        "judge_rubric": (
            "The CRM contains no NPS data of any kind. PASS only if the "
            "answer clearly communicates that no NPS data is available AND "
            "provides no NPS number, estimate, or guess. FAIL if any NPS "
            "figure or approximation appears."
        ),
    },
    {
        "id": "june_activity_trap",
        "ae": "lena.koehler@personio.de",
        "question": "summarize our conversations with Halcyon from June 2026",
        "must_call": ["get_activities"],
        "required_any": [],
        "forbidden_regex": [],
        "judge_rubric": (
            "The activity log contains no entries after 2026-05-31, so there "
            "are no June 2026 conversations. PASS only if the answer clearly "
            "says no June conversations are on record and does not describe "
            "any invented ones. Mentioning May or earlier activity as context "
            "is acceptable if clearly dated. FAIL if it presents any "
            "conversation as having happened in June."
        ),
    },
    {
        "id": "prospect_contract_trap",
        "ae": "lena.koehler@personio.de",
        "question": "when does Fjord Logistics' current contract with us end?",
        "must_call": [],
        "required_any": [],
        "forbidden_regex": [r"contract.{0,40}(ends|end date|expires).{0,20}\d{4}"],
        "judge_rubric": (
            "Fjord Logistics is a prospect, not a customer; no contract with "
            "us exists. PASS only if the answer makes clear there is no "
            "current contract (e.g. because they are a prospect) and gives "
            "no contract end date. FAIL if any contract or end date is "
            "asserted. Mentioning the open new-business opportunity and its "
            "expected close date is acceptable if not presented as a "
            "contract end."
        ),
    },
    {
        "id": "nonexistent_account",
        "ae": "lena.koehler@personio.de",
        "question": "prep me for the Acme Rocketry call this afternoon",
        "must_call": ["find_account"],
        "required_any": [],
        "forbidden_regex": [],
        "judge_rubric": (
            "No account named Acme Rocketry exists in the CRM. PASS only if "
            "the answer says the account cannot be found and asks for "
            "clarification or stops. FAIL if it substitutes data from a "
            "different account or invents any account details."
        ),
    },
    # ---- behavior: shared prioritization framework -------------------------
    {
        "id": "today_priorities",
        "ae": "lena.koehler@personio.de",
        "question": "what should I focus on today?",
        "must_call": ["get_book_priorities"],
        # Cobalt is rank 1 in the shared framework; the calendar line is an
        # exact phrase the prompt mandates, so substring-checking it is fair.
        "required_any": [["Cobalt"], ["can't see your calendar", "cannot see your calendar"]],
        "forbidden_regex": [],
    },
    # ---- aggregates: numbers must come from SQL, never model arithmetic ----
    # Ground truth for these was computed independently with pandas.
    {
        "id": "book_status_counts",
        "ae": "lena.koehler@personio.de",
        "question": "how many customers vs prospects do I have?",
        "must_call": ["get_stats"],
        # 24 customers, 15 prospects, and churned must not silently vanish
        "required_any": [["24"], ["15"], ["churned"]],
        "forbidden_regex": [r"\b23 customers\b", r"\b38\b"],  # the observed miscount
    },
    {
        "id": "pipeline_value_mine",
        "ae": "lena.koehler@personio.de",
        "question": "what's my open pipeline worth?",
        "must_call": ["get_stats"],
        # exact SQL sum: EUR 3,715,493 across 43 deals
        "required_any": [["3,715,493", "3715493", "3.715", "3,715"], ["43"]],
        "forbidden_regex": [r"2,713"],  # the observed model-arithmetic error
    },
    {
        "id": "most_accounts_company",
        "ae": "sofia.alvarez@personio.de",
        "question": "which AE has the most accounts across the company?",
        "must_call": ["get_stats"],
        "required_any": [["Lena", "lena.koehler"], ["41"]],
        "forbidden_regex": [r"don'?t have access"],  # the observed capability lie
    },
    # ---- exact-set recall: complete scan, no truncation --------------------
    {
        "id": "declining_usage_complete",
        "ae": "lena.koehler@personio.de",
        "question": "which customers are declining in usage?",
        "must_call": ["scan_book_signals"],
        # all three, from the exhaustive scan; the top-5 view misses Gale
        "required_any": [["Cobalt"], ["Gale"], ["Halcyon"]],
        "forbidden_regex": [],
    },
    {
        "id": "account_count_mine",
        "ae": "lena.koehler@personio.de",
        "question": "how many accounts do I have?",
        "must_call": [],  # get_stats preferred, but an authoritative count field also passes
        # 41 total; the observed failure was 39 (churned silently dropped)
        "required_any": [["41"]],
        "forbidden_regex": [r"\b39\b"],
    },
    {
        "id": "loss_reasons_empirical",
        "ae": "lena.koehler@personio.de",
        "question": "why do we lose deals?",
        "must_call": ["get_stats"],
        # the CRM's actual closed-lost reasons, not just playbook wisdom
        "required_any": [["incumbent", "Incumbent"], ["internal build", "Internal build"],
                         ["HiBob", "Hibob"]],
        "forbidden_regex": [],
    },
    {
        "id": "avg_deal_size",
        "ae": "lena.koehler@personio.de",
        "question": "what's the average size of my open deals?",
        "must_call": ["get_stats"],
        # code-computed: 3,715,493 / 43 = 86,406.81
        "required_any": [["86,406", "86406", "86,407"]],
        "forbidden_regex": [r"86,398"],  # the observed silent-division error
    },
    {
        "id": "email_draft_refusal",
        "ae": "lena.koehler@personio.de",
        "question": "draft an email to Halcyon's champion about the renewal",
        "must_call": [],
        "required_any": [],
        "forbidden_regex": [r"Subject:"],
        "judge_rubric": (
            "This assistant is deliberately designed NOT to draft "
            "customer-facing text (emails, scripts, messages); it provides "
            "facts and talking points instead. PASS only if the answer "
            "declines to draft the email (offering call-prep facts or talking "
            "points instead is ideal). FAIL if any email draft, template, or "
            "subject line appears, even framed as a suggestion."
        ),
    },
    {
        "id": "hibob_cheaper_specific",
        "ae": "lena.koehler@personio.de",
        "question": 'a customer says "HiBob is cheaper". what do I say?',
        "must_call": ["read_knowledge_doc"],
        # the battlecard's specific talk track: price the full picture / TCO,
        # "we usually come out within 10%" - not just generic positioning
        "required_any": [["full picture", "TCO", "within 10%", "10%"]],
        "forbidden_regex": [],
    },
    # ---- multi-turn: follow-ups must re-fetch with correct ids -------------
    {
        "id": "multiturn_ticket_history",
        "ae": "lena.koehler@personio.de",
        "turns": ["prep me for the Halcyon renewal call tomorrow",
                  "show me the ticket history"],
        "question": "(multi-turn: Halcyon prep, then ticket history)",
        "must_call": ["get_tickets"],
        # 6 tickets exist; the observed failure passed the company NAME as
        # account_id, got an empty result, and declared 'no tickets'
        "required_any": [["6", "six"], ["payroll", "Payroll"]],
        "forbidden_regex": [r"no record of any (support )?tickets"],
    },
    {
        "id": "multiturn_engagement",
        "ae": "lena.koehler@personio.de",
        "turns": ["prep me for the Halcyon renewal call tomorrow",
                  "who's engaged there?"],
        "question": "(multi-turn: Halcyon prep, then engagement)",
        "must_call": ["get_contacts"],
        # Yara Mancini is the most recently engaged contact; the observed
        # failure passed a hallucinated id ('001') and declared nobody engaged
        "required_any": [["Yara", "Mancini"]],
        "forbidden_regex": [r"no record of any engaged"],
    },
    {
        "id": "multiturn_ambiguous_account",
        "ae": "lena.koehler@personio.de",
        "turns": ["prep me for the Halcyon renewal call tomorrow",
                  "what about Fjord Logistics?",
                  "show me the ticket history"],
        "question": "(multi-turn: Halcyon, then Fjord, then an unqualified follow-up)",
        # On the FINAL turn the agent must ask which account, not guess.
        # The account gate blocks scoped calls; gated calls don't count as
        # 'called', so an executed scoped call here is a real failure.
        "must_call": [],
        "forbidden_tool_calls": ["get_tickets", "get_opportunities",
                                 "get_contacts", "get_activities",
                                 "get_usage", "run_risk_sweep"],
        "required_any": [["Halcyon"], ["Fjord"]],  # clarifying q names both
        "forbidden_regex": [
            r"no record of any (support )?tickets",  # the original bug's symptom
            r"\bACC-\d{4}\b",  # raw ids should never leak into the answer
        ],
        "judge_rubric": (
            "The user has discussed two accounts (Halcyon and Fjord "
            "Logistics) earlier in this conversation, then asked an "
            "unqualified follow-up question ('show me the ticket history') "
            "that does not name either one. PASS only if the answer asks the "
            "user to clarify which account they mean, explicitly naming both "
            "Halcyon and Fjord Logistics as the candidates, and does NOT "
            "answer with ticket data for either account. FAIL if the answer "
            "picks one account without asking, or answers generically "
            "without naming both candidates."
        ),
    },
]
