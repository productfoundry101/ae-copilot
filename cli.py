"""Terminal chat with the AE Copilot. Demo-day fallback for the Streamlit UI.

Usage:  python cli.py
"""

from __future__ import annotations

import db
import agent


def main():
    aes = db.all_aes()
    print("\nAE Copilot (terminal). Who are you?")
    for i, email in enumerate(aes, 1):
        name = agent.AE_NAMES.get(email, email)
        n_accounts = len(db.accounts_for_ae(email))
        print(f"  {i}. {name} ({n_accounts} accounts)")
    choice = input("Number: ").strip()
    ae_email = aes[int(choice) - 1]
    print(f"\nSigned in as {agent.AE_NAMES.get(ae_email, ae_email)}. "
          f"Provider: {agent.PROVIDER}. Data mode: {db.DATA_MODE}. "
          f"As-of date: {db.AS_OF}.")
    print("Ask about your accounts. 'exit' to quit.\n")

    history: list[dict] = []
    all_tool_calls: list[dict] = []  # powers the account gate across turns
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        history.append({"role": "user", "content": q})
        result = agent.run_turn(history, ae_email,
                                prior_tool_calls=all_tool_calls)
        all_tool_calls.extend(result["tool_calls"])
        if result["tool_calls"]:
            names = ", ".join(c["name"] for c in result["tool_calls"])
            print(f"   [tools used: {names}]")
        print(f"\ncopilot> {result['answer']}\n")
        history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    main()
