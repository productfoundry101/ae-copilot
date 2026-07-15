"""Data access layer. Every other module gets data through this file only.

Two modes, switched with the DATA_MODE environment variable:

  snapshot (default)  reads the CSV files in ./data using DuckDB, a small
                      in-process SQL engine. Fast, free, works offline.
                      Used for development and testing.

  live                runs the exact same SQL against Snowflake.
                      Used for the real demo.

Why both: the agent's logic never changes between modes, so we can develop
and test deterministically on the snapshot, then flip one env var for the
live demo. Nothing else in the codebase knows or cares where data lives.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_MODE = os.getenv("DATA_MODE", "snapshot").strip().lower()
if DATA_MODE not in ("snapshot", "live"):
    raise ValueError(
        f"DATA_MODE must be 'snapshot' or 'live', got '{DATA_MODE}'. "
        "Check your .env file.")
DATA_DIR = Path(__file__).parent / "data"

# The dataset's clock stops at end of May 2026 (last activity: 2026-05-31).
# All relative-time reasoning ("what changed in the last 2 weeks") is anchored
# to AS_OF so it behaves the way it would on live, current data.
# In production this constant disappears and we use the real current date.
AS_OF = os.getenv("AS_OF_DATE", "2026-06-01")

# Logical table name -> physical name in each mode.
TABLES = {
    "accounts":      {"snapshot": "accounts",      "live": "PERSONIO.CRM.ACCOUNTS"},
    "contacts":      {"snapshot": "contacts",      "live": "PERSONIO.CRM.CONTACTS"},
    "opportunities": {"snapshot": "opportunities", "live": "PERSONIO.CRM.OPPORTUNITIES"},
    "activities":    {"snapshot": "activities",    "live": "PERSONIO.CRM.ACTIVITIES"},
    "usage":         {"snapshot": "usage_data",    "live": "PERSONIO.PRODUCT.USAGE"},
    "tickets":       {"snapshot": "tickets",       "live": "PERSONIO.SUPPORT.TICKETS"},
}

_conn = None


def _connect():
    """Create the database connection once and reuse it."""
    global _conn
    if _conn is not None:
        return _conn

    if DATA_MODE == "snapshot":
        import duckdb

        _conn = duckdb.connect(":memory:")
        # Register each CSV as a queryable view.
        for logical, physical in TABLES.items():
            csv_path = DATA_DIR / f"{logical}.csv"
            _conn.execute(
                f"CREATE VIEW {physical['snapshot']} AS "
                f"SELECT * FROM read_csv_auto('{csv_path}', header=true)"
            )
    else:
        import snowflake.connector

        kwargs = dict(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_XS_WH"),
            database="PERSONIO",
            role=os.environ.get("SNOWFLAKE_ROLE", "APPLICANT_FR"),
        )
        # Human Snowflake users with MFA can't authenticate with a plain
        # password from code. Supported alternatives, in precedence order:
        #   SNOWFLAKE_PAT               programmatic access token (Snowsight UI)
        #   SNOWFLAKE_PRIVATE_KEY_PATH  key-pair auth (production pattern)
        #   SNOWFLAKE_AUTHENTICATOR     e.g. externalbrowser (needs SAML IdP)
        #   SNOWFLAKE_PASSWORD          plain password (service accounts only)
        pat = os.environ.get("SNOWFLAKE_PAT", "")
        key_path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH", "")
        authenticator = os.environ.get("SNOWFLAKE_AUTHENTICATOR", "")
        if pat:
            # PAT passed via the `token` param WITH the explicit authenticator
            # (confirmed working config with the Snowflake admin team). Passing
            # it as `password` returns "token is invalid"; `token` is correct.
            kwargs["token"] = pat
            kwargs["authenticator"] = "PROGRAMMATIC_ACCESS_TOKEN"
        elif key_path:
            kwargs["private_key_file"] = key_path
            if os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"):
                kwargs["private_key_file_pwd"] = os.environ[
                    "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"]
        elif authenticator:
            kwargs["authenticator"] = authenticator
            if authenticator.lower() == "externalbrowser":
                kwargs["client_store_temporary_credential"] = True
        else:
            kwargs["password"] = os.environ["SNOWFLAKE_PASSWORD"]
        _conn = snowflake.connector.connect(**kwargs)
    return _conn


# SQL narration only at COPILOT_DEBUG=2; level 1 keeps the story readable.
DEBUG_SQL = os.getenv("COPILOT_DEBUG", "").strip() == "2"

# Rolling log of the actual queries run, rendered for humans (tables resolved,
# parameters inlined). tools.py resets this per tool call and reads it back to
# show the real SQL in the UI's "How this was calculated" panel.
_QUERY_LOG: list[str] = []


def reset_query_log() -> None:
    _QUERY_LOG.clear()


def captured_sql() -> list[str]:
    return list(_QUERY_LOG)


def _inline(sql: str, params: tuple) -> str:
    """Inline ? parameters into an already-table-resolved SQL string, for
    display only (never executed)."""
    out = sql
    for p in params:
        out = out.replace("?", f"'{p}'" if isinstance(p, str) else str(p), 1)
    return " ".join(out.split())  # collapse whitespace to one line


def query(sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT and return rows as a list of dicts.

    SQL templates use {table} placeholders for table names and ? for
    parameters. Parameters are always bound, never string-formatted into
    the SQL, which rules out injection even though the model never writes
    SQL itself (all queries in this codebase are hand-written).
    """
    # Two different substitutions, resolved at two different times:
    # {table} names are swapped in Python, here, before the DB ever sees the
    # query (just picks which physical table to read). ? placeholders are
    # resolved by the DB driver itself, from `params`, at execution time
    # below — that's the part that makes injection impossible.
    physical = {k: v[DATA_MODE] for k, v in TABLES.items()}
    sql = sql.format(**physical)
    _QUERY_LOG.append(_inline(sql, params))  # display copy of what actually ran

    conn = _connect()
    if DATA_MODE == "snapshot":
        cur = conn.execute(sql, list(params))
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    else:
        cur = conn.cursor()
        cur.execute(sql.replace("?", "%s"), params)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    if DEBUG_SQL:
        print(f"          [SQL/{DATA_MODE}] {sql[:100]} | params={params} "
              f"| {len(rows)} rows")
    return rows


# ---------------------------------------------------------------------------
# Typed fetch helpers. Each one is a single, reviewable query.
# ---------------------------------------------------------------------------

def find_accounts(text: str) -> list[dict]:
    """Fuzzy account lookup by name or exact ID."""
    return query(
        "SELECT * FROM {accounts} "
        "WHERE UPPER(COMPANY_NAME) LIKE UPPER(?) OR UPPER(ACCOUNT_ID) = UPPER(?) "
        "ORDER BY COMPANY_NAME",
        (f"%{text}%", text),
    )


def get_account(account_id: str) -> dict | None:
    rows = query("SELECT * FROM {accounts} WHERE ACCOUNT_ID = ?", (account_id,))
    return rows[0] if rows else None


def accounts_for_ae(ae_email: str) -> list[dict]:
    return query(
        "SELECT * FROM {accounts} WHERE OWNER_AE = ? ORDER BY COMPANY_NAME",
        (ae_email,),
    )


def opportunities_for(account_id: str) -> list[dict]:
    return query(
        "SELECT * FROM {opportunities} WHERE ACCOUNT_ID = ? ORDER BY CLOSE_DATE",
        (account_id,),
    )


def contacts_for(account_id: str) -> list[dict]:
    return query(
        "SELECT * FROM {contacts} WHERE ACCOUNT_ID = ? ORDER BY LAST_INTERACTION DESC",
        (account_id,),
    )


def activities_for(account_id: str, since: str | None = None,
                   contains: str | None = None, limit: int = 200) -> list[dict]:
    sql = "SELECT * FROM {activities} WHERE ACCOUNT_ID = ?"
    params: list = [account_id]
    if since:
        sql += " AND ACTIVITY_DATE >= CAST(? AS DATE)"
        params.append(since)
    if contains:
        sql += " AND (UPPER(SUMMARY) LIKE UPPER(?) OR UPPER(SUBJECT) LIKE UPPER(?))"
        params.extend([f"%{contains}%", f"%{contains}%"])
    sql += f" ORDER BY ACTIVITY_DATE DESC LIMIT {int(limit)}"
    return query(sql, tuple(params))


def usage_for(account_id: str) -> list[dict]:
    return query(
        "SELECT * FROM {usage} WHERE ACCOUNT_ID = ? ORDER BY MONTH",
        (account_id,),
    )


def tickets_for(account_id: str) -> list[dict]:
    return query(
        "SELECT * FROM {tickets} WHERE ACCOUNT_ID = ? ORDER BY CREATED_DATE DESC",
        (account_id,),
    )


OPEN_STAGES_SQL = "('Discovery','Qualification','Demo','Proposal','Negotiation')"


def open_opps_for_ae(ae_email: str, opp_type: str | None = None,
                     closing_within_days: int | None = None) -> list[dict]:
    """Every open deal in an AE's book, joined with the account name.
    Optional filters run in SQL so date windows and type filters are never
    left to the model's in-context judgment (it drops edge rows)."""
    sql = ("SELECT o.*, a.COMPANY_NAME, a.STATUS AS ACCOUNT_STATUS, a.SEGMENT "
           "FROM {opportunities} o JOIN {accounts} a ON o.ACCOUNT_ID = a.ACCOUNT_ID "
           f"WHERE a.OWNER_AE = ? AND o.STAGE IN {OPEN_STAGES_SQL}")
    params: list = [ae_email]
    if opp_type:
        sql += " AND UPPER(o.TYPE) = UPPER(?)"
        params.append(opp_type)
    if closing_within_days is not None:
        sql += " AND o.CLOSE_DATE <= CAST(? AS DATE)"
        params.append(closing_cutoff(closing_within_days))
    sql += " ORDER BY o.CLOSE_DATE"
    return query(sql, tuple(params))


def closing_cutoff(days: int) -> str:
    """AS_OF + N days, ISO string. Shared so the tool's method block states
    the exact same window the SQL used."""
    from datetime import date, timedelta

    return (date.fromisoformat(AS_OF) + timedelta(days=int(days))).isoformat()


def stats(scope: str, ae_email: str) -> dict:
    """All counts and sums the model is NOT allowed to compute itself.
    Everything here is SQL GROUP BY / SUM; the model only reports it.
    The 'method' block is machine-authored provenance the agent repeats
    verbatim and the UI renders as a method card."""
    if scope == "mine":
        acc_where, params = "WHERE OWNER_AE = ?", (ae_email,)
        scope_desc = f"your book only (accounts owned by {ae_email})"
    else:
        acc_where, params = "", ()
        scope_desc = "company-wide (all accounts, all AEs)"

    # f-strings below: {{accounts}} (doubled braces) survives as a literal
    # {accounts} placeholder for query()'s later .format(**physical) call;
    # {acc_where} (single braces) is substituted immediately, right here.
    sql_status = (f"SELECT STATUS, COUNT(*) AS N FROM {{accounts}} {acc_where} "
                  "GROUP BY STATUS ORDER BY N DESC")
    sql_segment = (f"SELECT SEGMENT, COUNT(*) AS N FROM {{accounts}} {acc_where} "
                   "GROUP BY SEGMENT ORDER BY N DESC")
    sql_owner = ("SELECT OWNER_AE, COUNT(*) AS N FROM {accounts} "
                 "GROUP BY OWNER_AE ORDER BY N DESC")
    opp_where = "WHERE o.STAGE IN " + OPEN_STAGES_SQL + (
        " AND a.OWNER_AE = ?" if scope == "mine" else "")
    sql_pipeline = ("SELECT o.TYPE, COUNT(*) AS N_DEALS, SUM(o.AMOUNT_EUR) AS TOTAL_EUR "
                    "FROM {opportunities} o JOIN {accounts} a "
                    f"ON o.ACCOUNT_ID = a.ACCOUNT_ID {opp_where} GROUP BY o.TYPE")

    reason_where = ("WHERE o.STAGE = ? " + ("AND a.OWNER_AE = ?" if scope == "mine" else ""))
    sql_reasons = ("SELECT o.WON_LOST_REASON, COUNT(*) AS N "
                   "FROM {opportunities} o JOIN {accounts} a "
                   f"ON o.ACCOUNT_ID = a.ACCOUNT_ID {reason_where} "
                   "GROUP BY o.WON_LOST_REASON ORDER BY N DESC")

    status_counts = query(sql_status, params)
    segment_counts = query(sql_segment, params)
    owner_counts = query(sql_owner, ())
    pipeline_by_type = query(sql_pipeline, params)
    loss_reasons = query(sql_reasons, ("Closed Lost",) + params)
    win_reasons = query(sql_reasons, ("Closed Won",) + params)
    total_open = sum(r["TOTAL_EUR"] or 0 for r in pipeline_by_type)
    total_deals = sum(r["N_DEALS"] for r in pipeline_by_type)
    total_accounts = sum(r["N"] for r in status_counts)

    return {
        "method": {
            "scope": scope_desc,
            "definitions": [
                "open deal = stage in Discovery, Qualification, Demo, "
                "Proposal or Negotiation",
            ],
            # SQL is attached generically in tools.execute_tool from the real
            # captured queries, so every tool shows the exact SQL it ran.
        },
        "total_accounts": total_accounts,
        "accounts_by_status": status_counts,
        "accounts_by_segment": segment_counts,
        "accounts_by_owner": owner_counts,
        "open_pipeline": {
            "total_eur": total_open,
            "total_deals": total_deals,
            "avg_deal_eur": round(total_open / total_deals, 2) if total_deals else 0,
            "by_type": pipeline_by_type,
        },
        "closed_lost_reasons": loss_reasons,
        "closed_won_reasons": win_reasons,
    }


def all_aes() -> list[str]:
    """Every distinct AE email that owns at least one account. Powers the
    sidebar's identity picker in app.py; there is no separate auth system."""
    rows = query("SELECT DISTINCT OWNER_AE FROM {accounts} ORDER BY OWNER_AE", ())
    return [r["OWNER_AE"] for r in rows]
