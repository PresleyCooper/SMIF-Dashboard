"""
smif/store.py — the canonical transaction ledger (SQLite) and its audit trail.

The Schwab export is treated as read-only seed data: it is loaded once into the
database and never written back to. Everything after that — a single manual
entry, a bulk import of a fresh export, an edit, a delete — happens here, and
every one of those writes leaves an audit row behind.

Two decisions worth stating:

  · DELETES ARE SOFT. A row is flagged, not removed, because the whole point of
    the audit trail is to answer "why did last month's number change" and a
    hard delete destroys exactly the evidence needed. Reads filter on the flag.

  · DE-DUPLICATION IS ON THE TRADE'S OWN IDENTITY — date, activity, symbol,
    quantity, amount — not on a row number or a file hash. Schwab exports
    overlap by design (you pull "year to date" every month), so the same trade
    legitimately arrives many times and must land once. Amount and quantity are
    rounded to the cent / 1e-6 share before comparison so a float round-trip
    through Excel cannot manufacture a "new" row.

    It matches on MULTIPLICITY, not membership: if the ledger already holds two
    rows with a key and an import carries three, one is new. The fund really
    does place two identical fills in a name on the same day — this export has
    three such pairs — and a plain "have I seen this key" test silently drops
    the second one and takes the share count with it.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date  TEXT    NOT NULL,
    activity    TEXT    NOT NULL,
    description TEXT,
    symbol      TEXT,
    quantity    REAL,
    amount      REAL,
    notes       TEXT,
    source      TEXT    NOT NULL DEFAULT 'manual',
    added_by    TEXT,
    added_at    TEXT,
    deleted     INTEGER NOT NULL DEFAULT 0,
    dedupe_key  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tx_dedupe ON transactions (dedupe_key);
CREATE INDEX IF NOT EXISTS ix_tx_date   ON transactions (trade_date);

CREATE TABLE IF NOT EXISTS audit (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    action  TEXT NOT NULL,
    tx_id   INTEGER,
    actor   TEXT,
    detail  TEXT
);

CREATE TABLE IF NOT EXISTS symbol_class (
    symbol   TEXT PRIMARY KEY,
    sleeve   TEXT NOT NULL,
    added_by TEXT,
    added_at TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

READ_COLUMNS = ["id", "trade_date", "activity", "description", "symbol",
                "quantity", "amount", "notes", "source", "added_by", "added_at"]


# ── plumbing ──────────────────────────────────────────────────────────────────

@contextmanager
def connect(path: Path | str | None = None):
    path = Path(path) if path is not None else config.LEDGER_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def dedupe_key(trade_date, activity, symbol, quantity, amount) -> str:
    """The trade's identity. See the module docstring for why these five."""
    d = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    sym = "" if symbol is None or pd.isna(symbol) else str(symbol).strip().upper()
    q = 0.0 if quantity is None or pd.isna(quantity) else round(float(quantity), 6)
    a = 0.0 if amount is None or pd.isna(amount) else round(float(amount), 2)
    return f"{d}|{str(activity).strip()}|{sym}|{q:.6f}|{a:.2f}"


def _row_values(row: dict, source: str, actor: str) -> tuple:
    return (
        pd.Timestamp(row["Trade Date"]).strftime("%Y-%m-%d"),
        str(row["Activity"]).strip(),
        None if pd.isna(row.get("Description")) else str(row["Description"]),
        None if pd.isna(row.get("Symbol")) else str(row["Symbol"]).strip().upper(),
        None if pd.isna(row.get("Quantity")) else float(row["Quantity"]),
        None if pd.isna(row.get("Amount")) else float(row["Amount"]),
        None if pd.isna(row.get("Notes")) else str(row["Notes"]),
        source, actor, _now(),
        dedupe_key(row["Trade Date"], row["Activity"], row.get("Symbol"),
                   row.get("Quantity"), row.get("Amount")),
    )


def _audit(con, action: str, tx_id: int | None, actor: str, detail: str = "") -> None:
    con.execute("INSERT INTO audit (ts, action, tx_id, actor, detail) VALUES (?,?,?,?,?)",
                (_now(), action, tx_id, actor, detail))


# ── reads ─────────────────────────────────────────────────────────────────────

def read_transactions(path=None, include_deleted: bool = False) -> pd.DataFrame:
    """The ledger in chronological order, in Schwab export column names."""
    with connect(path) as con:
        where = "" if include_deleted else "WHERE deleted = 0"
        df = pd.read_sql_query(
            f"SELECT {', '.join(READ_COLUMNS)}, deleted FROM transactions {where}", con)
    df = df.rename(columns={
        "trade_date": "Trade Date", "activity": "Activity",
        "description": "Description", "symbol": "Symbol",
        "quantity": "Quantity", "amount": "Amount", "notes": "Notes"})
    if df.empty:
        df = pd.DataFrame(columns=["id", *config.TX_COLUMNS,
                                   "source", "added_by", "added_at", "deleted"])
    df["Trade Date"] = pd.to_datetime(df["Trade Date"])
    # id breaks ties so the order is deterministic across runs — several rows
    # share a trade date and the engine replays them in sequence.
    return df.sort_values(["Trade Date", "id"]).reset_index(drop=True)


def existing_keys(path=None) -> Counter:
    """How many live rows the ledger holds per dedupe key."""
    with connect(path) as con:
        rows = con.execute(
            "SELECT dedupe_key FROM transactions WHERE deleted = 0").fetchall()
    return Counter(r["dedupe_key"] for r in rows)


def is_empty(path=None) -> bool:
    with connect(path) as con:
        n = con.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()["n"]
    return n == 0


def audit_log(path=None, limit: int = 500) -> pd.DataFrame:
    with connect(path) as con:
        return pd.read_sql_query(
            "SELECT ts, action, tx_id, actor, detail FROM audit "
            "ORDER BY id DESC LIMIT ?", con, params=(limit,))


def get_meta(key: str, path=None) -> str | None:
    with connect(path) as con:
        r = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return r["value"] if r else None


def set_meta(key: str, value: str, path=None) -> None:
    with connect(path) as con:
        con.execute("INSERT INTO meta (key, value) VALUES (?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, str(value)))


# ── writes ────────────────────────────────────────────────────────────────────

def add_transactions(rows: pd.DataFrame, *, source: str = "manual",
                     actor: str = "unknown", path=None) -> dict:
    """
    Insert rows that are not already in the ledger.

    Returns {"added": n, "skipped": n, "added_ids": [...], "skipped_keys": [...]}.
    Each incoming row consumes one of the ledger's existing copies of its key;
    only rows left over once those are used up are inserted. That is what lets
    a re-imported export land nothing while a genuinely repeated fill lands.
    """
    if rows is None or rows.empty:
        return {"added": 0, "skipped": 0, "added_ids": [], "skipped_keys": []}

    budget = existing_keys(path)
    added_ids, skipped = [], []
    with connect(path) as con:
        for _, r in rows.iterrows():
            vals = _row_values(r.to_dict(), source, actor)
            key = vals[-1]
            if budget.get(key, 0) > 0:
                budget[key] -= 1
                skipped.append(key)
                continue
            cur = con.execute(
                "INSERT INTO transactions (trade_date, activity, description, symbol,"
                " quantity, amount, notes, source, added_by, added_at, dedupe_key)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)", vals)
            added_ids.append(cur.lastrowid)
            _audit(con, "add", cur.lastrowid, actor, f"{source}: {key}")
    return {"added": len(added_ids), "skipped": len(skipped),
            "added_ids": added_ids, "skipped_keys": skipped}


def preview_import(rows: pd.DataFrame, path=None) -> pd.DataFrame:
    """
    What add_transactions would do, without doing it.

    The import page shows this before anything is written, so "47 rows, 3 of
    them new" is a statement the user can check rather than a result to undo.
    """
    if rows is None or rows.empty:
        return pd.DataFrame(columns=[*config.TX_COLUMNS, "status"])
    budget = existing_keys(path)
    status = []
    for _, r in rows.iterrows():
        key = dedupe_key(r["Trade Date"], r["Activity"], r.get("Symbol"),
                         r.get("Quantity"), r.get("Amount"))
        if budget.get(key, 0) > 0:
            budget[key] -= 1
            status.append("already in ledger")
        else:
            status.append("new")
    out = rows.copy()
    out["status"] = status
    return out


def update_transaction(tx_id: int, changes: dict, *, actor: str = "unknown",
                       path=None) -> None:
    """Edit one row. The audit detail records the before and after values."""
    field_map = {"Trade Date": "trade_date", "Activity": "activity",
                 "Description": "description", "Symbol": "symbol",
                 "Quantity": "quantity", "Amount": "amount", "Notes": "notes"}
    with connect(path) as con:
        before = con.execute("SELECT * FROM transactions WHERE id = ?",
                             (tx_id,)).fetchone()
        if before is None:
            raise KeyError(f"no transaction with id {tx_id}")
        merged = {k: before[v] for k, v in field_map.items()}
        merged["Trade Date"] = pd.Timestamp(merged["Trade Date"])
        for k, v in changes.items():
            if k not in field_map:
                raise KeyError(f"not an editable field: {k}")
            merged[k] = v
        sets, params = [], []
        for k, col in field_map.items():
            v = merged[k]
            if k == "Trade Date":
                v = pd.Timestamp(v).strftime("%Y-%m-%d")
            elif k == "Symbol" and v is not None and not pd.isna(v):
                v = str(v).strip().upper()
            elif v is not None and pd.isna(v):
                v = None
            sets.append(f"{col} = ?")
            params.append(v)
        sets.append("dedupe_key = ?")
        params.append(dedupe_key(merged["Trade Date"], merged["Activity"],
                                 merged["Symbol"], merged["Quantity"],
                                 merged["Amount"]))
        params.append(tx_id)
        con.execute(f"UPDATE transactions SET {', '.join(sets)} WHERE id = ?", params)
        was = {k: before[v] for k, v in field_map.items()}
        _audit(con, "edit", tx_id, actor,
               "; ".join(f"{k}: {was[k]!r} -> {changes[k]!r}" for k in changes))


def delete_transaction(tx_id: int, *, actor: str = "unknown", reason: str = "",
                       path=None) -> None:
    with connect(path) as con:
        row = con.execute("SELECT dedupe_key FROM transactions WHERE id = ?",
                          (tx_id,)).fetchone()
        if row is None:
            raise KeyError(f"no transaction with id {tx_id}")
        con.execute("UPDATE transactions SET deleted = 1 WHERE id = ?", (tx_id,))
        _audit(con, "delete", tx_id, actor, f"{row['dedupe_key']}; {reason}".strip("; "))


def restore_transaction(tx_id: int, *, actor: str = "unknown", path=None) -> None:
    with connect(path) as con:
        con.execute("UPDATE transactions SET deleted = 0 WHERE id = ?", (tx_id,))
        _audit(con, "restore", tx_id, actor, "")


# ── symbol classification ─────────────────────────────────────────────────────

def known_symbols(path=None) -> set[str]:
    df = read_transactions(path)
    return {s for s in df["Symbol"].dropna().unique()}


def symbol_classes(path=None) -> dict[str, str]:
    """User-declared sleeves for symbols the built-in config does not cover."""
    with connect(path) as con:
        rows = con.execute("SELECT symbol, sleeve FROM symbol_class").fetchall()
    return {r["symbol"]: r["sleeve"] for r in rows}


def classify_symbol(symbol: str, sleeve: str, *, actor: str = "unknown",
                    path=None) -> None:
    if sleeve not in config.SLEEVES:
        raise ValueError(f"sleeve must be one of {config.SLEEVES}, got {sleeve!r}")
    sym = str(symbol).strip().upper()
    with connect(path) as con:
        con.execute("INSERT INTO symbol_class (symbol, sleeve, added_by, added_at)"
                    " VALUES (?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET"
                    " sleeve = excluded.sleeve, added_by = excluded.added_by,"
                    " added_at = excluded.added_at",
                    (sym, sleeve, actor, _now()))
        _audit(con, "classify", None, actor, f"{sym} -> {sleeve}")


# ── seeding ───────────────────────────────────────────────────────────────────

def read_export(path: Path | str) -> pd.DataFrame:
    """
    Read a Schwab export (.xlsx or .csv) into the canonical column shape.

    The export is newest-first; order is irrelevant here because the engine
    sorts by date, but the columns must be exactly the seven the app knows.
    """
    path = Path(path)
    if path.suffix.lower() in (".csv", ".txt"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in config.TX_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing column(s): {', '.join(missing)}")
    df = df[config.TX_COLUMNS].copy()
    df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
    bad = df["Trade Date"].isna()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} row(s) have an unreadable Trade Date")
    df["Activity"] = df["Activity"].astype(str).str.strip()
    unknown = sorted(set(df["Activity"]) - set(config.ACTIVITIES))
    if unknown:
        raise ValueError("unrecognised Activity value(s): " + ", ".join(unknown))
    return df


def seed_from_export(path: Path | str | None = None, *, actor: str = "seed",
                     db_path=None) -> dict:
    """Load the original export into an empty ledger. A no-op once populated."""
    path = Path(path) if path else config.SEED_TRANSACTIONS
    if not is_empty(db_path):
        return {"added": 0, "skipped": 0, "note": "ledger already populated"}
    df = read_export(path)
    out = add_transactions(df, source=f"seed:{Path(path).name}", actor=actor,
                           path=db_path)
    set_meta("seeded_from", Path(path).name, db_path)
    return out
