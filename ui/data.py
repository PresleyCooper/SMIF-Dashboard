"""
ui/data.py — the only place the UI talks to the engine, and the only place
anything is cached.

The engine is deliberately free of Streamlit, so caching is this layer's job.
Everything is keyed on the modification times of the two files that can change
an answer — the ledger database and the price cache — plus the as-of date.
That means saving a transaction invalidates every derived number immediately,
with nothing to remember to clear by hand, and a TTL that could serve a stale
return for a few minutes never exists.

`st.cache_resource` rather than `cache_data` for the Analytics object: it holds
a dozen DataFrames spanning 4,292 days and pickling it on every page switch
would cost more than rebuilding it. Nothing mutates it, so sharing one instance
across sessions is safe.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from smif import config, engine, returns as returns_mod, store, validate


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _fingerprint() -> tuple[float, float]:
    """
    What has to change for a cached number to be wrong.

    Passed as the FIRST POSITIONAL ARGUMENT of every cached function, and
    never named with a leading underscore: Streamlit excludes underscore-
    prefixed parameters from the cache key by design, so calling this `_fp`
    silently disables the whole invalidation scheme and the app keeps serving
    yesterday's returns after a transaction is saved.
    """
    return _mtime(config.LEDGER_DB), _mtime(config.PRICE_CACHE)


# ── readiness ─────────────────────────────────────────────────────────────────

def status() -> dict:
    """
    Whether the app has what it needs, and what to say if not.

    Checked before any page renders, because every failure here has a different
    fix and "something went wrong" would hide which one applies.
    """
    config.ensure_dirs()
    have_tx = config.SEED_TRANSACTIONS.exists()
    have_wb = config.SEED_WORKBOOK.exists()
    ledger_rows = 0
    if config.LEDGER_DB.exists():
        try:
            ledger_rows = len(store.read_transactions())
        except Exception:
            ledger_rows = 0
    return {
        "ready": ledger_rows > 0 and config.PRICE_CACHE.exists(),
        "ledger_rows": ledger_rows,
        "has_prices": config.PRICE_CACHE.exists(),
        "has_seed_transactions": have_tx,
        "has_workbook": have_wb,
    }


def require_data() -> dict:
    """Render the blocking explanation and stop, or return the status."""
    s = status()
    if s["ready"]:
        return s

    st.error("The fund history has not been built on this machine yet.")
    if s["ledger_rows"] == 0:
        st.markdown(
            "The transaction ledger is empty. Seed it from the Schwab export:\n\n"
            "```\npython scripts/bootstrap.py "
            '"Portfolio Transaction History.xlsx" "SMIF Portfolio.xlsx"\n```')
    elif not s["has_prices"]:
        st.markdown(
            f"The ledger holds **{s['ledger_rows']:,} transactions** but no prices "
            "have been downloaded. The first pull takes a few minutes:\n\n"
            "```\npython scripts/bootstrap.py "
            '"Portfolio Transaction History.xlsx" "SMIF Portfolio.xlsx"\n```')
    st.stop()


# ── the analytics object ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Rebuilding the fund's daily history…")
def _load(fingerprint: tuple[float, float], end: str | None) -> engine.Analytics:
    return engine.load(end=pd.Timestamp(end) if end else None)


def analytics(end=None) -> engine.Analytics:
    key = str(pd.Timestamp(end).date()) if end is not None else None
    return _load(_fingerprint(), key)


def refresh() -> None:
    """Drop every cached figure. Called after a save or a price refresh."""
    _load.clear()
    _period_table.clear()
    _calendar_table.clear()
    _validation.clear()


# ── derived tables ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _period_table(fingerprint: tuple[float, float], which: str,
                  end: str | None) -> pd.DataFrame:
    a = analytics(end)
    track = a.tracks.get(which)
    if track is None:
        return pd.DataFrame()
    return returns_mod.period_table(track, end=a.asof)


def period_table(which: str, end=None) -> pd.DataFrame:
    key = str(pd.Timestamp(end).date()) if end is not None else None
    return _period_table(_fingerprint(), which, key)


@st.cache_data(show_spinner=False)
def _calendar_table(fingerprint: tuple[float, float], which: str,
                    end: str | None) -> pd.DataFrame:
    a = analytics(end)
    track = a.tracks.get(which)
    if track is None:
        return pd.DataFrame()
    return returns_mod.calendar_years(track, end=a.asof)


def calendar_table(which: str, end=None) -> pd.DataFrame:
    key = str(pd.Timestamp(end).date()) if end is not None else None
    return _calendar_table(_fingerprint(), which, key)


@st.cache_data(show_spinner="Reconciling against the custodian…")
def _validation(fingerprint: tuple[float, float], end: str | None) -> dict:
    return analytics(end).validation()


def validation(end=None) -> dict:
    key = str(pd.Timestamp(end).date()) if end is not None else None
    return _validation(_fingerprint(), key)


def cash_gap(end=None) -> dict | None:
    """
    Just the cash reconciliation, without the full report.

    Separate because the Overview wants to surface the gap on every load, and
    the full report walks 500-odd fills to check prices — far too much work for
    one line of text.
    """
    a = analytics(end)
    if a.subtotals is None:
        return None
    try:
        return validate.check_cash(a.book, a.subtotals)
    except (KeyError, IndexError):
        return None
