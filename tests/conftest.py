"""
Shared fixtures.

Two kinds of test live in here and they are kept strictly apart:

  UNIT tests build their own three-row ledger and a fake price store. They need
  no network, no database and no workbook, and they are the ones that say
  whether a RULE is right.

  REGRESSION tests load the real ledger and the real price cache and check the
  published numbers. They need the cache to exist, so they skip (rather than
  fail) on a fresh checkout — a missing cache is a setup state, not a defect.
  Their as-of date is PINNED to 2026-11-04 so that tomorrow's Yahoo pull cannot
  silently move a number the repository claims.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smif import config, engine, prices as price_mod, store as store_mod  # noqa: E402

REGRESSION_END = pd.Timestamp("2026-11-04")


@pytest.fixture(scope="session")
def have_cache() -> bool:
    return config.PRICE_CACHE.exists() and config.LEDGER_DB.exists()


@pytest.fixture(scope="session")
def analytics(have_cache):
    if not have_cache:
        pytest.skip("no local price cache / ledger — run scripts/bootstrap.py first")
    return engine.load(end=REGRESSION_END)


@pytest.fixture(scope="session")
def validation(analytics):
    v = analytics.validation()
    if not v:
        pytest.skip("workbook not available, so nothing to reconcile against")
    return v


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    return tmp_path / "ledger.sqlite"


def tx_frame(rows: list[dict]) -> pd.DataFrame:
    """A ledger from a list of partial dicts, with the dull fields defaulted."""
    out = []
    for i, r in enumerate(rows, 1):
        out.append({
            "id": r.get("id", i),
            "Trade Date": pd.Timestamp(r["date"]),
            "Activity": r["activity"],
            "Description": r.get("description", r.get("symbol") or "CASH"),
            "Symbol": r.get("symbol"),
            "Quantity": r.get("qty"),
            "Amount": r.get("amount"),
            "Notes": r.get("notes", ""),
        })
    return pd.DataFrame(out)


class FakeStore(price_mod.PriceStore):
    """
    A price store backed by a dict instead of parquet.

    It subclasses the real one so the code under test cannot tell the
    difference, and so a change to the store's interface breaks the fakes too
    rather than letting the unit tests drift away from reality.
    """

    def __init__(self, series: dict[str, pd.DataFrame]):
        self._series = series
        self.path = Path("memory://fake")

    @property
    def frame(self):
        parts = []
        for t, df in self._series.items():
            d = df.reset_index().rename(columns={"index": "date"})
            d["ticker"] = t
            parts.append(d)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=price_mod.CACHE_COLUMNS)

    def tickers(self):
        return set(self._series)

    def series(self, ticker):
        return self._series.get(ticker, pd.DataFrame(
            columns=["close", "adj_close", "split"]))

    def refresh(self, *a, **k):            # never touch the network in a test
        return {"fetched": [], "failed": [], "missing": [], "asof": None}


def flat_series(days: pd.DatetimeIndex, price: float,
                splits: dict | None = None) -> pd.DataFrame:
    s = pd.Series(0.0, index=days)
    for d, f in (splits or {}).items():
        s.loc[pd.Timestamp(d)] = f
    return pd.DataFrame({"close": float(price), "adj_close": float(price),
                         "split": s}, index=days)


def ramp_series(days: pd.DatetimeIndex, start: float, step: float) -> pd.DataFrame:
    px = start + step * pd.Series(range(len(days)), index=days, dtype=float)
    return pd.DataFrame({"close": px, "adj_close": px,
                         "split": pd.Series(0.0, index=days)}, index=days)


@pytest.fixture
def calendar() -> pd.DatetimeIndex:
    """Ten business days from 2020-01-02 — enough to open and move a sleeve."""
    return pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=10))


@pytest.fixture
def fake_store(calendar):
    """SPY sets the trading calendar; the rest are flat so moves are intentional."""
    return FakeStore({
        "SPY": ramp_series(calendar, 300.0, 1.0),
        "AGG": flat_series(calendar, 100.0),
        "AAA": flat_series(calendar, 50.0),
        "BBB": flat_series(calendar, 20.0),
        "IEF": flat_series(calendar, 95.0),
        "NEWBOND": flat_series(calendar, 10.0),
    })
