"""
The price cache's refresh behaviour — specifically, not keeping a partial bar.

Yahoo serves a session in progress as if its running quote were a close. Two
rules keep that out of the fund's history, and both are easy to regress:
today's bar is never stored at all, and a refresh re-fetches the last day it
already has so that yesterday's partial bar is overwritten by the settled
close rather than frozen for ever.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from smif import config, prices


class FakeYF:
    """Stands in for the yfinance module, recording what it was asked for."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls: list[dict] = []

    def download(self, tickers, start=None, end=None, **kw):
        self.calls.append({"tickers": tickers, "start": start, "end": end})
        cols = pd.MultiIndex.from_product(
            [["AAA"], ["Close", "Adj Close", "Stock Splits"]])
        out = pd.DataFrame(self.frame.to_numpy(), index=self.frame.index,
                           columns=cols)
        return out


def bars(dates, closes) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.DataFrame({"Close": closes, "Adj Close": closes,
                         "Stock Splits": [0.0] * len(closes)}, index=idx)


@pytest.fixture
def fake_yf(monkeypatch):
    def install(frame):
        fake = FakeYF(frame)
        module = types.ModuleType("yfinance")
        module.download = fake.download
        monkeypatch.setitem(sys.modules, "yfinance", module)
        return fake
    return install


@pytest.fixture
def store(tmp_path):
    return prices.PriceStore(tmp_path / "prices.parquet")


def test_todays_bar_is_never_stored(fake_yf, store, monkeypatch):
    """A session in progress has no close, and a price that changes on every
    refresh cannot be reconciled against anything."""
    monkeypatch.setattr(config, "today", lambda: pd.Timestamp("2026-11-06"))
    fake_yf(bars(["2026-11-04", "2026-11-05", "2026-11-06"], [100.0, 101.0, 95.0]))

    store.refresh(["AAA"], end=pd.Timestamp("2026-11-06"))
    cached = store.series("AAA")

    assert list(cached.index.strftime("%Y-%m-%d")) == ["2026-11-04", "2026-11-05"]
    assert 95.0 not in set(cached["close"])


def test_a_refresh_re_fetches_the_last_cached_day(fake_yf, store, monkeypatch):
    """Otherwise a bar written during market hours is frozen as written."""
    monkeypatch.setattr(config, "today", lambda: pd.Timestamp("2026-11-06"))
    fake = fake_yf(bars(["2026-11-04", "2026-11-05"], [100.0, 101.0]))
    store.refresh(["AAA"], end=pd.Timestamp("2026-11-05"))

    fake = fake_yf(bars(["2026-11-05"], [107.0]))
    store.refresh(["AAA"], end=pd.Timestamp("2026-11-05"))

    assert fake.calls[0]["start"] == "2026-11-05"      # overlaps, not 11-06


def test_a_partial_bar_is_replaced_by_the_settled_close(fake_yf, store,
                                                        monkeypatch):
    monkeypatch.setattr(config, "today", lambda: pd.Timestamp("2026-11-05"))
    fake_yf(bars(["2026-11-03", "2026-11-04"], [100.0, 101.0]))
    store.refresh(["AAA"], end=pd.Timestamp("2026-11-04"))
    assert float(store.series("AAA")["close"].iloc[-1]) == 101.0

    # the next day, the same session settles two dollars higher
    monkeypatch.setattr(config, "today", lambda: pd.Timestamp("2026-11-06"))
    fake_yf(bars(["2026-11-04", "2026-11-05"], [103.0, 104.0]))
    store.refresh(["AAA"], end=pd.Timestamp("2026-11-05"))

    settled = store.series("AAA")
    assert float(settled.loc[pd.Timestamp("2026-11-04"), "close"]) == 103.0
    assert float(settled.loc[pd.Timestamp("2026-11-05"), "close"]) == 104.0
    assert len(settled) == 3                            # no duplicate rows


def test_an_up_to_date_ticker_is_not_re_fetched_past_the_end(fake_yf, store,
                                                             monkeypatch):
    monkeypatch.setattr(config, "today", lambda: pd.Timestamp("2026-11-06"))
    fake_yf(bars(["2026-11-04", "2026-11-05"], [100.0, 101.0]))
    store.refresh(["AAA"], end=pd.Timestamp("2026-11-05"))

    fake = fake_yf(bars(["2026-11-05"], [101.0]))
    store.refresh(["AAA"], end=pd.Timestamp("2026-11-04"))
    assert fake.calls == []        # already has everything up to the end date


def test_the_cache_holds_one_row_per_ticker_and_date(fake_yf, store, monkeypatch):
    monkeypatch.setattr(config, "today", lambda: pd.Timestamp("2026-11-06"))
    frame = bars(["2026-11-04", "2026-11-05"], [100.0, 101.0])
    for _ in range(3):
        fake_yf(frame)
        store.refresh(["AAA"], end=pd.Timestamp("2026-11-05"))
    assert not store.frame.duplicated(subset=["ticker", "date"]).any()
    assert len(store.series("AAA")) == 2
