"""
smif/engine.py — one call that turns the ledger into every number the app shows.

Deliberately the only module that knows the *order* of operations. Everything
else takes its inputs as arguments, which is what makes the pipeline testable:
a test can hand build_book a three-row ledger and a fake price store without
touching the database or the network.

No Streamlit import anywhere in this package. Caching is the UI's problem, and
the UI can cache on (ledger mtime, price-cache mtime, end date) because those
three are the only inputs that can change an answer.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config, ledger, nav, prices as price_mod, returns, store as store_mod
from . import validate, workbook


@dataclass
class Analytics:
    tx: pd.DataFrame
    book: nav.Book
    tracks: dict[str, returns.Track]
    store: price_mod.PriceStore
    holdings: pd.DataFrame | None
    custodian: pd.DataFrame | None
    subtotals: pd.Series | None
    custodian_as_of: pd.Timestamp | None
    price_report: dict

    @property
    def asof(self) -> pd.Timestamp:
        return self.book.asof

    def validation(self) -> dict:
        if self.holdings is None or self.subtotals is None:
            return {}
        return validate.report(self.book, self.tx, self.holdings, self.subtotals)


def install_seed(transactions: Path | str | None = None,
                 portfolio: Path | str | None = None) -> dict:
    """
    Copy the two source files into data/seed/ under their canonical names.

    Copying rather than reading in place is what keeps the promise that the
    original export is never modified: the app only ever opens its own copy,
    and the file the treasurer downloaded from Schwab stays where it is.
    """
    config.ensure_dirs()
    out = {}
    for src, dst in ((transactions, config.SEED_TRANSACTIONS),
                     (portfolio, config.SEED_WORKBOOK)):
        if src is None:
            continue
        src = Path(src)
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        out[dst.name] = str(dst)
    return out


def last_complete_session(store: price_mod.PriceStore) -> pd.Timestamp | None:
    """
    The most recent close the whole book can be valued on.

    Yahoo serves a live, moving quote for the current session. Valuing the fund
    on it produces a number that changes every refresh and cannot be reconciled
    against anything, so the as-of date is the last session in the cache that is
    strictly before today.
    """
    days = price_mod.trading_days(store)
    if len(days) == 0:
        return None
    past = days[days < config.today()]
    return past[-1] if len(past) else days[-1]


def refresh_prices(tx: pd.DataFrame | None = None, *, end=None, force: bool = False,
                   progress=None, store: price_mod.PriceStore | None = None) -> dict:
    """Fetch whatever the current ledger needs and is not already cached."""
    store = store or price_mod.PriceStore()
    tx = tx if tx is not None else store_mod.read_transactions()
    events = ledger.build_events(tx)
    symbols = ledger.traded_symbols(events)
    return store.refresh(price_mod.required_tickers(symbols), end=end,
                         force=force, progress=progress)


def load(*, end=None, db_path=None, refresh: bool = False,
         workbook_path=None) -> Analytics:
    """
    Build the whole analytics set.

    `end` pins the as-of date. Left out, it is the last complete session in the
    price cache — never today's moving quote.
    """
    config.ensure_dirs()
    tx = store_mod.read_transactions(db_path)
    if tx.empty:
        raise ValueError(
            f"the ledger is empty — seed it from {config.SEED_TRANSACTIONS.name} first")

    store = price_mod.PriceStore()
    report = {}
    if refresh or not store.path.exists():
        report = refresh_prices(tx, end=end, store=store)

    if end is None:
        end = last_complete_session(store)
    end = pd.Timestamp(end)

    overrides = store_mod.symbol_classes(db_path)
    book = nav.build_book(tx, store, end=end, overrides=overrides)
    tracks = returns.build_tracks(book, store, overrides=overrides)

    holdings = custodian = subtotals = as_of = None
    try:
        wb = workbook_path or config.SEED_WORKBOOK
        if Path(wb).exists():
            holdings = workbook.read_holdings(wb)
            as_of, custodian, subtotals = workbook.read_custodian_snapshot(wb)
    except Exception as exc:                      # the workbook is optional
        report.setdefault("workbook_error", str(exc))

    return Analytics(tx=tx, book=book, tracks=tracks, store=store,
                     holdings=holdings, custodian=custodian, subtotals=subtotals,
                     custodian_as_of=as_of, price_report=report)


def regression_table(analytics: Analytics, end=None) -> pd.DataFrame:
    """
    The table the build is checked against: equity and total portfolio against
    SPY over every standard period. Used by the tests and shown on the
    Methodology page, so the published numbers and the tested numbers are the
    same numbers.
    """
    end = pd.Timestamp(end) if end is not None else analytics.asof
    eq = returns.period_table(analytics.tracks["equity"], end=end)
    tot = returns.period_table(analytics.tracks["total"], end=end)
    out = pd.DataFrame({
        "period": eq["period"],
        "equity": eq["cum"] * 100,
        "spy_vs_equity": eq["bench_cum"] * 100,
        "total": tot["cum"] * 100,
        "spy_vs_total": tot["bench_cum"] * 100,
        "equity_base": eq["start"].dt.strftime("%Y-%m-%d"),
        "total_base": tot["start"].dt.strftime("%Y-%m-%d"),
    })
    return out
