"""
smif/prices.py — real historical closes, cached locally.

Two things make this harder than "download the close".

1. YAHOO'S CLOSE IS SPLIT-ADJUSTED, THE LEDGER'S SHARES ARE NOT. The fund held
   116 actual Apple shares in 2014, not 116 of today's post-split shares, so a
   split-adjusted price multiplied by a real share count understates the
   position by the split factor. The real close is recovered as

       close(t) * product of every split factor dated strictly after t

   which is general: it needs no list of known splits and no special cases.
   Yahoo also records some spin-offs as fractional splits (RTX carries a 1.589
   "split" on 2020-04-03 for the Carrier and Otis separation), and the same
   arithmetic turns today's RTX close back into the UTX price the fund actually
   owned. That is a feature of doing it generally rather than by hand.

2. SOME NAMES CANNOT BE DOWNLOADED AT ALL. A ticker that was acquired, wound up
   or reissued to a different company either has no series or — worse — has
   somebody else's. Those are listed in config.NO_YAHOO_DATA and priced by
   log-linear interpolation between the fund's own fills. Log-linear because a
   straight line in price space implies a changing growth rate and can cross
   zero; a straight line in log space is a constant compound rate, which is the
   honest shape of "we don't know what happened in between". Every position
   priced this way is flagged, and the Methodology page lists them.

The cache is a single long-format parquet keyed on (ticker, date). A refresh
asks each ticker only for the days after its last cached close, so the second
run of the day costs one small request per name instead of seventeen years.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, ledger

CACHE_COLUMNS = ["ticker", "date", "close", "adj_close", "split"]
CHUNK = 20              # tickers per yfinance call
CHUNK_PAUSE = 1.5       # seconds between calls — Yahoo throttles without warning


# ── ticker mapping ────────────────────────────────────────────────────────────

def yahoo_ticker(symbol: str) -> str | None:
    """
    The Yahoo ticker whose series carries this ledger symbol's history.

    None means "do not ask Yahoo": the money market (a constant $1.00), the
    synthetic Randgold position (priced off Barrick), and the names on the
    no-data list.
    """
    s = str(symbol).strip().upper()
    if s in config.CASH_LIKE or s == config.RANDGOLD_SYMBOL:
        return None
    if s in config.NO_YAHOO_DATA:
        return None
    if s in config.YAHOO_MAP:
        return config.YAHOO_MAP[s]
    return s.replace(".", "-")          # BRK.B -> BRK-B


def required_tickers(symbols) -> list[str]:
    """Yahoo tickers needed to price these symbols, benchmarks included."""
    want = set()
    for s in symbols:
        y = yahoo_ticker(s)
        if y:
            want.add(y)
        if str(s).strip().upper() == config.RANDGOLD_SYMBOL:
            want.add(config.RANDGOLD_PROXY)
    want |= {b for b in config.BENCHMARKS.values() if b}
    return sorted(want)


# ── the cache ─────────────────────────────────────────────────────────────────

class PriceStore:
    """Long-format parquet cache of (ticker, date) -> close / adj close / split."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else config.PRICE_CACHE
        self._df: pd.DataFrame | None = None

    # -- reading ---------------------------------------------------------------
    @property
    def frame(self) -> pd.DataFrame:
        if self._df is None:
            if self.path.exists():
                df = pd.read_parquet(self.path)
                df["date"] = pd.to_datetime(df["date"])
                self._df = df[CACHE_COLUMNS]
            else:
                self._df = pd.DataFrame(columns=CACHE_COLUMNS)
                self._df["date"] = pd.to_datetime(self._df["date"])
        return self._df

    def tickers(self) -> set[str]:
        return set(self.frame["ticker"].unique())

    def last_date(self, ticker: str) -> pd.Timestamp | None:
        s = self.frame[self.frame["ticker"] == ticker]
        return None if s.empty else pd.Timestamp(s["date"].max())

    def series(self, ticker: str) -> pd.DataFrame:
        """close / adj_close / split for one ticker, date-indexed ascending."""
        s = self.frame[self.frame["ticker"] == ticker]
        if s.empty:
            return pd.DataFrame(columns=["close", "adj_close", "split"])
        return (s.set_index("date")[["close", "adj_close", "split"]]
                .sort_index())

    def real_close(self, ticker: str) -> pd.Series:
        """The un-split-adjusted close — what a share actually traded at."""
        return unadjust(self.series(ticker))

    def adj_close(self, ticker: str) -> pd.Series:
        s = self.series(ticker)
        return s["adj_close"] if not s.empty else pd.Series(dtype=float)

    def coverage(self) -> pd.DataFrame:
        if self.frame.empty:
            return pd.DataFrame(columns=["ticker", "first", "last", "rows"])
        g = self.frame.groupby("ticker")["date"]
        return pd.DataFrame({"first": g.min(), "last": g.max(),
                             "rows": g.count()}).reset_index()

    # -- writing ---------------------------------------------------------------
    def _save(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df = (df.dropna(subset=["close"])
                .drop_duplicates(subset=["ticker", "date"], keep="last")
                .sort_values(["ticker", "date"])
                .reset_index(drop=True))
        df.to_parquet(self.path, index=False)
        self._df = df[CACHE_COLUMNS]

    def refresh(self, tickers, *, end=None, force: bool = False,
                progress=None) -> dict:
        """
        Top the cache up to `end`.

        force=True re-downloads the whole history, which is what to do after a
        ticker map changes or a split lands late: the un-adjust factor depends
        on every split in the series, so a stale tail is not a local error.
        """
        end = pd.Timestamp(end) if end is not None else config.today()
        end_excl = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        tickers = sorted(set(tickers))
        have = self.frame
        pieces, fetched, failed = [have], [], []

        todo = []
        for t in tickers:
            last = None if force else self.last_date(t)
            if last is not None and last >= end:
                continue
            start = (config.PRICE_START if last is None
                     else (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
            todo.append((t, start))

        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            if progress:
                progress(i, len(todo), chunk[0][0])
            # One start date per call, so group by it; in practice the cache is
            # either empty (all the same start) or uniformly one day behind.
            by_start: dict[str, list[str]] = {}
            for t, s in chunk:
                by_start.setdefault(s, []).append(t)
            for start, names in by_start.items():
                got, bad = _download(names, start, end_excl)
                for t, df in got.items():
                    df = df.copy()
                    df["ticker"] = t
                    pieces.append(df[CACHE_COLUMNS])
                    fetched.append(t)
                failed.extend(bad)
            time.sleep(CHUNK_PAUSE)

        pieces = [p for p in pieces if not p.empty]
        if len(pieces) > 1 or (pieces and have.empty):
            self._save(pd.concat(pieces, ignore_index=True))
        missing = [t for t in tickers if t not in self.tickers()]
        return {"fetched": sorted(set(fetched)), "failed": sorted(set(failed)),
                "missing": missing, "asof": str(end.date())}


def _download(tickers: list[str], start: str, end: str) -> tuple[dict, list]:
    """One yfinance call. Returns ({ticker: frame}, [tickers that gave nothing])."""
    import yfinance as yf
    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=False,
                          actions=True, progress=False, group_by="ticker",
                          threads=True)
    except Exception:
        return {}, list(tickers)
    if raw is None or raw.empty:
        return {}, list(tickers)
    out, bad = {}, []
    for t in tickers:
        try:
            x = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = pd.DataFrame({
                "date": pd.DatetimeIndex(x.index).tz_localize(None),
                "close": x["Close"].to_numpy(dtype=float),
                "adj_close": x["Adj Close"].to_numpy(dtype=float),
                "split": x.get("Stock Splits",
                               pd.Series(0.0, index=x.index)).to_numpy(dtype=float),
            }).dropna(subset=["close"])
            if df.empty:
                bad.append(t)
            else:
                out[t] = df
        except Exception:
            bad.append(t)
    return out, bad


# ── the un-adjust ─────────────────────────────────────────────────────────────

def unadjust(series: pd.DataFrame) -> pd.Series:
    """
    close * (product of split factors dated strictly after each date).

    `shift(-1)` on a reverse cumulative product is what makes it "strictly
    after": on the ex-split date Yahoo's close is already post-split, so that
    day's own factor must not be applied again.
    """
    if series is None or series.empty:
        return pd.Series(dtype=float)
    sp = series["split"].replace(0, 1).fillna(1)
    factor = sp[::-1].cumprod()[::-1].shift(-1).fillna(1)
    return (series["close"] * factor).rename("close")


def split_factor_table(series: pd.DataFrame) -> pd.DataFrame:
    """Every split in a series — shown on the data-quality page."""
    if series is None or series.empty:
        return pd.DataFrame(columns=["date", "split"])
    sp = series["split"].fillna(0)
    sp = sp[(sp != 0) & (sp != 1)]
    return sp.rename("split").reset_index()


# ── the price matrix ──────────────────────────────────────────────────────────

def trading_days(store: PriceStore, start=None, end=None,
                 calendar: str = "SPY") -> pd.DatetimeIndex:
    """
    The trading calendar, taken from the benchmark's own series.

    Using SPY rather than a holiday library means the calendar can never
    disagree with the prices: if a day has no SPY close it is not a day the
    portfolio can be valued on.
    """
    idx = pd.DatetimeIndex(store.series(calendar).index)
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    return idx


def interpolated_series(anchors: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Log-linear interpolation through known fills, held flat at both ends."""
    if anchors.empty:
        return pd.Series(np.nan, index=days)
    a = anchors.sort_index()
    a = a[a > 0]
    if a.empty:
        return pd.Series(np.nan, index=days)
    a.index = ledger.to_trading_day(a.index, days)
    a = a.groupby(level=0).mean()
    logp = (np.log(a).reindex(days)
            .interpolate(method="time", limit_area="inside")
            .ffill().bfill())
    return np.exp(logp)


def _anchor_set(tx: pd.DataFrame, symbol: str) -> pd.Series:
    """Fill-price anchors for a no-Yahoo-data name, with the known fix-ups."""
    a = ledger.fill_anchors(tx, symbol)
    if symbol in config.DROP_ZERO_ANCHORS:
        a = a[a > 0]                      # ITT's final $0 sale is a write-off
    if symbol == "MNK" and not a.empty:
        # Mallinckrodt arrived as the Questcor consideration, before its first
        # fill: back-date the earliest known price to the exchange date so the
        # position is not valued by a flat extrapolation from months later.
        a = pd.concat([pd.Series({config.QCOR_EXIT["date"]: a.iloc[0]}), a])
        a = a.groupby(level=0).mean().sort_index()
    if symbol == "QCOR":
        mnk = ledger.fill_anchors(tx, "MNK")
        if not mnk.empty:
            q = config.QCOR_EXIT
            a = a.copy()
            a.loc[q["date"]] = (q["cash"] + q["mnk_shares"] * mnk.iloc[0]) / q["qcor_shares"]
            a = a.sort_index()
    return a


def build_price_matrix(symbols, days: pd.DatetimeIndex, tx: pd.DataFrame,
                       store: PriceStore,
                       held: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """
    A price for every (symbol, trading day).

    `held` is the share-count matrix; it is used only to decide whether a gap
    before a ticker's first Yahoo print actually matters. Returns the matrix and
    a {symbol: note} dict of every non-obvious pricing decision, which the UI
    surfaces so an interpolated or back-filled valuation is never silent.
    """
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}

    for s in sorted(set(symbols)):
        s = str(s).strip().upper()
        if s == "CASH":
            continue
        if s in config.MONEY_MARKET:
            cols[s] = pd.Series(1.0, index=days)
            notes[s] = "money market, held at $1.00 by construction"
            continue
        if s == config.RANDGOLD_SYMBOL:
            base = store.real_close(config.RANDGOLD_PROXY).reindex(days).ffill().bfill()
            cols[s] = base * config.RANDGOLD_RATIO
            notes[s] = (f"Randgold, priced as {config.RANDGOLD_PROXY} x "
                        f"{config.RANDGOLD_RATIO:.4f} (the 263.5-for-43 Barrick "
                        f"exchange of 2019-01-02)")
            continue
        if s in config.NO_YAHOO_DATA:
            a = _anchor_set(tx, s)
            cols[s] = interpolated_series(a, days)
            notes[s] = (f"no Yahoo history — log-linear interpolation between "
                        f"{len(a)} of the fund's own fills")
            continue

        y = yahoo_ticker(s)
        p = store.real_close(y).reindex(days).ffill()
        if p.isna().all():
            a = _anchor_set(tx, s)
            cols[s] = interpolated_series(a, days)
            notes[s] = (f"Yahoo returned nothing for {y} — interpolated from "
                        f"{len(a)} fills")
            continue
        if p.isna().any():
            gap = p.isna()
            if held is not None and s in held.columns:
                n = int((held[s][gap] != 0).sum())
                if n:
                    notes[s] = (f"held for {n} day(s) before {y}'s first Yahoo "
                                f"print — back-filled at the first known close")
            p = p.bfill()
        if y != s:
            notes.setdefault(s, f"priced from {y} (ticker history)")
        cols[s] = p

    # Assembled in one pass: adding 140 columns to a live frame fragments it
    # badly enough that pandas warns, and this runs on every recompute.
    P = (pd.DataFrame(cols, index=days) if cols else pd.DataFrame(index=days))
    return P, notes


def benchmark_series(store: PriceStore, ticker: str,
                     days: pd.DatetimeIndex) -> pd.Series:
    """Total-return benchmark level: Adj Close, forward-filled onto the calendar."""
    s = store.adj_close(ticker)
    if s.empty:
        return pd.Series(np.nan, index=days)
    return s.reindex(days.union(s.index)).ffill().reindex(days)
