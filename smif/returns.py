"""
smif/returns.py — time-weighted returns for each sleeve and for the fund.

TWRR exists to answer "how well were these assets managed", which means the
contributions and withdrawals the manager did not choose must not count. What
qualifies as a flow depends on whose return you are measuring, and that is the
whole of the method:

  TOTAL PORTFOLIO — only Deposits and Withdrawals are flows. A dividend is not:
  the fund earned it, and it moved from the equity pocket to the cash pocket
  without leaving the fund.

        r(t) = (V(t) - external flows(t)) / V(t-1) - 1

  A SLEEVE — the sleeve cannot see the cash pocket, so a dividend leaves it and
  has to be added back, and a purchase arrives from outside it and has to be
  taken out.

        r(t) = (V(t) + income(t) - net purchases(t)) / V(t-1) - 1

  Flows are treated as END-OF-DAY. The export carries no intraday timestamp, so
  any other assumption would be a guess dressed up as precision. The error it
  can introduce is one day of return on the flow, which on this fund's cash
  movements is immaterial.

  THE FIRST DAY of a sleeve is the exception: there is no prior value to divide
  by, so the return is the close against the cost of the opening purchase. That
  is why each growth index is based on the day BEFORE its first purchase — it
  gives the fill-to-close move of day one somewhere to live instead of silently
  discarding it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25
MIN_SLEEVE_VALUE = 1.0        # below this a sleeve is empty, not down 99%

PERIODS: list[tuple[str, object]] = [
    ("All time", None),
    ("10Y", pd.DateOffset(years=10)),
    ("5Y", pd.DateOffset(years=5)),
    ("3Y", pd.DateOffset(years=3)),
    ("1Y", pd.DateOffset(years=1)),
    ("YTD", "ytd"),
]
PERIOD_YEARS = {"All time": None, "10Y": 10, "5Y": 5, "3Y": 3, "1Y": 1, "YTD": 0}


@dataclass
class Track:
    """One return track: its daily returns, its growth index, its benchmark."""
    name: str
    daily: pd.Series
    index: pd.Series
    value: pd.Series
    benchmark: pd.Series | None = None
    benchmark_name: str | None = None

    @property
    def base(self) -> pd.Timestamp:
        return self.index.index[0]

    @property
    def start(self) -> pd.Timestamp:
        """First day the track actually earned a return."""
        nz = self.daily[self.daily != 0]
        return nz.index[0] if len(nz) else self.base


# ── daily returns ─────────────────────────────────────────────────────────────

def total_daily_returns(book) -> pd.Series:
    """Fund-level TWRR: neutralise deposits and withdrawals, nothing else."""
    v, f = book.total, book.flows
    prev = v.shift(1)
    r = (v - f) / prev - 1
    r = r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    r.iloc[0] = 0.0
    return r


def sleeve_daily_returns(book, sleeve: str) -> tuple[pd.Series, pd.Timestamp | None]:
    """
    A sleeve's TWRR and the day it opened.

    Days before the sleeve holds anything return zero rather than NaN so the
    series can be chained without special-casing; the index is based at the
    sleeve's own opening anyway, which is where the zeros stop mattering.
    """
    v = book.sleeve_value[sleeve]
    inc = book.sleeve_income[sleeve]
    nb = book.sleeve_netbuy[sleeve]
    prev = v.shift(1)

    r = pd.Series(0.0, index=book.days)
    live = prev > MIN_SLEEVE_VALUE
    r[live] = ((v + inc - nb) / prev - 1)[live]

    opened = v[v > MIN_SLEEVE_VALUE]
    if opened.empty:
        return r, None
    start = opened.index[0]
    # Day one: the sleeve's close against what it cost to open.
    cost = nb.loc[start]
    r.loc[start] = (v.loc[start] - cost) / cost if cost > MIN_SLEEVE_VALUE else 0.0
    return r.replace([np.inf, -np.inf], 0.0).fillna(0.0), start


# ── chaining ──────────────────────────────────────────────────────────────────

def growth_index(daily: pd.Series, base: pd.Timestamp) -> pd.Series:
    """Chain daily returns from 1.0 on `base`; the base day itself earns nothing."""
    r = daily.loc[base:].copy()
    if r.empty:
        return pd.Series(dtype=float)
    r.iloc[0] = 0.0
    return (1.0 + r).cumprod()


def base_day(days: pd.DatetimeIndex, start: pd.Timestamp | None) -> pd.Timestamp:
    """The close before a sleeve's first purchase — see the module docstring."""
    if start is None:
        return days[0]
    pos = days.get_loc(start)
    return days[max(pos - 1, 0)]


def build_track(book, which: str, store=None, *, overrides=None) -> Track | None:
    """
    A complete track for 'total' or a sleeve name, benchmark attached.

    The benchmark is rebased to the same day as the track, so every comparison
    in the app is like-for-like by construction rather than by convention.
    """
    from . import prices as price_mod

    if which == "total":
        daily = total_daily_returns(book)
        base = book.days[0]
        value = book.total
    else:
        daily, start = sleeve_daily_returns(book, which)
        if start is None:
            return None
        base = base_day(book.days, start)
        value = book.sleeve_value[which]

    idx = growth_index(daily, base)
    bname = config.BENCHMARKS.get(which)
    bench = None
    if bname and store is not None:
        b = price_mod.benchmark_series(store, bname, book.days).loc[base:]
        if b.notna().any():
            bench = b / b.iloc[0]
    return Track(name=config.SLEEVE_LABEL.get(which, which), daily=daily, index=idx,
                 value=value, benchmark=bench, benchmark_name=bname)


def build_tracks(book, store, *, overrides=None) -> dict[str, Track]:
    out = {}
    for which in ("total", "equity", "fixed_income", "cash"):
        t = build_track(book, which, store, overrides=overrides)
        if t is not None:
            out[which] = t
    return out


# ── statistics ────────────────────────────────────────────────────────────────

def max_drawdown(index: pd.Series) -> float:
    if index.empty:
        return float("nan")
    return float((index / index.cummax() - 1).min())


def drawdown_window(index: pd.Series) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """(peak date, trough date) of the worst drawdown — shaded on the chart."""
    if index.empty:
        return None, None
    dd = index / index.cummax() - 1
    trough = dd.idxmin()
    peak = index.loc[:trough].idxmax()
    return peak, trough


def annualized_vol(daily: pd.Series) -> float:
    d = daily.dropna()
    if len(d) < 2:
        return float("nan")
    return float(d.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def period_base(index: pd.Series, label: str, offset, end: pd.Timestamp) -> pd.Timestamp:
    """
    The last close on or before the period's nominal start.

    "Last close on or before" rather than "nearest" matters: a base one day
    later than it should be silently shortens the window and inflates the
    return whenever the market fell on the boundary day.
    """
    if offset is None:
        return index.index[0]
    target = (pd.Timestamp(f"{end.year - 1}-12-31") if offset == "ytd"
              else end - offset)
    on_or_before = index.index[index.index <= target]
    return on_or_before[-1] if len(on_or_before) else index.index[0]


def _stats_between(track: Track, base: pd.Timestamp, end: pd.Timestamp,
                   label: str, annualize: bool) -> dict:
    """Every statistic the app reports, over one window. The only place the
    arithmetic lives, so a custom range and a named period cannot disagree."""
    idx = track.index
    a = idx.loc[base:end]
    years = (end - base).days / DAYS_PER_YEAR
    cum = float(a.iloc[-1] / a.iloc[0] - 1)

    out = {
        "period": label, "start": base, "end": end, "years": years,
        "cum": cum,
        "ann": ((1 + cum) ** (1 / years) - 1) if (annualize and years > 0) else None,
        "mdd": max_drawdown(a),
        "vol": annualized_vol(track.daily.loc[base:end].iloc[1:]),
        "bench_cum": None, "bench_ann": None, "bench": track.benchmark_name,
        "excess": None,
    }
    if track.benchmark is not None:
        b = track.benchmark.loc[base:end].dropna()
        if len(b) > 1:
            bc = float(b.iloc[-1] / b.iloc[0] - 1)
            out["bench_cum"] = bc
            out["bench_ann"] = (((1 + bc) ** (1 / years) - 1)
                                if (annualize and years > 0) else None)
            out["excess"] = cum - bc
    return out


def period_stats(track: Track, label: str, offset, end: pd.Timestamp | None = None) -> dict:
    idx = track.index
    end = pd.Timestamp(end) if end is not None else idx.index[-1]
    end = idx.index[idx.index <= end][-1]
    base = period_base(idx, label, offset, end)
    years = (end - base).days / DAYS_PER_YEAR
    nominal = PERIOD_YEARS.get(label)
    annualize = years >= 1.0 or (nominal is not None and nominal >= 1)
    return _stats_between(track, base, end, label, annualize)


def range_stats(track: Track, start, end=None) -> dict | None:
    """
    The same statistics over an arbitrary window, for the custom date picker.

    Both ends snap to the last close on or before the date asked for, so a
    weekend boundary measures a real session instead of silently shifting. None
    when the track does not reach the window at all — a fixed income sleeve
    bought in September has nothing to say about 2015, and inventing a zero
    would be worse than saying so.
    """
    idx = track.index
    start, end = pd.Timestamp(start), pd.Timestamp(end or idx.index[-1])
    within = idx.index[(idx.index >= start) & (idx.index <= end)]
    if len(within) < 2:
        return None
    on_or_before = idx.index[idx.index <= start]
    base = on_or_before[-1] if len(on_or_before) else within[0]
    last = idx.index[idx.index <= end][-1]
    years = (last - base).days / DAYS_PER_YEAR
    return _stats_between(track, base, last, "Custom", years >= 1.0)


def period_table(track: Track, end=None, periods=None) -> pd.DataFrame:
    """One row per period: cumulative, annualized, drawdown, volatility."""
    rows = []
    for label, offset in (periods or PERIODS):
        try:
            rows.append(period_stats(track, label, offset, end))
        except (IndexError, ZeroDivisionError):
            continue
    return pd.DataFrame(rows)


def calendar_years(track: Track, end=None) -> pd.DataFrame:
    """
    Calendar-year returns against the benchmark.

    The first and last rows are partial years — the fund did not start on 1
    January and the current year is not finished — and are labelled as such
    rather than quietly presented as full years.
    """
    idx = track.index
    if idx.empty:
        return pd.DataFrame(columns=["year", "portfolio", "benchmark", "partial"])
    end = pd.Timestamp(end) if end is not None else idx.index[-1]
    idx = idx.loc[:end]
    ye = idx.resample("YE").last()
    rows = []
    prev_p = 1.0
    prev_b = None
    bench = track.benchmark.loc[:end] if track.benchmark is not None else None
    if bench is not None:
        bye = bench.resample("YE").last()
        prev_b = 1.0
    for i, (ts, p) in enumerate(ye.items()):
        year = ts.year
        b = None
        if bench is not None and ts in bye.index:
            b = float(bye.loc[ts] / prev_b - 1)
            prev_b = bye.loc[ts]
        partial = (year == idx.index[0].year) or (year == end.year and end.month < 12)
        rows.append({"year": year, "portfolio": float(p / prev_p - 1),
                     "benchmark": b, "partial": partial})
        prev_p = p
    return pd.DataFrame(rows)


def rebased(series: pd.Series, base=None) -> pd.Series:
    """Percent growth from a chosen base day — what the charts plot."""
    s = series.loc[pd.Timestamp(base):] if base is not None else series
    s = s.dropna()
    if s.empty:
        return s
    return (s / s.iloc[0] - 1) * 100.0
