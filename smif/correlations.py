"""
smif/correlations.py — what is actually diversifying what.

ONE THING HERE IS EASY TO GET CATASTROPHICALLY WRONG, so it is said first.
These returns come from ADJUSTED closes, not from the real un-split-adjusted
prices the rest of the engine uses to value the book. The NAV needs the price a
share really traded at, because the ledger holds real share counts. A return
series needs the opposite: Nvidia's 10-for-1 in 2024 is a 90% fall in the real
close and no loss at all to a holder. Feed the valuation series into a
correlation and every split becomes a crash that every other holding failed to
join, which drags correlations toward zero and betas toward nothing. Dividends
are adjusted away too, for the same reason — an ex-dividend drop is not a move.

Three views, because "correlation" means three different questions:

  HOLDINGS      — which names actually move together. The diversification
                  question, asked at the level decisions are made.
  SECTORS       — dollar-weighted sector returns built from those same
                  holdings, so the sectors are the fund's, not the index's.
  SLEEVES       — equity, fixed income, total and their benchmarks, which is
                  where you find out whether the bond sleeve is doing the job
                  it was bought for.

Frequency is a choice with consequences and is left to the reader: daily
correlations are noisy and understate genuine co-movement (non-synchronous
closes, bid-ask bounce); monthly ones are cleaner but a three-year window
holds only 36 observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, nav, prices as price_mod

# label -> (pandas resample rule or None for daily, minimum useful observations)
FREQUENCIES: dict[str, tuple[str | None, int]] = {
    "Daily": (None, 30),
    "Weekly": ("W-FRI", 20),
    "Monthly": ("ME", 12),
}


def _resample(level: pd.DataFrame, freq: str) -> pd.DataFrame:
    rule, _ = FREQUENCIES[freq]
    return level if rule is None else level.resample(rule).last()


def _window(s: pd.Series, start, end) -> pd.Series:
    """
    Clip a series to the window, tolerating one that is not there at all.

    An absent ticker comes back as an empty Series carrying a RangeIndex, and
    comparing that to a Timestamp raises TypeError rather than returning
    nothing — so a benchmark missing from the cache would take the whole page
    down instead of leaving one row blank.
    """
    if s is None or len(s) == 0:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    if not isinstance(s.index, pd.DatetimeIndex):
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


def price_returns(store: price_mod.PriceStore, symbols, start, end,
                  freq: str = "Daily") -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Total-return series for each symbol over the window, at the given frequency.

    Returns (frame, {symbol: why it was dropped}). A name is dropped rather
    than forward-filled into existence: a constant price reads as zero variance,
    which produces a correlation of NaN at best and a confident 0.00 at worst.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    _rule, min_obs = FREQUENCIES[freq]
    series, dropped = {}, {}

    for sym in sorted(set(symbols)):
        sym = str(sym).upper()
        if sym in config.CASH_LIKE:
            dropped[sym] = "cash-like: no price to correlate"
            continue
        if sym in config.NO_YAHOO_DATA:
            dropped[sym] = "interpolated price, not an observed series"
            continue
        ticker = price_mod.yahoo_ticker(sym)
        s = _window(store.adj_close(ticker) if ticker else None,
                    start, end).dropna()
        if len(s) < min_obs:
            dropped[sym] = f"only {len(s)} closes in the window"
            continue
        series[sym] = s

    if not series:
        return pd.DataFrame(), dropped

    level = pd.DataFrame(series).sort_index()
    rets = _resample(level, freq).pct_change().iloc[1:]
    rets = rets.replace([np.inf, -np.inf], np.nan)

    # A column that is mostly missing cannot support a correlation; pandas
    # would happily compute one from the handful of overlapping rows.
    keep = rets.columns[rets.notna().sum() >= min_obs]
    for c in set(rets.columns) - set(keep):
        dropped[c] = f"only {int(rets[c].notna().sum())} {freq.lower()} returns"
    return rets[list(keep)], dropped


def benchmark_returns(store: price_mod.PriceStore, start, end,
                      freq: str = "Daily") -> pd.DataFrame:
    out = {}
    for name in ("SPY", "AGG"):
        s = _window(store.adj_close(name), start, end)
        if len(s) > 2:
            out[name] = s
    if not out:
        return pd.DataFrame()
    return _resample(pd.DataFrame(out).sort_index(), freq).pct_change().iloc[1:]


def sleeve_returns(tracks: dict, store: price_mod.PriceStore, start, end,
                   freq: str = "Daily") -> pd.DataFrame:
    """
    Sleeve and benchmark returns on one frequency.

    Sleeve returns are COMPOUNDED into the period, not resampled from a level:
    the growth index is the only continuous series a sleeve has, and taking
    its last value each week is the same thing as compounding the days inside.
    Doing it explicitly keeps it obvious that a weekly sleeve return is the
    product of its days, not a Friday-to-Friday price change.
    """
    rule, _ = FREQUENCIES[freq]
    cols = {}
    for key in ("total", "equity", "fixed_income", "cash"):
        t = tracks.get(key)
        if t is None:
            continue
        # A sleeve's daily series is zero-padded back to the fund's inception
        # so it can be chained without special-casing. Those zeros are not
        # observations: correlating a sleeve bought in September across three
        # years would compare six weeks of returns against seven hundred days
        # of manufactured calm, which reads as "uncorrelated with everything"
        # and deflates its volatility toward nothing. Start each sleeve where
        # it actually starts and let the join produce NaN before that.
        first = max(pd.Timestamp(start), t.base)
        r = t.daily.loc[first:pd.Timestamp(end)]
        if first == t.base and len(r):
            # The base day is the close BEFORE the sleeve's first return, and
            # growth_index() discards its return by construction. Including it
            # here would contribute one padded zero.
            r = r.iloc[1:]
        if r.empty:
            continue
        cols[config.SLEEVE_LABEL[key]] = (
            r if rule is None else r.resample(rule).apply(lambda x: (1 + x).prod() - 1))
    out = pd.DataFrame(cols)
    bench = benchmark_returns(store, start, end, freq)
    if not bench.empty:
        out = out.join(bench, how="outer")
    return out.dropna(how="all")


def sector_returns(holding_rets: pd.DataFrame, weights: pd.Series,
                   sectors: dict[str, str]) -> pd.DataFrame:
    """
    Dollar-weighted sector returns built from the fund's own holdings.

    Current weights, held fixed across the window: the question is "how do the
    sectors I own today move together", and re-weighting by history would
    answer a different one. Weights are renormalised inside each sector, so a
    sector return is a weighted average of its names rather than a sum that
    happens to be small because the sector is small.
    """
    if holding_rets.empty:
        return pd.DataFrame()
    cols = {}
    for sector in sorted({sectors.get(c) for c in holding_rets.columns
                          if sectors.get(c)}):
        members = [c for c in holding_rets.columns if sectors.get(c) == sector]
        w = weights.reindex(members).fillna(0.0)
        if w.sum() <= 0:
            continue
        cols[sector] = (holding_rets[members] * w).sum(axis=1, min_count=1) / w.sum()
    return pd.DataFrame(cols)


def correlation(returns: pd.DataFrame, min_obs: int = 20) -> pd.DataFrame:
    """Pairwise correlation, pairs with too little overlap left as NaN."""
    if returns.empty or returns.shape[1] < 2:
        return pd.DataFrame()
    corr = returns.corr(min_periods=min_obs)
    return corr


def clustering_available() -> bool:
    """
    Whether the heatmap can actually be clustered.

    cluster_order() falls back to the given order when SciPy is missing, which
    is the right behaviour — an unclustered heatmap beats no heatmap — but a
    silent fallback means a page promising a clustered view quietly stops
    delivering one. The UI asks this so it can say so instead.
    """
    try:
        import scipy.cluster.hierarchy  # noqa: F401
        return True
    except ImportError:
        return False


def cluster_order(corr: pd.DataFrame) -> list[str]:
    """
    Reorder so that things that move together sit together.

    Average-linkage hierarchical clustering on 1 - correlation, which is a
    proper distance on [0, 2]. Without this the heatmap is in alphabetical
    order and the block structure — the actual finding — is invisible. Falls
    back to the original order if SciPy is unavailable or the matrix is
    degenerate, because an unclustered heatmap is still a heatmap.
    """
    labels = list(corr.columns)
    if len(labels) < 3:
        return labels
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        return labels
    try:
        filled = corr.fillna(0.0).to_numpy(dtype=float)
        np.fill_diagonal(filled, 1.0)
        dist = 1.0 - filled
        dist = (dist + dist.T) / 2.0          # enforce exact symmetry
        np.fill_diagonal(dist, 0.0)
        dist[dist < 0] = 0.0
        order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
        return [labels[i] for i in order]
    except (ValueError, FloatingPointError):
        return labels


def beta_table(returns: pd.DataFrame, market: pd.Series,
               weights: pd.Series | None = None,
               min_obs: int = 20) -> pd.DataFrame:
    """
    Each column's correlation to the market, its beta, and its R-squared.

    Beta is cov / var on the SAME frequency as the correlation, so a monthly
    beta and a monthly correlation describe the same observations. R-squared
    is reported because beta alone cannot say whether the relationship is worth
    trusting: a 1.4 beta explaining 9% of the variance is noise with a slope.
    """
    rows = []
    market = market.dropna()
    for col in returns.columns:
        pair = pd.concat([returns[col], market], axis=1, join="inner").dropna()
        if len(pair) < min_obs:
            rows.append({"symbol": col, "corr_spy": np.nan, "beta": np.nan,
                         "r2": np.nan, "obs": len(pair), "vol": np.nan})
            continue
        y, x = pair.iloc[:, 0], pair.iloc[:, 1]
        var = float(x.var(ddof=1))
        corr = float(y.corr(x))
        beta = float(y.cov(x) / var) if var > 0 else np.nan
        rows.append({"symbol": col, "corr_spy": corr, "beta": beta,
                     "r2": corr ** 2 if corr == corr else np.nan,
                     "obs": len(pair), "vol": float(y.std(ddof=1))})
    out = pd.DataFrame(rows).set_index("symbol")
    if weights is not None:
        out["weight"] = weights.reindex(out.index)
    return out


def average_pair_correlation(corr: pd.DataFrame) -> float:
    """Mean of the off-diagonal entries — one number for "how diversified"."""
    if corr.empty or corr.shape[0] < 2:
        return float("nan")
    m = corr.to_numpy(dtype=float).copy()
    iu = np.triu_indices_from(m, k=1)
    vals = m[iu]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def effective_holdings(corr: pd.DataFrame, weights: pd.Series) -> float:
    """
    How many independent bets the book really holds.

    1 / (w' C w) — the number of equally weighted, uncorrelated positions that
    would give the same variance. Thirty-eight names that all move together are
    not thirty-eight bets, and this is the number that says so.
    """
    if corr.empty:
        return float("nan")
    cols = [c for c in corr.columns if c in weights.index]
    if len(cols) < 2:
        return float("nan")
    w = weights.reindex(cols).fillna(0.0).to_numpy(dtype=float)
    if w.sum() <= 0:
        return float("nan")
    w = w / w.sum()
    c = corr.loc[cols, cols].fillna(0.0).to_numpy(dtype=float)
    np.fill_diagonal(c, 1.0)
    denom = float(w @ c @ w)
    return 1.0 / denom if denom > 0 else float("nan")


def current_weights(book: nav.Book, sleeve: str | None = None) -> pd.Series:
    """Current dollar weights, optionally restricted to one sleeve."""
    h = book.holdings_on(min_value=config.MATERIAL_POSITION_VALUE)
    if sleeve is not None:
        h = h[h["sleeve"] == sleeve]
    v = h["value"]
    total = v.sum()
    return (v / total) if total else v
