"""
smif/factors.py — what kind of book is this, measured against the S&P 500.

Each holding's characteristics are z-scored CROSS-SECTIONALLY against the index,
then dollar-weighted into one number per factor. The index is zero by
construction, so every score reads as "standard deviations away from the
market".

Four rules decide whether the output means anything. Getting any of them wrong
produces a confident, plausible, wrong chart:

  · Z-SCORE FIRST, THEN WEIGHT-AVERAGE. Averaging raw P/Es and z-scoring the
    result aggregates in a skewed space and gives a different answer. Megacaps
    would dominate a mean of ratios in a way they do not dominate a mean of
    ranks.

  · EVERY METRIC IS STORED ALREADY ORIENTED so that higher means more of the
    factor. Value metrics are therefore YIELDS (E/P, FCF/P), never P/E-style
    ratios: yields behave far better statistically, and a negative yield is
    meaningfully "losing money" where a negative P/E is nonsense.

  · WINSORIZE BEFORE COMPUTING THE SCALE, then clip the z. One bad print — a
    spin-off with a stub share count, a company with $3 of book equity — would
    otherwise set the standard deviation for all 500.

  · MISSING DATA IS EXCLUDED AND THE REMAINING WEIGHTS RENORMALISED, never
    imputed as zero. Imputing zero is a fabricated claim that the holding is
    exactly average. Coverage is reported alongside every score so a thinly
    covered metric can be discounted.

Sign convention is INTUITIVE, not academic: Size up means LARGER (the academic
SMB factor is the inverse), Value up means CHEAPER, Momentum up means MORE.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

FACTOR_ORDER = ["Size", "Quality", "Growth", "Value", "Momentum"]

# key, factor, short label, definition shown on hover
METRICS: list[tuple[str, str, str, str]] = [
    ("logMcap", "Size", "Market cap",
     "Log market capitalisation. Log because raw cap is so skewed that a "
     "handful of megacaps would set the whole scale. Up = larger."),

    ("roic", "Quality", "ROIC",
     "NOPAT / invested capital, at a 21% statutory tax rate. The cleanest read "
     "on whether the business earns more than its cost of capital."),
    ("grossMargin", "Quality", "Gross margin",
     "Gross profit / revenue. Pricing power and mix, largely free of the "
     "accounting choices that move operating margin."),
    ("roe", "Quality", "ROE",
     "Return on equity. Inflated by leverage — a levered balance sheet can "
     "manufacture a high ROE, so read it alongside ROIC."),

    ("salesGrowth", "Growth", "Revenue growth",
     "Year-on-year revenue growth. Cleaner than EPS growth, which buybacks can "
     "inflate without the business growing at all."),
    ("epsGrowth", "Growth", "EPS growth",
     "Year-on-year earnings-per-share growth. Partly a capital-allocation "
     "signal, since buybacks raise it mechanically."),

    ("earnYield", "Value", "Earnings yield",
     "E/P — the inverse of the P/E. Higher = cheaper."),
    ("fcfYield", "Value", "FCF yield",
     "Free cash flow / market cap. The best single value metric here: cash "
     "rather than accruals."),
    ("ebitdaEv", "Value", "EBITDA / EV",
     "The inverse of EV/EBITDA. Capital-structure neutral, so it neither "
     "rewards nor punishes leverage the way a P/E does. Higher = cheaper."),

    ("mom12_1", "Momentum", "12-1 month",
     "Twelve-month return skipping the most recent month — the standard "
     "construction, since the latest month carries short-term reversal."),
    ("mom6", "Momentum", "6-month",
     "Six-month total return. A faster-moving confirmation of the 12-1 read."),
    ("high52", "Momentum", "52-week high",
     "Current price as a fraction of the 52-week high. Near 1.0 means the name "
     "is pressing its highs; a low reading means it is well off them."),
]

METRIC_FACTOR = {k: f for k, f, _s, _d in METRICS}
METRIC_LABEL = {k: s for k, _f, s, _d in METRICS}
METRIC_DESC = {k: d for k, _f, _s, d in METRICS}
METRIC_KEYS = [k for k, *_ in METRICS]
PRICE_METRICS = ["mom12_1", "mom6", "high52"]


# ── the statistics ────────────────────────────────────────────────────────────

def winsorize(values: np.ndarray, pct: float | None = None) -> np.ndarray:
    pct = config.FACTOR_WINSOR_PCT if pct is None else pct
    if values.size == 0:
        return values
    lo, hi = np.nanpercentile(values, [pct, 100 - pct])
    return np.clip(values, lo, hi)


def z_distribution(universe: dict[str, float | None]) -> dict | None:
    """Winsorized mean and sigma of one metric across the reference universe."""
    vals = np.array([v for v in universe.values()
                     if v is not None and np.isfinite(v)], dtype=float)
    if vals.size < config.FACTOR_MIN_UNIVERSE:
        return None
    w = winsorize(vals)
    sd = float(np.std(w, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        return None
    return {"mean": float(np.mean(w)), "sd": sd, "lo": float(np.min(w)),
            "hi": float(np.max(w)), "n": int(vals.size)}


def z_of(value: float | None, dist: dict | None) -> float | None:
    if value is None or dist is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    v = min(max(v, dist["lo"]), dist["hi"])          # winsorize the input too
    return float(np.clip((v - dist["mean"]) / dist["sd"], -config.FACTOR_Z_CLIP,
                         config.FACTOR_Z_CLIP))


def exposure(weights: dict[str, float],
             metrics_by_ticker: dict[str, dict],
             universe: dict[str, dict],
             top_n: int = 5) -> dict:
    """
    Portfolio factor exposure.

    weights           {ticker: dollar weight}, need not sum to 1
    metrics_by_ticker {ticker: {metric: raw value}} for the holdings
    universe          {ticker: {metric: raw value}} for the S&P 500

    Returns per-metric weighted z with coverage and biggest contributors, plus
    one composite per factor.
    """
    live = {t: w for t, w in weights.items() if w and w > 0}
    total_w = sum(live.values())
    if total_w <= 0:
        return {"metrics": [], "factors": [], "coverage": 0.0, "universeN": 0}

    dists = {key: z_distribution({t: (m or {}).get(key)
                                  for t, m in universe.items()})
             for key in METRIC_KEYS}

    metric_rows = []
    for key in METRIC_KEYS:
        num = den = 0.0
        contributors = []
        for t, w in live.items():
            z = z_of((metrics_by_ticker.get(t) or {}).get(key), dists[key])
            if z is None:
                continue                       # excluded, NOT counted as zero
            num += w * z
            den += w
            contributors.append((t, z, w))
        base = {"key": key, "factor": METRIC_FACTOR[key],
                "label": METRIC_LABEL[key], "desc": METRIC_DESC[key]}
        if den <= 0:
            metric_rows.append({**base, "z": None, "coverage": 0.0, "n": 0,
                                "top": [], "bottom": []})
            continue
        # ranked by CONTRIBUTION (weight x z), not by z alone: the names that
        # move the score are the big positions, not the most extreme ones
        contributors.sort(key=lambda c: -(c[1] * c[2]))
        metric_rows.append({
            **base,
            "z": round(num / den, 3),          # renormalised over covered weight
            "coverage": round(den / total_w * 100, 0),
            "n": len(contributors),
            "top": [{"ticker": t, "z": round(z, 2), "weight": round(w * 100, 1)}
                    for t, z, w in contributors[:top_n]],
            "bottom": [{"ticker": t, "z": round(z, 2), "weight": round(w * 100, 1)}
                       for t, z, w in contributors[-top_n:]][::-1],
        })

    factor_rows = []
    for f in FACTOR_ORDER:
        zs = [m["z"] for m in metric_rows if m["factor"] == f and m["z"] is not None]
        cov = [m["coverage"] for m in metric_rows
               if m["factor"] == f and m["z"] is not None]
        factor_rows.append({
            "factor": f,
            "z": round(float(np.mean(zs)), 3) if zs else None,
            "nMetrics": len(zs),
            "coverage": round(float(np.mean(cov)), 0) if cov else 0.0,
        })

    covered = [m["coverage"] for m in metric_rows if m["z"] is not None]
    return {
        "metrics": metric_rows,
        "factors": factor_rows,
        "coverage": round(float(np.mean(covered)), 0) if covered else 0.0,
        "universeN": max([(d or {}).get("n", 0) for d in dists.values()] or [0]),
    }


# ── raw-metric helpers ────────────────────────────────────────────────────────

def safe_div(a, b):
    try:
        if a is None or b is None:
            return None
        a, b = float(a), float(b)
        if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-9:
            return None
        return a / b
    except (TypeError, ValueError):
        return None


def log_mcap(mcap) -> float | None:
    try:
        m = float(mcap)
    except (TypeError, ValueError):
        return None
    return math.log(m) if m > 0 else None


def _row(frame, *names):
    """First matching row of a yfinance statement frame, or None."""
    if frame is None or getattr(frame, "empty", True):
        return None
    for n in names:
        if n in frame.index:
            s = frame.loc[n].dropna()
            if len(s):
                return s
    return None


def momentum_from_prices(closes: pd.Series) -> dict:
    """
    The three momentum metrics, from an adjusted-close series.

    Computed here rather than fetched because the price panel is already
    downloaded and these are arithmetic, not data. 12-1 skips the most recent
    month by construction.
    """
    s = closes.dropna()
    out = {k: None for k in PRICE_METRICS}
    if len(s) >= 252:
        out["mom12_1"] = float(s.iloc[-21] / s.iloc[-252] - 1.0)
    if len(s) >= 126:
        out["mom6"] = float(s.iloc[-1] / s.iloc[-126] - 1.0)
    if len(s) >= 252:
        high = float(s.iloc[-252:].max())
        if high > 0:
            out["high52"] = float(s.iloc[-1] / high)
    return out


def fetch_fundamentals(ticker: str) -> dict:
    """
    The nine fundamental metrics for one name, from yfinance.

    One .info call plus the balance sheet and cash-flow statements. Every value
    comes back ALREADY ORIENTED so higher means more of the factor — the value
    metrics are inverted into yields here, not later.
    """
    import yfinance as yf

    out = {k: None for k in METRIC_KEYS if k not in PRICE_METRICS}
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
    except Exception:
        return out

    # .info is unreliable field by field, so fall back rather than silently
    # dropping a metric: Visa has returned 170-odd fields and no marketCap.
    mcap = info.get("marketCap")
    if not mcap:
        try:
            mcap = tk.fast_info.get("marketCap")
        except Exception:
            mcap = None

    revenue = info.get("totalRevenue")
    out["logMcap"] = log_mcap(mcap)
    out["roe"] = info.get("returnOnEquity")
    out["grossMargin"] = info.get("grossMargins")
    out["salesGrowth"] = info.get("revenueGrowth")
    out["epsGrowth"] = info.get("earningsGrowth")

    pe = info.get("trailingPE")
    out["earnYield"] = safe_div(1.0, pe) if (pe or 0) > 0 else None
    out["fcfYield"] = safe_div(info.get("freeCashflow"), mcap)
    ev_ebitda = info.get("enterpriseToEbitda")
    out["ebitdaEv"] = safe_div(1.0, ev_ebitda) if (ev_ebitda or 0) > 0 else None

    # Recorded so fund detection does not have to stay a hardcoded list: an
    # equity ETF bought next year has no ROIC either, and nothing else in the
    # pipeline would notice.
    qt = info.get("quoteType")
    if qt:
        out["quoteType"] = str(qt).upper()

    ebit = (info.get("operatingMargins") or 0) * (revenue or 0) or None
    try:
        bs = tk.balance_sheet
    except Exception:
        bs = None
    invested = _row(bs, "Invested Capital")
    if invested is not None and ebit is not None:
        # NOPAT / invested capital. The 21% statutory rate is a constant, and a
        # cross-sectional ranking is insensitive to a constant.
        out["roic"] = safe_div(ebit * 0.79, float(invested.iloc[0]))
    return out


# ── the universe price panel ──────────────────────────────────────────────────

UNIVERSE_PRICES = "universe_prices.parquet"
UNIVERSE_LOOKBACK_DAYS = 500        # ~2 years of sessions, enough for 12-1


def universe_price_path() -> Path:
    return config.CACHE_DIR / UNIVERSE_PRICES


def fetch_universe_prices(tickers: list[str], *, end=None, chunk: int = 40,
                          progress=None) -> pd.DataFrame:
    """
    Adjusted closes for the reference universe, in its own small cache.

    Deliberately NOT merged into the fund's price store. That cache holds
    seventeen years of real, un-split-adjusted closes for the ~120 tickers the
    fund has actually owned, and it is the thing the NAV is rebuilt from.
    Adding 500 index members would multiply it fivefold for data used by one
    chart, and would mix a valuation series with a return series in one file.
    Two years of adjusted closes is all momentum needs.
    """
    import yfinance as yf

    end = pd.Timestamp(end) if end is not None else config.today()
    start = (end - pd.Timedelta(days=int(UNIVERSE_LOOKBACK_DAYS * 1.5))
             ).strftime("%Y-%m-%d")
    end_excl = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    tickers = sorted(set(tickers))
    frames = {}
    for i in range(0, len(tickers), chunk):
        names = tickers[i:i + chunk]
        if progress:
            progress(i, len(tickers), names[0])
        try:
            raw = yf.download(names, start=start, end=end_excl, auto_adjust=False,
                              actions=False, progress=False, group_by="ticker",
                              threads=True)
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        for t in names:
            try:
                x = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                s = x["Adj Close"].dropna()
                if len(s) > 100:
                    frames[t] = s
            except (KeyError, TypeError):
                continue
        time.sleep(1.0)

    if not frames:
        return pd.DataFrame()
    panel = pd.DataFrame(frames).sort_index()
    panel.index = pd.DatetimeIndex(panel.index).tz_localize(None)
    panel = panel[panel.index < config.today()]        # no partial session
    path = universe_price_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path)
    return panel


def load_universe_prices() -> pd.DataFrame:
    p = universe_price_path()
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except (OSError, ValueError):
        return pd.DataFrame()


# ── the cached snapshot ───────────────────────────────────────────────────────

def _cache_path() -> Path:
    return config.FUNDAMENTALS_CACHE


def load_snapshot() -> dict:
    p = _cache_path()
    if not p.exists():
        return {"metrics": {}, "asof": None, "n": 0, "status": "not_built"}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"metrics": {}, "asof": None, "n": 0, "status": "unreadable"}


def snapshot_age_days(snapshot: dict) -> float | None:
    asof = snapshot.get("asof")
    if not asof:
        return None
    try:
        when = datetime.fromisoformat(asof)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).total_seconds() / 86400


def is_stale(snapshot: dict) -> bool:
    age = snapshot_age_days(snapshot)
    return age is None or age > config.FUNDAMENTALS_TTL_DAYS


def _write(payload: dict) -> None:
    p = _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")


def build_snapshot(tickers: list[str], *, price_panel: pd.DataFrame | None = None,
                   progress=None, pause: float = 0.25,
                   checkpoint_every: int = 25) -> dict:
    """
    Crawl fundamentals for the universe and write the snapshot.

    Checkpointed: a crawl of 500 names takes several minutes and one that dies
    at minute seven must not lose everything, so the partial result is written
    every `checkpoint_every` names and a re-run resumes from disk.

    Momentum rides free off `price_panel` (adjusted closes, columns = tickers)
    because it is arithmetic on data already downloaded.
    """
    tickers = sorted(set(tickers))
    existing = load_snapshot().get("metrics", {})
    metrics: dict[str, dict] = {}
    fetched = failed = 0

    def payload(status: str = "ok") -> dict:
        return {
            "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n": len(metrics), "fetched": fetched, "failed": failed,
            "status": status,
            "coverage": {k: sum(1 for m in metrics.values()
                                if m.get(k) is not None) for k in METRIC_KEYS},
            "metrics": metrics,
        }

    for i, t in enumerate(tickers):
        base = dict(existing.get(t) or {})
        needs = all(base.get(k) is None for k in ("logMcap", "roe", "earnYield"))
        if needs:
            try:
                base.update(fetch_fundamentals(t))
                fetched += 1
            except Exception:
                failed += 1
            time.sleep(pause)
        if price_panel is not None and t in price_panel.columns:
            base.update(momentum_from_prices(price_panel[t]))
        metrics[t] = base
        if progress:
            progress(i + 1, len(tickers), t)
        if fetched and (i + 1) % checkpoint_every == 0:
            _write(payload("partial"))

    out = payload()
    _write(out)
    return out


def holding_metrics(snapshot: dict, tickers: list[str]) -> dict[str, dict]:
    m = snapshot.get("metrics", {})
    return {t: m.get(t, {}) for t in tickers}


FUND_QUOTE_TYPES = {"ETF", "MUTUALFUND", "MONEYMARKET", "INDEX"}


def is_fund(ticker: str, snapshot: dict | None = None) -> bool:
    """
    Whether a holding is a fund rather than a company.

    A fund has no ROIC, no revenue growth and no earnings yield — scoring one
    against a universe of operating companies is not a conservative
    approximation, it is a category error. The config lists what the fund holds
    today; the snapshot's quoteType catches whatever it buys next.
    """
    t = str(ticker).upper()
    if t in config.CASH_LIKE or t in config.FIXED_INCOME:
        return True
    qt = ((snapshot or {}).get("metrics", {}).get(t) or {}).get("quoteType")
    return qt in FUND_QUOTE_TYPES if qt else False


def coverage_table(snapshot: dict) -> pd.DataFrame:
    """How many universe names carry each metric — the data-quality view."""
    cov = snapshot.get("coverage") or {}
    n = snapshot.get("n") or 0
    rows = [{"metric": METRIC_LABEL.get(k, k), "factor": METRIC_FACTOR.get(k, ""),
             "covered": cov.get(k, 0),
             "share": (cov.get(k, 0) / n) if n else float("nan")}
            for k in METRIC_KEYS]
    return pd.DataFrame(rows)
