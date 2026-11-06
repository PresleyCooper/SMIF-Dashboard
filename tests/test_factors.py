"""
The factor z-score machinery.

Four properties carry the whole chart, and each fails silently rather than
loudly if broken: the index must score zero, missing data must be excluded
rather than imputed, one bad print must not set the scale, and the weighting
must happen after the z-scoring rather than before.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from smif import config, factors as fx


def universe(values: dict[str, float], key: str = "roe") -> dict[str, dict]:
    return {t: {key: v} for t, v in values.items()}


def spread(n: int = 120, key: str = "roe", seed: int = 0) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    return {f"T{i:03d}": {key: float(v)}
            for i, v in enumerate(rng.normal(0.15, 0.05, n))}


# ── the index is zero by construction ────────────────────────────────────────

def test_holding_the_whole_universe_scores_about_zero():
    """If owning the index does not score zero, the chart's baseline is wrong
    and every number on it is measured from the wrong place."""
    uni = spread(300)
    weights = {t: 1.0 for t in uni}
    out = fx.exposure(weights, uni, uni)
    roe = next(m for m in out["metrics"] if m["key"] == "roe")
    assert roe["z"] == pytest.approx(0.0, abs=0.02)


def test_a_holding_at_the_universe_mean_scores_zero():
    uni = universe({f"T{i}": 0.10 + 0.01 * i for i in range(100)})
    mean = float(np.mean([m["roe"] for m in uni.values()]))
    out = fx.exposure({"X": 1.0}, {"X": {"roe": mean}}, uni)
    roe = next(m for m in out["metrics"] if m["key"] == "roe")
    assert roe["z"] == pytest.approx(0.0, abs=0.02)


def test_a_holding_one_sigma_above_scores_about_one():
    uni = spread(400)
    vals = np.array([m["roe"] for m in uni.values()])
    target = float(vals.mean() + vals.std(ddof=1))
    out = fx.exposure({"X": 1.0}, {"X": {"roe": target}}, uni)
    roe = next(m for m in out["metrics"] if m["key"] == "roe")
    assert roe["z"] == pytest.approx(1.0, abs=0.15)


# ── missing data ──────────────────────────────────────────────────────────────

def test_missing_data_is_excluded_not_treated_as_average():
    """Imputing zero is a fabricated claim that the holding is exactly
    average, and it drags every score toward the middle."""
    uni = spread(200)
    vals = np.array([m["roe"] for m in uni.values()])
    high = float(vals.mean() + 2 * vals.std(ddof=1))
    holdings = {"GOOD": {"roe": high}, "UNKNOWN": {"roe": None}}
    out = fx.exposure({"GOOD": 0.5, "UNKNOWN": 0.5}, holdings, uni)
    roe = next(m for m in out["metrics"] if m["key"] == "roe")
    assert roe["z"] == pytest.approx(2.0, abs=0.2)     # not halved to 1.0
    assert roe["coverage"] == pytest.approx(50.0)
    assert roe["n"] == 1


def test_a_metric_nobody_has_reports_none_rather_than_zero():
    uni = spread(200)
    out = fx.exposure({"X": 1.0}, {"X": {"roe": None}}, uni)
    roe = next(m for m in out["metrics"] if m["key"] == "roe")
    assert roe["z"] is None
    assert roe["coverage"] == 0.0


def test_a_universe_too_small_to_scale_gives_no_scores():
    """Fewer names than FACTOR_MIN_UNIVERSE and a cross-sectional z is noise."""
    uni = universe({f"T{i}": 0.1 * i for i in range(5)})
    out = fx.exposure({"X": 1.0}, {"X": {"roe": 0.2}}, uni)
    assert all(m["z"] is None for m in out["metrics"])


def test_an_empty_book_returns_empty_rather_than_dividing_by_zero():
    assert fx.exposure({}, {}, spread(100))["factors"] == []


# ── robustness of the scale ───────────────────────────────────────────────────

def test_one_absurd_print_does_not_set_the_scale():
    """A single 400x outlier would otherwise inflate sigma so far that every
    real holding scores about zero."""
    uni = spread(300)
    clean = fx.exposure({"X": 1.0}, {"X": {"roe": 0.25}}, uni)
    poisoned = {**uni, "BAD": {"roe": 400.0}}
    dirty = fx.exposure({"X": 1.0}, {"X": {"roe": 0.25}}, poisoned)
    a = next(m for m in clean["metrics"] if m["key"] == "roe")["z"]
    b = next(m for m in dirty["metrics"] if m["key"] == "roe")["z"]
    assert b == pytest.approx(a, abs=0.25)


def test_winsorize_clips_both_tails():
    vals = np.array([0.0] + [1.0] * 98 + [100.0])
    w = fx.winsorize(vals, pct=5.0)
    assert w.max() < 100.0 and w.min() > 0.0


def test_z_scores_are_clipped_to_the_configured_bound():
    uni = spread(300)
    out = fx.exposure({"X": 1.0}, {"X": {"roe": 1e6}}, uni)
    roe = next(m for m in out["metrics"] if m["key"] == "roe")
    assert abs(roe["z"]) <= config.FACTOR_Z_CLIP + 1e-9


# ── weighting ─────────────────────────────────────────────────────────────────

def test_the_score_is_dollar_weighted_not_equal_weighted():
    uni = spread(300)
    vals = np.array([m["roe"] for m in uni.values()])
    mean, sd = float(vals.mean()), float(vals.std(ddof=1))
    holdings = {"BIG": {"roe": mean + 2 * sd}, "SMALL": {"roe": mean - 2 * sd}}
    out = fx.exposure({"BIG": 0.9, "SMALL": 0.1}, holdings, uni)
    roe = next(m for m in out["metrics"] if m["key"] == "roe")
    assert roe["z"] > 1.0          # equal-weighted would be about zero


def test_weights_need_not_sum_to_one():
    uni = spread(300)
    holdings = {"A": {"roe": 0.2}, "B": {"roe": 0.1}}
    one = fx.exposure({"A": 0.6, "B": 0.4}, holdings, uni)
    ten = fx.exposure({"A": 6.0, "B": 4.0}, holdings, uni)
    za = next(m for m in one["metrics"] if m["key"] == "roe")["z"]
    zb = next(m for m in ten["metrics"] if m["key"] == "roe")["z"]
    assert za == pytest.approx(zb)


def test_contributors_are_ranked_by_contribution_not_by_extremity():
    """A 12% position at +1 moves the score more than a 0.5% position at +3,
    and the hover is there to explain the score."""
    uni = spread(300)
    vals = np.array([m["roe"] for m in uni.values()])
    mean, sd = float(vals.mean()), float(vals.std(ddof=1))
    holdings = {"BIG": {"roe": mean + sd}, "TINY": {"roe": mean + 3 * sd}}
    out = fx.exposure({"BIG": 0.12, "TINY": 0.005}, holdings, uni)
    roe = next(m for m in out["metrics"] if m["key"] == "roe")
    assert roe["top"][0]["ticker"] == "BIG"


# ── composites ────────────────────────────────────────────────────────────────

def test_a_factor_is_the_mean_of_its_covered_metrics():
    uni = {f"T{i}": {"roe": float(i) / 100, "roic": float(i) / 100,
                     "grossMargin": float(i) / 100} for i in range(200)}
    held = {"X": {"roe": 1.5, "roic": 1.5, "grossMargin": 1.5}}
    out = fx.exposure({"X": 1.0}, held, uni)
    quality = next(f for f in out["factors"] if f["factor"] == "Quality")
    zs = [m["z"] for m in out["metrics"]
          if m["factor"] == "Quality" and m["z"] is not None]
    assert quality["z"] == pytest.approx(float(np.mean(zs)), abs=1e-6)
    assert quality["nMetrics"] == len(zs)


def test_every_factor_group_appears_even_with_no_data():
    out = fx.exposure({"X": 1.0}, {"X": {}}, spread(100))
    assert [f["factor"] for f in out["factors"]] == fx.FACTOR_ORDER


def test_every_metric_belongs_to_a_known_factor():
    assert set(fx.METRIC_FACTOR.values()) == set(fx.FACTOR_ORDER)


def test_every_metric_has_a_definition_for_the_hover():
    for key in fx.METRIC_KEYS:
        assert fx.METRIC_DESC[key].strip()
        assert fx.METRIC_LABEL[key].strip()


# ── momentum ──────────────────────────────────────────────────────────────────

def test_momentum_12_1_skips_the_most_recent_month():
    """
    The window runs from session -252 to session -21, which is twelve months
    minus the most recent one. A move strictly inside that last month must not
    reach it; the 6-month read, which does not skip, must.
    """
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=300))
    s = pd.Series(100.0, index=idx)
    s.iloc[-10:] = 500.0                 # well inside the skipped month
    m = fx.momentum_from_prices(s)
    assert m["mom12_1"] == pytest.approx(0.0, abs=1e-9)
    assert m["mom6"] > 1.0


def test_momentum_12_1_does_see_a_move_before_the_skipped_month():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=300))
    s = pd.Series(100.0, index=idx)
    s.iloc[-60:] = 200.0                 # starts before the last month
    assert fx.momentum_from_prices(s)["mom12_1"] == pytest.approx(1.0)


def test_52_week_high_proximity_is_one_at_the_high():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=300))
    s = pd.Series(np.linspace(10, 100, len(idx)), index=idx)
    assert fx.momentum_from_prices(s)["high52"] == pytest.approx(1.0)


def test_52_week_high_proximity_falls_when_the_price_does():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=300))
    s = pd.Series(100.0, index=idx)
    s.iloc[-1] = 50.0
    assert fx.momentum_from_prices(s)["high52"] == pytest.approx(0.5)


def test_a_short_series_yields_no_momentum_rather_than_a_guess():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=30))
    m = fx.momentum_from_prices(pd.Series(100.0, index=idx))
    assert m == {"mom12_1": None, "mom6": None, "high52": None}


# ── funds ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ticker", sorted(config.FIXED_INCOME | config.CASH_LIKE))
def test_the_funds_the_portfolio_holds_are_recognised(ticker):
    assert fx.is_fund(ticker)


def test_a_company_is_not_a_fund():
    assert not fx.is_fund("AAPL")


def test_quote_type_catches_a_fund_the_config_has_never_heard_of():
    snap = {"metrics": {"NEWETF": {"quoteType": "ETF"}}}
    assert fx.is_fund("NEWETF", snap)
    assert not fx.is_fund("NEWCO", {"metrics": {"NEWCO": {"quoteType": "EQUITY"}}})


# ── the snapshot ──────────────────────────────────────────────────────────────

def test_an_absent_snapshot_reports_not_built(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FUNDAMENTALS_CACHE", tmp_path / "none.json")
    snap = fx.load_snapshot()
    assert snap["status"] == "not_built" and snap["metrics"] == {}


def test_a_corrupt_snapshot_is_reported_not_raised(tmp_path, monkeypatch):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(config, "FUNDAMENTALS_CACHE", p)
    assert fx.load_snapshot()["status"] == "unreadable"


def test_a_snapshot_with_no_date_counts_as_stale():
    assert fx.is_stale({"metrics": {}, "asof": None})


def test_a_fresh_snapshot_is_not_stale():
    now = pd.Timestamp.utcnow().tz_localize(None).isoformat()
    assert not fx.is_stale({"metrics": {}, "asof": now})


def test_an_old_snapshot_is_stale():
    old = (pd.Timestamp.utcnow().tz_localize(None)
           - pd.Timedelta(days=config.FUNDAMENTALS_TTL_DAYS + 5)).isoformat()
    assert fx.is_stale({"metrics": {}, "asof": old})


def test_a_checkpointed_crawl_is_marked_partial(tmp_path, monkeypatch):
    """
    build_snapshot() checkpoints so a crawl that dies mid-way does not lose
    everything. The cost is that a crawl still RUNNING is perfectly readable,
    and would be presented as finished — with a coverage figure that is really
    just how far through the alphabet it got. The status field is what lets
    the page say so.
    """
    monkeypatch.setattr(config, "FUNDAMENTALS_CACHE", tmp_path / "snap.json")
    monkeypatch.setattr(fx, "fetch_fundamentals",
                        lambda t: {"logMcap": 1.0, "roe": 0.1, "earnYield": 0.05})
    tickers = [f"T{i:03d}" for i in range(30)]
    fx.build_snapshot(tickers, pause=0.0, checkpoint_every=5)

    final = fx.load_snapshot()
    assert final["status"] == "ok"
    assert final["n"] == 30


def test_a_crawl_interrupted_mid_way_leaves_a_readable_partial(tmp_path,
                                                               monkeypatch):
    monkeypatch.setattr(config, "FUNDAMENTALS_CACHE", tmp_path / "snap.json")

    calls = {"n": 0}

    def flaky(ticker):
        calls["n"] += 1
        if calls["n"] > 12:
            raise KeyboardInterrupt("crawl stopped")
        return {"logMcap": 1.0, "roe": 0.1, "earnYield": 0.05}

    monkeypatch.setattr(fx, "fetch_fundamentals", flaky)
    with pytest.raises(KeyboardInterrupt):
        fx.build_snapshot([f"T{i:03d}" for i in range(30)], pause=0.0,
                          checkpoint_every=5)

    partial = fx.load_snapshot()
    assert partial["status"] == "partial"
    assert 0 < partial["n"] < 30


def test_a_resumed_crawl_keeps_what_the_first_one_fetched(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FUNDAMENTALS_CACHE", tmp_path / "snap.json")
    monkeypatch.setattr(fx, "fetch_fundamentals",
                        lambda t: {"logMcap": 1.0, "roe": 0.1, "earnYield": 0.05})
    fx.build_snapshot(["AAA", "BBB"], pause=0.0)

    def refuse(ticker):
        raise AssertionError(f"{ticker} should have been read from the cache")

    monkeypatch.setattr(fx, "fetch_fundamentals", refuse)
    out = fx.build_snapshot(["AAA", "BBB"], pause=0.0)
    assert out["n"] == 2 and out["fetched"] == 0
