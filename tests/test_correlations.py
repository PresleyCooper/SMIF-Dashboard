"""
Correlation inputs and statistics.

The test that matters most is the first one: correlations are built from
ADJUSTED closes. The valuation series deliberately un-does Yahoo's split
adjustment, and feeding that into a return calculation turns every split into a
crash nothing else joined — which pushes correlations toward zero and betas
toward nothing, silently and in the safe-looking direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import FakeStore
from smif import config, correlations as corr_mod


def days(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.bdate_range(start, periods=n))


def price_frame(idx, closes, adj=None, splits=None) -> pd.DataFrame:
    sp = pd.Series(0.0, index=idx)
    for d, f in (splits or {}).items():
        sp.loc[pd.Timestamp(d)] = f
    return pd.DataFrame({"close": closes,
                         "adj_close": adj if adj is not None else closes,
                         "split": sp}, index=idx)


# ── the split trap ────────────────────────────────────────────────────────────

def test_a_split_is_not_a_crash():
    """
    The real (un-adjusted) close halves across a 2-for-1 while the adjusted one
    does not move. Returns must come from the adjusted series.
    """
    idx = days(60)
    real = pd.Series(100.0, index=idx)
    real.iloc[30:] = 50.0                       # what a share actually traded at
    adj = pd.Series(50.0, index=idx)            # what a holder actually earned
    store = FakeStore({"SPY": price_frame(idx, pd.Series(300.0, index=idx)),
                       "AAA": price_frame(idx, real, adj=adj)})

    rets, dropped = corr_mod.price_returns(store, ["AAA"], idx[0], idx[-1], "Daily")
    assert dropped == {}
    assert rets["AAA"].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert (rets["AAA"] < -0.4).sum() == 0


def test_two_identical_series_correlate_at_one():
    idx = days(80)
    rng = np.random.default_rng(0)
    walk = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))),
                     index=idx)
    store = FakeStore({"AAA": price_frame(idx, walk),
                       "BBB": price_frame(idx, walk * 3.0)})
    rets, _ = corr_mod.price_returns(store, ["AAA", "BBB"], idx[0], idx[-1])
    c = corr_mod.correlation(rets)
    assert c.loc["AAA", "BBB"] == pytest.approx(1.0, abs=1e-9)


def test_a_mirrored_series_correlates_at_minus_one():
    idx = days(80)
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, len(idx))
    up = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    down = pd.Series(100 * np.exp(np.cumsum(-r)), index=idx)
    store = FakeStore({"AAA": price_frame(idx, up), "BBB": price_frame(idx, down)})
    rets, _ = corr_mod.price_returns(store, ["AAA", "BBB"], idx[0], idx[-1])
    assert corr_mod.correlation(rets).loc["AAA", "BBB"] < -0.97


# ── what gets excluded, and why ───────────────────────────────────────────────

def test_cash_like_names_are_excluded_with_a_reason():
    idx = days(60)
    store = FakeStore({"AAA": price_frame(idx, pd.Series(10.0, index=idx))})
    _rets, dropped = corr_mod.price_returns(store, ["SNOXX", "CASH"],
                                            idx[0], idx[-1])
    assert set(dropped) == {"SNOXX", "CASH"}
    assert all("cash-like" in v for v in dropped.values())


def test_interpolated_names_are_excluded():
    """An interpolated price is a straight line in log space: correlating it
    with anything measures the interpolation, not the company."""
    idx = days(60)
    store = FakeStore({"AAA": price_frame(idx, pd.Series(10.0, index=idx))})
    _rets, dropped = corr_mod.price_returns(store, ["QCOR"], idx[0], idx[-1])
    assert "interpolated" in dropped["QCOR"]


def test_a_name_with_too_little_history_is_dropped_not_padded():
    idx = days(60)
    short = days(5, "2024-03-01")
    store = FakeStore({"AAA": price_frame(idx, pd.Series(10.0, index=idx)),
                       "NEW": price_frame(short, pd.Series(5.0, index=short))})
    rets, dropped = corr_mod.price_returns(store, ["AAA", "NEW"], idx[0], idx[-1])
    assert "NEW" not in rets.columns
    assert "NEW" in dropped


# ── frequency ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("freq,expected", [("Daily", 399), ("Weekly", 80),
                                           ("Monthly", 18)])
def test_frequency_changes_the_number_of_observations(freq, expected):
    """Two years of business days, so even the monthly series clears its own
    minimum — a window too short leaves the column dropped and the length
    assertion passing against an empty frame."""
    idx = days(400)
    walk = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    store = FakeStore({"AAA": price_frame(idx, walk)})
    rets, dropped = corr_mod.price_returns(store, ["AAA"], idx[0], idx[-1], freq)
    assert list(rets.columns) == ["AAA"], dropped
    assert len(rets) == pytest.approx(expected, abs=2)


def test_a_short_window_drops_a_column_rather_than_reporting_on_two_points():
    idx = days(45)
    walk = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    store = FakeStore({"AAA": price_frame(idx, walk)})
    rets, dropped = corr_mod.price_returns(store, ["AAA"], idx[0], idx[-1],
                                           "Monthly")
    assert rets.empty or "AAA" not in rets.columns
    assert "AAA" in dropped


def test_a_monthly_return_compounds_its_days():
    """Chaining the monthly returns must land on the same level ratio as the
    month-end prices themselves — that is what "compounded" has to mean."""
    idx = days(400)
    walk = pd.Series(100 * (1.001 ** np.arange(len(idx))), index=idx)
    store = FakeStore({"AAA": price_frame(idx, walk)})
    monthly, _ = corr_mod.price_returns(store, ["AAA"], idx[0], idx[-1], "Monthly")

    month_ends = walk.resample("ME").last().dropna()
    expected = month_ends.iloc[-1] / month_ends.iloc[0]
    assert (1 + monthly["AAA"]).prod() == pytest.approx(expected, rel=1e-9)


def test_daily_returns_chain_to_the_full_window_ratio():
    idx = days(400)
    walk = pd.Series(100 * (1.001 ** np.arange(len(idx))), index=idx)
    store = FakeStore({"AAA": price_frame(idx, walk)})
    daily, _ = corr_mod.price_returns(store, ["AAA"], idx[0], idx[-1], "Daily")
    assert (1 + daily["AAA"]).prod() == pytest.approx(
        walk.iloc[-1] / walk.iloc[0], rel=1e-9)


# ── sectors ───────────────────────────────────────────────────────────────────

def test_a_sector_return_is_the_weighted_average_of_its_names():
    idx = days(5)
    rets = pd.DataFrame({"AAA": [0.10] * 5, "BBB": [0.00] * 5,
                         "CCC": [0.20] * 5}, index=idx)
    weights = pd.Series({"AAA": 0.30, "BBB": 0.10, "CCC": 0.60})
    sectors = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"}
    out = corr_mod.sector_returns(rets, weights, sectors)
    # Tech renormalises 0.30/0.10 to 0.75/0.25 -> 0.075
    assert out["Tech"].iloc[0] == pytest.approx(0.075)
    assert out["Energy"].iloc[0] == pytest.approx(0.20)


def test_a_sector_with_no_weight_is_left_out_rather_than_divided_by_zero():
    idx = days(3)
    rets = pd.DataFrame({"AAA": [0.1] * 3}, index=idx)
    out = corr_mod.sector_returns(rets, pd.Series({"AAA": 0.0}), {"AAA": "Tech"})
    assert out.empty or "Tech" not in out.columns


def test_names_without_a_sector_are_simply_absent():
    idx = days(3)
    rets = pd.DataFrame({"AAA": [0.1] * 3, "ZZZ": [0.2] * 3}, index=idx)
    out = corr_mod.sector_returns(rets, pd.Series({"AAA": 1.0, "ZZZ": 1.0}),
                                  {"AAA": "Tech"})
    assert list(out.columns) == ["Tech"]


# ── beta ──────────────────────────────────────────────────────────────────────

def test_beta_of_the_market_against_itself_is_one():
    idx = days(80)
    rng = np.random.default_rng(2)
    m = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    out = corr_mod.beta_table(pd.DataFrame({"MKT": m}), m)
    assert out.loc["MKT", "beta"] == pytest.approx(1.0)
    assert out.loc["MKT", "corr_spy"] == pytest.approx(1.0)
    assert out.loc["MKT", "r2"] == pytest.approx(1.0)


def test_a_doubled_market_has_a_beta_of_two():
    idx = days(80)
    rng = np.random.default_rng(3)
    m = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    out = corr_mod.beta_table(pd.DataFrame({"LEV": 2 * m}), m)
    assert out.loc["LEV", "beta"] == pytest.approx(2.0)
    assert out.loc["LEV", "r2"] == pytest.approx(1.0)


def test_an_uncorrelated_series_has_a_low_r_squared():
    idx = days(400)
    rng = np.random.default_rng(4)
    m = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    noise = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    out = corr_mod.beta_table(pd.DataFrame({"NOISE": noise}), m)
    assert abs(out.loc["NOISE", "r2"]) < 0.05


def test_too_few_overlapping_observations_give_nan_not_a_number():
    idx = days(10)
    m = pd.Series(np.linspace(0, 0.01, len(idx)), index=idx)
    out = corr_mod.beta_table(pd.DataFrame({"A": m}), m, min_obs=20)
    assert np.isnan(out.loc["A", "beta"])
    assert out.loc["A", "obs"] == 10


# ── summary statistics ────────────────────────────────────────────────────────

def test_average_pair_correlation_ignores_the_diagonal():
    c = pd.DataFrame([[1.0, 0.5, 0.1], [0.5, 1.0, 0.3], [0.1, 0.3, 1.0]],
                     index=list("ABC"), columns=list("ABC"))
    assert corr_mod.average_pair_correlation(c) == pytest.approx(0.3)


def test_effective_holdings_counts_independent_bets():
    """Four uncorrelated equal positions are four bets; four identical ones
    are a single bet wearing four names."""
    names = list("ABCD")
    w = pd.Series(0.25, index=names)
    independent = pd.DataFrame(np.eye(4), index=names, columns=names)
    identical = pd.DataFrame(np.ones((4, 4)), index=names, columns=names)
    assert corr_mod.effective_holdings(independent, w) == pytest.approx(4.0)
    assert corr_mod.effective_holdings(identical, w) == pytest.approx(1.0)


# ── clustering ────────────────────────────────────────────────────────────────

def test_clustering_puts_things_that_move_together_together():
    names = ["A1", "B1", "A2", "B2"]          # deliberately interleaved
    c = pd.DataFrame(
        [[1.0, 0.0, 0.9, 0.0],
         [0.0, 1.0, 0.0, 0.9],
         [0.9, 0.0, 1.0, 0.0],
         [0.0, 0.9, 0.0, 1.0]], index=names, columns=names)
    order = corr_mod.cluster_order(c)
    assert set(order) == set(names)
    assert abs(order.index("A1") - order.index("A2")) == 1
    assert abs(order.index("B1") - order.index("B2")) == 1


def test_clustering_is_actually_available():
    """cluster_order falls back to the input order without SciPy, so without
    this the clustering tests would pass on an unclustered heatmap."""
    assert corr_mod.clustering_available()


def test_clustering_a_tiny_matrix_is_a_no_op():
    c = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=list("AB"),
                     columns=list("AB"))
    assert corr_mod.cluster_order(c) == ["A", "B"]


def test_clustering_survives_missing_pairs():
    names = list("ABC")
    c = pd.DataFrame([[1.0, np.nan, 0.4], [np.nan, 1.0, 0.1], [0.4, 0.1, 1.0]],
                     index=names, columns=names)
    assert set(corr_mod.cluster_order(c)) == set(names)


# ── sleeve returns start when the sleeve does ────────────────────────────────

def _fake_track(name, daily, base):
    from smif import returns as R
    return R.Track(name=name, daily=daily,
                   index=R.growth_index(daily, base),
                   value=pd.Series(1.0, index=daily.index))


def test_a_sleeve_contributes_no_returns_before_it_was_bought():
    """
    sleeve_daily_returns() zero-pads back to inception so the series can be
    chained. Those zeros are not observations — correlating them says a sleeve
    bought six weeks ago is uncorrelated with everything, and reports a
    volatility diluted by hundreds of days of manufactured calm.
    """
    idx = days(300)
    rng = np.random.default_rng(7)
    old = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)

    young = pd.Series(0.0, index=idx)
    young.iloc[-40:] = rng.normal(0, 0.004, 40)
    base = idx[-41]

    tracks = {"total": _fake_track("Total portfolio", old, idx[0]),
              "fixed_income": _fake_track("Fixed income", young, base)}
    store = FakeStore({"SPY": price_frame(idx, pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx))})

    out = corr_mod.sleeve_returns(tracks, store, idx[0], idx[-1], "Daily")
    fi = out["Fixed income"]
    assert fi.notna().sum() == 40
    assert fi.loc[:base].notna().sum() == 0

    table = corr_mod.beta_table(out[["Fixed income"]], out["SPY"].dropna(),
                                min_obs=20)
    assert table.loc["Fixed income", "obs"] == 40


def test_a_young_sleeves_volatility_is_not_diluted_by_padding():
    idx = days(300)
    young = pd.Series(0.0, index=idx)
    young.iloc[-40:] = 0.01                      # a steady 1% a day while held
    base = idx[-41]
    tracks = {"fixed_income": _fake_track("Fixed income", young, base)}
    store = FakeStore({"SPY": price_frame(idx, pd.Series(100.0, index=idx))})

    out = corr_mod.sleeve_returns(tracks, store, idx[0], idx[-1], "Daily")
    held = out["Fixed income"].dropna()
    assert len(held) == 40
    assert held.std(ddof=1) == pytest.approx(0.0, abs=1e-12)   # not diluted


def test_a_missing_benchmark_leaves_a_blank_row_rather_than_crashing():
    """An absent ticker returns an empty Series with a RangeIndex; comparing
    that to a Timestamp raises instead of returning nothing."""
    idx = days(60)
    store = FakeStore({"SPY": price_frame(idx, pd.Series(100.0, index=idx))})
    out = corr_mod.benchmark_returns(store, idx[0], idx[-1])      # no AGG
    assert list(out.columns) == ["SPY"]
