"""Return mechanics on synthetic data, then the published regression numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import FakeStore, flat_series, ramp_series, tx_frame
from smif import engine, returns


# ── mechanics ─────────────────────────────────────────────────────────────────

def test_chaining_compounds_and_starts_the_base_day_at_one():
    days = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=4))
    daily = pd.Series([0.5, 0.10, -0.05, 0.02], index=days)
    idx = returns.growth_index(daily, days[0])
    assert idx.iloc[0] == 1.0              # the base day's own return is discarded
    assert idx.iloc[-1] == pytest.approx(1.10 * 0.95 * 1.02)


def test_the_base_day_is_the_close_before_the_first_purchase():
    """Day one of a sleeve is a fill-to-close move; basing the index on the day
    before is what gives it somewhere to live."""
    days = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=5))
    assert returns.base_day(days, days[2]) == days[1]
    assert returns.base_day(days, days[0]) == days[0]
    assert returns.base_day(days, None) == days[0]


def test_a_period_base_is_the_last_close_on_or_before_its_start():
    """Not the nearest: a base one day late shortens the window and flatters
    the return whenever the market fell on the boundary."""
    days = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=600))
    idx = pd.Series(1.0, index=days)
    end = days[-1]
    base = returns.period_base(idx, "1Y", pd.DateOffset(years=1), end)
    assert base <= end - pd.DateOffset(years=1)
    assert days[days.get_loc(base) + 1] > end - pd.DateOffset(years=1)


def test_ytd_is_based_on_the_prior_year_end():
    days = pd.DatetimeIndex(pd.bdate_range("2025-11-01", "2026-03-01"))
    idx = pd.Series(1.0, index=days)
    base = returns.period_base(idx, "YTD", "ytd", pd.Timestamp("2026-03-01"))
    assert base == pd.Timestamp("2025-12-31")


def test_max_drawdown_is_measured_from_the_running_peak():
    idx = pd.Series([1.0, 1.5, 0.9, 1.2],
                    index=pd.bdate_range("2020-01-01", periods=4))
    assert returns.max_drawdown(idx) == pytest.approx(0.9 / 1.5 - 1)


def test_the_drawdown_window_is_the_peak_and_the_trough():
    days = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=4))
    idx = pd.Series([1.0, 1.5, 0.9, 1.2], index=days)
    peak, trough = returns.drawdown_window(idx)
    assert peak == days[1] and trough == days[2]


def test_volatility_is_annualized_from_daily_returns():
    daily = pd.Series([0.01, -0.01] * 50,
                      index=pd.bdate_range("2020-01-01", periods=100))
    assert returns.annualized_vol(daily) == pytest.approx(
        daily.std(ddof=1) * np.sqrt(252))


def test_short_periods_report_no_annualized_figure():
    days = pd.DatetimeIndex(pd.bdate_range("2026-01-01", periods=60))
    track = returns.Track(name="t", daily=pd.Series(0.0, index=days),
                          index=pd.Series(1.0, index=days),
                          value=pd.Series(1.0, index=days))
    stats = returns.period_stats(track, "YTD", "ytd", days[-1])
    assert stats["ann"] is None


def test_a_one_year_period_is_annualized_even_though_365_is_under_365_25():
    days = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=600))
    daily = pd.Series(0.0004, index=days)
    track = returns.Track(name="t", daily=daily,
                          index=returns.growth_index(daily, days[0]),
                          value=pd.Series(1.0, index=days))
    assert returns.period_stats(track, "1Y", pd.DateOffset(years=1),
                                days[-1])["ann"] is not None


def test_a_benchmark_is_rebased_to_the_tracks_own_base(calendar):
    d = [c.strftime("%Y-%m-%d") for c in calendar]
    store = FakeStore({"SPY": ramp_series(calendar, 300.0, 3.0),
                       "AAA": flat_series(calendar, 50.0)})
    from smif import nav
    tx = tx_frame([
        {"date": d[0], "activity": "Deposit", "symbol": "CASH", "amount": 10_000.0},
        {"date": d[2], "activity": "Buy", "symbol": "AAA", "qty": 100.0,
         "amount": 5_000.0},
    ])
    book = nav.build_book(tx, store)
    track = returns.build_track(book, "equity", store)
    assert track.base == calendar[1]
    assert track.benchmark.iloc[0] == pytest.approx(1.0)
    assert track.benchmark.index[0] == calendar[1]


def test_calendar_years_flag_the_partial_first_and_last_years():
    days = pd.DatetimeIndex(pd.bdate_range("2024-06-03", "2026-03-02"))
    daily = pd.Series(0.0002, index=days)
    track = returns.Track(name="t", daily=daily,
                          index=returns.growth_index(daily, days[0]),
                          value=pd.Series(1.0, index=days))
    cal = returns.calendar_years(track)
    assert list(cal["year"]) == [2024, 2025, 2026]
    assert list(cal["partial"]) == [True, False, True]


# ── the regression ────────────────────────────────────────────────────────────
#
# These are the numbers the repository publishes. They are pinned to the
# 2026-11-04 close, and the tolerance is 0.15 percentage points: wide enough to
# absorb Yahoo restating a dividend somewhere in seventeen years, far too tight
# to absorb a transaction rule being wrong.

EXPECTED = {
    #  period      equity   spy_eq   total   spy_total
    "All time":  (620.21,  804.41,  501.15,  863.68),
    "10Y":       (319.21,  330.55,  267.14,  330.55),
    "5Y":        ( 59.51,   75.59,   55.55,   75.59),
    "3Y":        ( 69.90,   82.92,   63.02,   82.92),
    "1Y":        ( 21.95,   14.79,   19.08,   14.79),
    "YTD":       ( 18.68,   13.33,   16.11,   13.33),
}
TOL = 0.15


@pytest.fixture(scope="module")
def table(analytics):
    from conftest import REGRESSION_END
    return engine.regression_table(analytics, end=REGRESSION_END).set_index("period")


@pytest.mark.parametrize("period", list(EXPECTED))
@pytest.mark.parametrize("column,slot", [("equity", 0), ("spy_vs_equity", 1),
                                         ("total", 2), ("spy_vs_total", 3)])
def test_regression(table, period, column, slot):
    assert table.loc[period, column] == pytest.approx(EXPECTED[period][slot],
                                                      abs=TOL)


def test_the_equity_sleeve_is_based_the_day_before_its_first_purchase(table):
    assert table.loc["All time", "equity_base"] == "2010-01-20"


def test_the_total_portfolio_is_based_on_the_first_deposit(table):
    assert table.loc["All time", "total_base"] == "2009-10-13"


def test_fixed_income_beats_agg_since_it_was_bought(analytics):
    fi = analytics.tracks["fixed_income"]
    assert fi.base == pd.Timestamp("2026-09-21")
    assert (fi.index.iloc[-1] - 1) * 100 == pytest.approx(0.54, abs=0.05)
    assert (fi.benchmark.iloc[-1] - 1) * 100 == pytest.approx(0.18, abs=0.05)


def test_every_sleeve_has_a_track(analytics):
    assert set(analytics.tracks) == {"total", "equity", "fixed_income", "cash"}


def test_the_cash_sleeve_only_moves_by_interest_and_fees(analytics):
    """
    Sweep cash and a money market have no price, so the sleeve can only move by
    what it earns and what is charged against it. A few days are slightly
    negative — an account fee debited to cash — and the largest of those is
    worth about a dollar. What would NOT be a fee is a transfer mistaken for a
    loss: moving $20,000 of cash into SNOXX would read as -100% on the day.
    """
    r = analytics.tracks["cash"].daily
    assert r.abs().max() < 0.01
    assert analytics.tracks["cash"].index.iloc[-1] > 1.0


def test_sleeve_values_reconcile_to_the_total_every_single_day(analytics):
    book = analytics.book
    parts = sum(book.sleeve_value[s] for s in ("equity", "fixed_income", "cash"))
    assert (parts - book.total).abs().max() < 1e-6
