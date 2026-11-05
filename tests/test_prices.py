"""The split un-adjust, the ticker map and the interpolation fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import tx_frame
from smif import config, prices


def series(closes, splits=None):
    days = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=len(closes)))
    sp = pd.Series(0.0, index=days)
    for i, f in (splits or {}).items():
        sp.iloc[i] = f
    return pd.DataFrame({"close": closes, "adj_close": closes, "split": sp},
                        index=days)


# ── un-adjust ─────────────────────────────────────────────────────────────────

def test_with_no_splits_the_real_close_is_the_close():
    s = series([10.0, 11.0, 12.0])
    assert list(prices.unadjust(s)) == [10.0, 11.0, 12.0]


def test_a_split_lifts_every_earlier_close_and_leaves_later_ones_alone():
    """Yahoo shows 50 before and after a 2-for-1; the share really traded at
    100 before it, and the ledger's share count is the pre-split one."""
    s = series([50.0, 50.0, 50.0, 50.0], splits={2: 2.0})
    out = prices.unadjust(s)
    assert list(out) == [100.0, 100.0, 50.0, 50.0]


def test_the_split_date_itself_is_already_adjusted_and_is_not_doubled():
    s = series([50.0, 50.0], splits={1: 2.0})
    assert prices.unadjust(s).iloc[1] == 50.0


def test_successive_splits_compound():
    s = series([10.0] * 5, splits={2: 2.0, 4: 5.0})
    assert list(prices.unadjust(s)) == [100.0, 100.0, 50.0, 50.0, 10.0]


def test_a_fractional_split_recovers_the_pre_spinoff_price():
    """Yahoo records the Carrier/Otis separation as a 1.589 RTX split. Treating
    it like any other split is what turns today's RTX back into UTX."""
    s = series([100.0, 100.0], splits={1: 1.589})
    assert prices.unadjust(s).iloc[0] == pytest.approx(158.9)


def test_unadjust_of_an_empty_series_is_empty():
    assert prices.unadjust(pd.DataFrame(
        columns=["close", "adj_close", "split"])).empty


# ── ticker map ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("symbol,expected", [
    ("MHP", "SPGI"), ("MHFI", "SPGI"), ("NU", "ES"), ("UTX", "RTX"),
    ("RDS.B", "SHEL"), ("GOLD", "B"), ("AAPL", "AAPL"), ("BRK.B", "BRK-B")])
def test_ticker_history_is_followed_to_the_surviving_name(symbol, expected):
    assert prices.yahoo_ticker(symbol) == expected


@pytest.mark.parametrize("symbol", ["SNOXX", "CASH", config.RANDGOLD_SYMBOL,
                                    "QCOR", "XLNX"])
def test_names_yahoo_must_not_be_asked_about_return_none(symbol):
    assert prices.yahoo_ticker(symbol) is None


def test_randgold_pulls_barrick_as_well_as_itself():
    want = prices.required_tickers([config.RANDGOLD_SYMBOL])
    assert config.RANDGOLD_PROXY in want


def test_benchmarks_are_always_fetched():
    want = prices.required_tickers(["AAPL"])
    assert {"SPY", "AGG"} <= set(want)


# ── interpolation ─────────────────────────────────────────────────────────────

def test_interpolation_passes_exactly_through_its_anchors():
    days = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=11))
    anchors = pd.Series({days[0]: 10.0, days[10]: 20.0})
    out = prices.interpolated_series(anchors, days)
    assert out.iloc[0] == pytest.approx(10.0)
    assert out.iloc[-1] == pytest.approx(20.0)


def test_interpolation_is_log_linear_so_the_growth_rate_is_constant():
    """
    A calendar-day index, because the interpolation is linear in TIME, not in
    session count: over a weekend gap it compounds for three days, not one.
    That is the right behaviour — the price moved while the market was shut —
    and it is why a business-day index would show uneven step ratios.
    """
    days = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=5, freq="D"))
    anchors = pd.Series({days[0]: 10.0, days[4]: 160.0})
    out = prices.interpolated_series(anchors, days)
    ratios = (out / out.shift(1)).dropna()
    assert ratios.std() == pytest.approx(0.0, abs=1e-9)
    assert out.iloc[2] == pytest.approx(40.0)


def test_interpolation_compounds_over_a_closed_market():
    days = pd.DatetimeIndex(pd.bdate_range("2020-01-03", periods=3))  # Fri, Mon, Tue
    anchors = pd.Series({days[0]: 1.0, days[2]: 8.0})
    out = prices.interpolated_series(anchors, days)
    # Friday -> Monday is three calendar days of the four, so 2^3 of the 2^... no:
    # the total is 8 = 2^3 over 4 days, so Monday sits at 8^(3/4).
    assert out.iloc[1] == pytest.approx(8.0 ** 0.75)


def test_interpolation_is_held_flat_outside_the_anchor_range():
    days = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=7))
    anchors = pd.Series({days[2]: 10.0, days[4]: 20.0})
    out = prices.interpolated_series(anchors, days)
    assert out.iloc[0] == pytest.approx(10.0)
    assert out.iloc[-1] == pytest.approx(20.0)
    assert out.notna().all()


def test_a_single_anchor_gives_a_flat_series():
    days = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=4))
    out = prices.interpolated_series(pd.Series({days[1]: 7.0}), days)
    assert out.to_numpy() == pytest.approx(7.0)


def test_no_anchors_at_all_gives_nan_rather_than_a_made_up_price():
    days = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=3))
    assert prices.interpolated_series(pd.Series(dtype=float), days).isna().all()


def test_itt_educationals_zero_proceeds_sale_is_not_used_as_a_price():
    """A $0 write-off anchor would drag the whole interpolation to zero."""
    tx = tx_frame([
        {"date": "2014-01-02", "activity": "Buy", "symbol": "ESI",
         "qty": 100.0, "amount": 3_000.0},
        {"date": "2016-09-01", "activity": "Sell", "symbol": "ESI",
         "qty": 100.0, "amount": 0.0},
    ])
    a = prices._anchor_set(tx, "ESI")
    assert (a > 0).all()


def test_questcors_exit_is_valued_as_cash_plus_mallinckrodt_stock():
    tx = tx_frame([
        {"date": "2014-01-02", "activity": "Buy", "symbol": "QCOR",
         "qty": 30.0, "amount": 1_500.0},
        {"date": "2014-10-01", "activity": "Sell", "symbol": "MNK",
         "qty": 26.91, "amount": 2_691.0},
    ])
    a = prices._anchor_set(tx, "QCOR")
    q = config.QCOR_EXIT
    expected = (q["cash"] + q["mnk_shares"] * 100.0) / q["qcor_shares"]
    assert a.loc[q["date"]] == pytest.approx(expected)


def test_mallinckrodt_is_back_dated_to_the_questcor_exchange():
    tx = tx_frame([
        {"date": "2014-10-01", "activity": "Sell", "symbol": "MNK",
         "qty": 26.91, "amount": 2_691.0}])
    a = prices._anchor_set(tx, "MNK")
    assert config.QCOR_EXIT["date"] in a.index


# ── the matrix ────────────────────────────────────────────────────────────────

def test_the_money_market_is_priced_at_a_dollar(fake_store, calendar):
    P, notes = prices.build_price_matrix(["SNOXX"], calendar,
                                         tx_frame([]), fake_store)
    assert (P["SNOXX"] == 1.0).all()
    assert "money market" in notes["SNOXX"]


def test_randgold_is_priced_off_barrick_at_the_exchange_ratio(calendar):
    from conftest import FakeStore, flat_series
    store = FakeStore({"SPY": flat_series(calendar, 300.0),
                       "B": flat_series(calendar, 20.0)})
    P, notes = prices.build_price_matrix([config.RANDGOLD_SYMBOL], calendar,
                                         tx_frame([]), store)
    assert P[config.RANDGOLD_SYMBOL].iloc[0] == pytest.approx(
        20.0 * config.RANDGOLD_RATIO)
    assert "Barrick" in notes[config.RANDGOLD_SYMBOL]


def test_a_gap_before_the_first_print_is_backfilled_and_flagged(calendar):
    from conftest import FakeStore, flat_series
    late = flat_series(calendar[3:], 30.0)
    store = FakeStore({"SPY": flat_series(calendar, 300.0), "MRP": late})
    held = pd.DataFrame({"MRP": 10.0}, index=calendar)
    P, notes = prices.build_price_matrix(["MRP"], calendar, tx_frame([]),
                                          store, held=held)
    assert P["MRP"].notna().all()
    assert P["MRP"].iloc[0] == pytest.approx(30.0)
    assert "before" in notes["MRP"]


def test_a_renamed_ticker_is_noted_so_the_substitution_is_visible(fake_store,
                                                                   calendar):
    from conftest import FakeStore, flat_series
    store = FakeStore({"SPY": flat_series(calendar, 300.0),
                       "SPGI": flat_series(calendar, 400.0)})
    _P, notes = prices.build_price_matrix(["MHP"], calendar, tx_frame([]), store)
    assert "SPGI" in notes["MHP"]


def test_the_trading_calendar_comes_from_the_benchmarks_own_sessions(fake_store,
                                                                     calendar):
    assert list(prices.trading_days(fake_store)) == list(calendar)
