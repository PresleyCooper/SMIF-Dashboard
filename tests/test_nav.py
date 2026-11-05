"""Sleeve assignment and the daily book, on a ledger small enough to verify by hand."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import FakeStore, flat_series, ramp_series, tx_frame
from smif import config, nav, returns


def test_cash_like_instruments_are_the_cash_sleeve():
    assert nav.sleeve_of("CASH") == "cash"
    assert nav.sleeve_of("SNOXX") == "cash"


def test_the_configured_bond_funds_are_fixed_income():
    for t in config.FIXED_INCOME:
        assert nav.sleeve_of(t) == "fixed_income"


def test_anything_else_defaults_to_equity():
    assert nav.sleeve_of("AAPL") == "equity"
    assert nav.sleeve_of("EQIX") == "equity"      # a REIT is still the equity sleeve


def test_a_new_symbol_can_be_classified_without_touching_the_code():
    assert nav.sleeve_of("NEWBOND") == "equity"
    assert nav.sleeve_of("NEWBOND", {"NEWBOND": "fixed_income"}) == "fixed_income"


def test_config_wins_over_a_stale_classification():
    """A bond fund promoted into the config file must not stay mis-sleeved by
    an old row in the database."""
    assert nav.sleeve_of("IEF", {"IEF": "equity"}) == "fixed_income"


# ── the book ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_book(fake_store, calendar):
    d = [c.strftime("%Y-%m-%d") for c in calendar]
    tx = tx_frame([
        {"date": d[0], "activity": "Deposit", "symbol": "CASH", "amount": 100_000.0},
        {"date": d[0], "activity": "Buy", "symbol": "AAA", "qty": 100.0,
         "amount": 5_000.0},
        {"date": d[1], "activity": "Buy", "symbol": "IEF", "qty": 10.0,
         "amount": 950.0},
        {"date": d[2], "activity": "Buy", "symbol": "NEWBOND", "qty": 100.0,
         "amount": 1_000.0},
        {"date": d[3], "activity": "Income (Qualified Dividend)", "symbol": "AAA",
         "amount": 10.0},
        {"date": d[4], "activity": "Buy", "symbol": "SNOXX", "qty": 1_000.0,
         "amount": 1_000.0},
        {"date": d[5], "activity": "Income (Dividend)", "symbol": "SNOXX",
         "amount": 5.0},
    ])
    book = nav.build_book(tx, fake_store, overrides={"NEWBOND": "fixed_income"})
    return tx, book


def test_shares_accumulate_and_hold_until_changed(simple_book):
    _tx, book = simple_book
    assert book.shares["AAA"].iloc[0] == 100.0
    assert book.shares["AAA"].iloc[-1] == 100.0
    assert book.shares["IEF"].iloc[0] == 0.0       # bought on day two
    assert book.shares["IEF"].iloc[1] == 10.0


def test_a_classified_bond_fund_lands_in_the_fixed_income_sleeve(simple_book):
    _tx, book = simple_book
    assert set(book.sleeve_members["fixed_income"]) == {"IEF", "NEWBOND"}
    assert book.sleeve_members["equity"] == ["AAA"]
    assert "SNOXX" in book.sleeve_members["cash"]


def test_sleeve_values_add_up_to_the_total(simple_book):
    _tx, book = simple_book
    parts = sum(book.sleeve_value[s] for s in config.SLEEVES)
    pd.testing.assert_series_equal(parts, book.total, check_names=False)


def test_the_cash_sleeve_holds_the_sweep_plus_the_money_market(simple_book):
    _tx, book = simple_book
    last = book.days[-1]
    # 100,000 in, 5,000 + 950 + 1,000 out, 10 + 5 of income, 1,000 into SNOXX
    assert book.cash.loc[last] == pytest.approx(92_065.0)
    assert book.sleeve_value["cash"].loc[last] == pytest.approx(93_065.0)


def test_external_flows_are_only_deposits_and_withdrawals(simple_book):
    _tx, book = simple_book
    assert book.flows.sum() == pytest.approx(100_000.0)
    assert book.flows.iloc[1:].sum() == pytest.approx(0.0)


def test_a_sleeve_opens_at_zero_return_on_its_first_day(simple_book):
    _tx, book = simple_book
    r, start = returns.sleeve_daily_returns(book, "equity")
    assert start == book.days[0]
    assert r.loc[start] == pytest.approx(0.0)      # bought at 50, closed at 50


def test_a_dividend_is_a_sleeve_return_even_though_the_cash_left(simple_book):
    """The equity sleeve cannot see the cash pocket, so a $10 dividend on a
    $5,000 book has to read as +0.20% there."""
    _tx, book = simple_book
    r, _ = returns.sleeve_daily_returns(book, "equity")
    assert r.iloc[3] == pytest.approx(10.0 / 5_000.0)


def test_the_same_dividend_is_not_a_total_portfolio_return_twice(simple_book):
    """At fund level the dividend only shows up once — as cash arriving."""
    _tx, book = simple_book
    r = returns.total_daily_returns(book)
    prev = book.total.iloc[2]
    assert r.iloc[3] == pytest.approx(10.0 / prev)


def test_buying_the_money_market_is_not_a_cash_sleeve_return(simple_book):
    """$1,000 of sweep becoming $1,000 of SNOXX is a transfer inside the
    sleeve, not a gain."""
    _tx, book = simple_book
    r, _ = returns.sleeve_daily_returns(book, "cash")
    assert r.iloc[4] == pytest.approx(0.0, abs=1e-12)


def test_money_market_interest_is_a_cash_sleeve_return(simple_book):
    _tx, book = simple_book
    r, _ = returns.sleeve_daily_returns(book, "cash")
    prev = book.sleeve_value["cash"].iloc[4]
    assert r.iloc[5] == pytest.approx(5.0 / prev)


def test_a_purchase_is_not_a_sleeve_return(simple_book):
    """The fixed income sleeve went from nothing to $950 on day two by buying,
    which must read as 0%, not as infinity."""
    _tx, book = simple_book
    r, start = returns.sleeve_daily_returns(book, "fixed_income")
    assert start == book.days[1]
    assert r.loc[start] == pytest.approx(0.0)
    assert r.iloc[2] == pytest.approx(0.0)         # NEWBOND added, no price move


def test_a_deposit_is_not_a_total_portfolio_return(simple_book):
    _tx, book = simple_book
    r = returns.total_daily_returns(book)
    assert r.iloc[0] == pytest.approx(0.0)


# ── the split, which is where shares and prices must stay in step ─────────────

def test_a_split_leaves_the_position_value_and_the_return_unchanged(calendar):
    """
    Yahoo's close halves across the split AND is restated backwards. The
    ledger's Split row doubles the share count on the same day. If either half
    of that is wrong the position value jumps 2x on a day nothing happened.
    """
    d = [c.strftime("%Y-%m-%d") for c in calendar]
    split_day = calendar[5]
    store = FakeStore({
        "SPY": ramp_series(calendar, 300.0, 1.0),
        "AAA": flat_series(calendar, 50.0, splits={split_day: 2.0}),
    })
    tx = tx_frame([
        {"date": d[0], "activity": "Deposit", "symbol": "CASH", "amount": 20_000.0},
        {"date": d[0], "activity": "Buy", "symbol": "AAA", "qty": 100.0,
         "amount": 10_000.0},
        {"date": d[5], "activity": "Split", "symbol": "AAA", "qty": 100.0},
    ])
    book = nav.build_book(tx, store)

    assert book.shares["AAA"].iloc[4] == 100.0
    assert book.shares["AAA"].iloc[5] == 200.0
    assert book.prices["AAA"].iloc[4] == pytest.approx(100.0)   # un-adjusted
    assert book.prices["AAA"].iloc[5] == pytest.approx(50.0)
    assert book.mv["AAA"].iloc[4] == pytest.approx(10_000.0)
    assert book.mv["AAA"].iloc[5] == pytest.approx(10_000.0)

    r, _ = returns.sleeve_daily_returns(book, "equity")
    assert r.iloc[5] == pytest.approx(0.0, abs=1e-12)


def test_a_spinoff_puts_the_new_ticker_in_the_right_sleeve(calendar):
    d = [c.strftime("%Y-%m-%d") for c in calendar]
    store = FakeStore({"SPY": ramp_series(calendar, 300.0, 1.0),
                       "AAA": flat_series(calendar, 50.0),
                       "BBB": flat_series(calendar, 20.0)})
    tx = tx_frame([
        {"date": d[0], "activity": "Deposit", "symbol": "CASH", "amount": 20_000.0},
        {"date": d[0], "activity": "Buy", "symbol": "AAA", "qty": 100.0,
         "amount": 5_000.0},
        {"date": d[4], "activity": "Debit Security", "symbol": "AAA",
         "qty": 100.0, "amount": 5_000.0},
        {"date": d[4], "activity": "Credit Security", "symbol": "AAA",
         "qty": 100.0, "amount": 4_000.0},
        {"date": d[4], "activity": "Credit Security", "symbol": "BBB",
         "qty": 25.0, "amount": 1_000.0},
    ])
    book = nav.build_book(tx, store)
    assert book.shares["AAA"].iloc[-1] == pytest.approx(100.0)
    assert book.shares["BBB"].iloc[-1] == pytest.approx(25.0)
    assert set(book.sleeve_members["equity"]) == {"AAA", "BBB"}
    # cost basis never became cash
    assert book.cash.iloc[-1] == pytest.approx(15_000.0)


def test_an_exited_position_leaves_no_float_ghost(calendar):
    d = [c.strftime("%Y-%m-%d") for c in calendar]
    store = FakeStore({"SPY": ramp_series(calendar, 300.0, 1.0),
                       "AAA": flat_series(calendar, 50.0)})
    tx = tx_frame([
        {"date": d[0], "activity": "Deposit", "symbol": "CASH", "amount": 10_000.0},
        {"date": d[0], "activity": "Buy", "symbol": "AAA", "qty": 33.3,
         "amount": 1_665.0},
        {"date": d[3], "activity": "Sell", "symbol": "AAA", "qty": 33.3,
         "amount": 1_665.0},
    ])
    book = nav.build_book(tx, store)
    assert book.shares["AAA"].iloc[-1] == 0.0
    assert book.holdings_on().index.tolist() == []


def test_a_sell_cannot_be_valued_before_the_shares_existed(calendar):
    """Snapping a weekend trade forward, never backward, is what guarantees it."""
    store = FakeStore({"SPY": ramp_series(calendar, 300.0, 1.0),
                       "AAA": flat_series(calendar, 50.0)})
    tx = tx_frame([
        {"date": "2020-01-02", "activity": "Deposit", "symbol": "CASH",
         "amount": 10_000.0},
        {"date": "2020-01-04", "activity": "Buy", "symbol": "AAA",   # a Saturday
         "qty": 10.0, "amount": 500.0},
    ])
    book = nav.build_book(tx, store)
    assert book.shares["AAA"].loc[pd.Timestamp("2020-01-03")] == 0.0
    assert book.shares["AAA"].loc[pd.Timestamp("2020-01-06")] == 10.0
