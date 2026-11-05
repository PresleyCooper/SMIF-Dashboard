"""Every transaction rule, one test each, on a ledger small enough to read."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import tx_frame
from smif import config, ledger


def effects(rows):
    return ledger.build_events(tx_frame(rows))


def test_deposit_is_cash_and_an_external_flow():
    ev = effects([{"date": "2020-01-02", "activity": "Deposit",
                   "symbol": "CASH", "amount": 100_000.0}])
    assert ev.loc[0, "dcash"] == 100_000.0
    assert ev.loc[0, "flow"] == 100_000.0
    assert ev.loc[0, "dq"] == 0.0


def test_withdrawal_is_negative_cash_and_a_negative_flow():
    ev = effects([{"date": "2020-01-02", "activity": "Withdrawal",
                   "symbol": "CASH", "amount": 5_000.0}])
    assert ev.loc[0, "dcash"] == -5_000.0
    assert ev.loc[0, "flow"] == -5_000.0


def test_buy_adds_shares_and_spends_cash_but_is_not_a_flow():
    ev = effects([{"date": "2020-01-02", "activity": "Buy", "symbol": "AAA",
                   "qty": 10.0, "amount": 500.0}])
    assert ev.loc[0, "dq"] == 10.0
    assert ev.loc[0, "dcash"] == -500.0
    assert ev.loc[0, "flow"] == 0.0
    assert ev.loc[0, "netbuy"] == 500.0


def test_sell_removes_shares_and_returns_cash():
    ev = effects([{"date": "2020-01-02", "activity": "Sell", "symbol": "AAA",
                   "qty": 4.0, "amount": 220.0}])
    assert ev.loc[0, "dq"] == -4.0
    assert ev.loc[0, "dcash"] == 220.0
    assert ev.loc[0, "netbuy"] == -220.0


@pytest.mark.parametrize("activity", ["Buy", "Sell"])
def test_cash_sweeps_change_nothing(activity):
    """A Buy or Sell of CASH is the sweep talking to itself."""
    ev = effects([{"date": "2020-01-02", "activity": activity, "symbol": "CASH",
                   "qty": 500.0, "amount": 500.0}])
    assert ev.loc[0, ["dq", "dcash", "flow", "netbuy"]].abs().sum() == 0.0


@pytest.mark.parametrize("activity", [
    "Income (Dividend)", "Income (Qualified Dividend)",
    "Income (Non-Qualified Dividend)", "Income (Reinvested Interest)",
    "Income (Short Term Gain)", "Income (Long Term Gain)"])
def test_cash_income_credits_cash_and_counts_as_sleeve_income(activity):
    ev = effects([{"date": "2020-01-02", "activity": activity, "symbol": "AAA",
                   "amount": 42.0}])
    assert ev.loc[0, "dcash"] == 42.0
    assert ev.loc[0, "income"] == 42.0
    assert ev.loc[0, "dq"] == 0.0


def test_reinvested_qualified_dividend_buys_shares_with_no_cash():
    """The dividend never appeared as cash, so counting it as income would
    invent money the fund never had."""
    ev = effects([{"date": "2020-01-02",
                   "activity": "Income (Reinvested Qualified Dividend)",
                   "symbol": "AAA", "qty": 0.3, "amount": 15.0}])
    assert ev.loc[0, "dq"] == 0.3
    assert ev.loc[0, "dcash"] == 0.0
    assert ev.loc[0, "income"] == 0.0


def test_reinvested_dividend_on_a_security_buys_shares():
    ev = effects([{"date": "2020-01-02", "activity": "Income (Reinvested Dividend)",
                   "symbol": "AAA", "qty": 0.5, "amount": 25.0}])
    assert ev.loc[0, "dq"] == 0.5
    assert ev.loc[0, "dcash"] == 0.0


def test_reinvested_dividend_on_cash_is_interest():
    """Same activity name, opposite meaning: on the sweep it is cash interest."""
    ev = effects([{"date": "2020-01-02", "activity": "Income (Reinvested Dividend)",
                   "symbol": "CASH", "qty": 0.99, "amount": 0.99}])
    assert ev.loc[0, "dq"] == 0.0
    assert ev.loc[0, "dcash"] == 0.99
    assert ev.loc[0, "income"] == 0.99


def test_expense_debits_cash_and_reduces_sleeve_income():
    ev = effects([{"date": "2020-01-02", "activity": "Expense (Foreign Tax Paid)",
                   "symbol": "AAA", "amount": 3.46}])
    assert ev.loc[0, "dcash"] == -3.46
    assert ev.loc[0, "income"] == -3.46


def test_a_negative_expense_is_a_refund():
    """Keeping the sign math rather than taking an absolute value is what makes
    a fee reversal give the money back."""
    ev = effects([{"date": "2020-01-02", "activity": "Expense (Other Expense)",
                   "symbol": "AAA", "amount": -9.0}])
    assert ev.loc[0, "dcash"] == 9.0
    assert ev.loc[0, "income"] == 9.0


def test_return_of_principal_is_income():
    ev = effects([{"date": "2020-01-02", "activity": "Return of Principal",
                   "symbol": "AAA", "amount": 12.0}])
    assert ev.loc[0, "dcash"] == 12.0
    assert ev.loc[0, "income"] == 12.0


def test_split_adds_the_shares_it_records():
    """Schwab writes the shares ADDED, not the ratio: a 2-for-1 on 100 shares
    is Quantity 100, and reading it as a ratio would give 100x the position."""
    ev = effects([{"date": "2020-01-02", "activity": "Split", "symbol": "AAA",
                   "qty": 100.0, "amount": 0.0}])
    assert ev.loc[0, "dq"] == 100.0
    assert ev.loc[0, "dcash"] == 0.0


@pytest.mark.parametrize("activity,sign", [
    ("Credit Security", 1), ("Credit Security (Custodian Lot Adjustment)", 1),
    ("Debit Security", -1), ("Debit Security (Custodian Lot Adjustment)", -1)])
def test_security_credits_and_debits_are_share_only(activity, sign):
    """Their Amount is cost basis, not market value — using it as cash (or as a
    price) is the single most expensive mistake available here."""
    ev = effects([{"date": "2020-01-02", "activity": activity, "symbol": "AAA",
                   "qty": 7.0, "amount": 1_234.0}])
    assert ev.loc[0, "dq"] == sign * 7.0
    assert ev.loc[0, "dcash"] == 0.0
    assert ev.loc[0, "income"] == 0.0


def test_an_unknown_activity_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown Activity"):
        effects([{"date": "2020-01-02", "activity": "Teleport", "symbol": "AAA",
                  "qty": 1.0, "amount": 1.0}])


# ── the DRIP pair ─────────────────────────────────────────────────────────────

def test_a_drip_pair_nets_to_zero_cash_and_adds_shares():
    ev = effects([
        {"date": "2012-03-01", "activity": "Income (Qualified Dividend)",
         "symbol": "AAA", "amount": 25.0},
        {"date": "2012-03-01", "activity": "Buy", "symbol": "AAA",
         "qty": 0.5, "amount": 25.0},
    ])
    assert ev["dcash"].sum() == pytest.approx(0.0)
    assert ev["dq"].sum() == pytest.approx(0.5)
    # the dividend is still income, and the purchase is still a purchase: the
    # sleeve's return formula adds one back and takes the other out
    assert ev["income"].sum() == pytest.approx(25.0)
    assert ev["netbuy"].sum() == pytest.approx(25.0)


# ── symbol resolution ─────────────────────────────────────────────────────────

def test_gold_before_2019_is_randgold():
    assert ledger.resolve_symbol("GOLD", "2018-12-31", "Buy") == config.RANDGOLD_SYMBOL


def test_gold_after_the_exchange_is_barrick():
    assert ledger.resolve_symbol("GOLD", "2019-03-01", "Buy") == "GOLD"


def test_the_exchange_days_debit_is_the_randgold_side():
    """On 2019-01-02 both companies appear: the Debit retires Randgold and the
    Credit creates Barrick."""
    assert ledger.resolve_symbol("GOLD", "2019-01-02", "Debit Security") == \
        config.RANDGOLD_SYMBOL
    assert ledger.resolve_symbol("GOLD", "2019-01-02", "Credit Security") == "GOLD"


def test_symbols_are_upper_cased_and_blanks_become_none():
    assert ledger.resolve_symbol(" aapl ", "2020-01-02") == "AAPL"
    assert ledger.resolve_symbol(float("nan"), "2020-01-02") is None


def test_an_unsymbolled_expense_is_attributed_through_its_description():
    """Foreign-tax rows arrive with no Symbol. Without the description map the
    tax floats free and the equity sleeve's income is overstated."""
    ev = effects([
        {"date": "2020-01-02", "activity": "Income (Qualified Dividend)",
         "symbol": "ASML", "description": "Asml Hldgs Nv", "amount": 23.05},
        {"date": "2020-01-02", "activity": "Expense (Foreign Tax Paid)",
         "symbol": None, "description": "Asml Hldgs Nv", "amount": 3.46},
    ])
    assert list(ev["sym"]) == ["ASML", "ASML"]
    assert ev["income"].sum() == pytest.approx(19.59)


# ── a spin-off, end to end ────────────────────────────────────────────────────

def test_a_spinoff_moves_shares_between_tickers_and_leaves_cash_alone():
    ev = effects([
        {"date": "2014-05-30", "activity": "Buy", "symbol": "NOV",
         "qty": 100.0, "amount": 7_500.0},
        {"date": "2014-06-02", "activity": "Debit Security", "symbol": "NOV",
         "qty": 100.0, "amount": 7_500.0},
        {"date": "2014-06-02", "activity": "Credit Security", "symbol": "NOV",
         "qty": 100.0, "amount": 6_000.0},
        {"date": "2014-06-02", "activity": "Credit Security", "symbol": "DNOW",
         "qty": 25.0, "amount": 1_500.0},
    ])
    pos = ledger.final_positions(ev)
    assert pos["NOV"] == pytest.approx(100.0)
    assert pos["DNOW"] == pytest.approx(25.0)
    assert ev["dcash"].sum() == pytest.approx(-7_500.0)


def test_non_trading_days_snap_forward_to_the_next_session():
    days = pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=5))
    snapped = ledger.to_trading_day([pd.Timestamp("2020-01-04")], days)  # a Saturday
    assert snapped[0] == pd.Timestamp("2020-01-06")


def test_fill_anchors_ignore_fractional_drip_topups():
    tx = tx_frame([
        {"date": "2012-03-01", "activity": "Buy", "symbol": "AAA",
         "qty": 100.0, "amount": 5_000.0},
        {"date": "2012-06-01", "activity": "Buy", "symbol": "AAA",
         "qty": 0.3, "amount": 18.0},
    ])
    a = ledger.fill_anchors(tx, "AAA")
    assert list(a.index) == [pd.Timestamp("2012-03-01")]
    assert a.iloc[0] == pytest.approx(50.0)
