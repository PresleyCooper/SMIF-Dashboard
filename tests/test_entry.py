"""
Entry validation, and the end-to-end scenarios the brief asks for: add,
import, de-duplication, a sell, a split, a spin-off and a new bond fund, each
landing in the right sleeve.

The validation tests all guard the same thing — that a bad row is refused at
the gate rather than silently producing a plausible number downstream. An
oversold position does not raise; it goes negative, gets priced, and subtracts
from the equity sleeve for the rest of the fund's history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import FakeStore, flat_series, ramp_series, tx_frame
from smif import config, entry, nav, store, validate


def row(**kw) -> dict:
    base = {"Trade Date": pd.Timestamp("2026-10-01"), "Activity": "Buy",
            "Description": "Apple Inc", "Symbol": "AAPL", "Quantity": 10.0,
            "Amount": 3000.0, "Notes": ""}
    base.update(kw)
    return base


KNOWN = {"AAPL", "MSFT", "IEF", "CASH", "SNOXX"}
HELD = pd.Series({"AAPL": 116.0, "MSFT": 67.0})
TODAY = pd.Timestamp("2026-11-06")


def check(**kw) -> list[entry.Issue]:
    r = kw.pop("row", None) or row(**kw)
    return entry.validate_row(r, holdings=HELD, known_symbols=KNOWN,
                              today=TODAY, **{k: v for k, v in kw.items()
                                              if k in ("new_symbol_confirmed",)})


def errors(issues) -> list[str]:
    return [i.message for i in entry.blocking(issues)]


# ── the happy path ────────────────────────────────────────────────────────────

def test_a_well_formed_buy_passes():
    assert entry.blocking(check()) == []


def test_a_well_formed_sell_within_the_position_passes():
    assert entry.blocking(check(row=row(Activity="Sell", Quantity=16.0,
                                        Amount=5600.0))) == []


# ── the rule that matters most ────────────────────────────────────────────────

def test_a_sell_cannot_exceed_the_shares_held():
    issues = check(row=row(Activity="Sell", Quantity=200.0, Amount=70000.0))
    assert any("holds 116" in m for m in errors(issues))


def test_selling_exactly_the_whole_position_is_allowed():
    assert entry.blocking(check(row=row(Activity="Sell", Quantity=116.0,
                                        Amount=40000.0))) == []


def test_a_debit_security_cannot_exceed_the_shares_held_either():
    """A spin-off's debit leg is as capable of going negative as a sell."""
    issues = check(row=row(Activity="Debit Security", Quantity=500.0,
                           Amount=1000.0))
    assert any("holds 116" in m for m in errors(issues))


def test_selling_a_name_the_fund_does_not_hold_is_refused():
    issues = check(row=row(Activity="Sell", Symbol="MSFT", Quantity=100.0,
                           Amount=50000.0))
    assert any("holds 67" in m for m in errors(issues))


# ── buys ──────────────────────────────────────────────────────────────────────

def test_a_buy_needs_a_quantity():
    assert any("needs a quantity" in m
               for m in errors(check(row=row(Quantity=None))))


def test_a_buy_needs_an_amount():
    assert any("needs an amount" in m
               for m in errors(check(row=row(Amount=None))))


def test_a_negative_quantity_is_refused():
    assert any("always positive" in m
               for m in errors(check(row=row(Quantity=-5.0))))


def test_a_negative_amount_is_refused_on_a_buy():
    assert any("always positive" in m
               for m in errors(check(row=row(Amount=-100.0))))


def test_a_negative_expense_is_allowed_as_a_reversal():
    issues = check(row=row(Activity="Expense (Other Expense)", Quantity=None,
                           Amount=-9.0))
    assert entry.blocking(issues) == []


# ── symbols ───────────────────────────────────────────────────────────────────

def test_an_unknown_symbol_is_refused_until_confirmed():
    issues = check(row=row(Symbol="ZZZZ"))
    assert any("never appeared" in m for m in errors(issues))


def test_a_confirmed_new_symbol_passes_with_a_warning():
    issues = entry.validate_row(row(Symbol="ZZZZ"), holdings=HELD,
                                known_symbols=KNOWN, today=TODAY,
                                new_symbol_confirmed=True)
    assert entry.blocking(issues) == []
    assert any("new to the ledger" in i.message for i in entry.warnings(issues))


def test_a_buy_with_no_symbol_is_refused():
    assert any("needs a symbol" in m for m in errors(check(row=row(Symbol=""))))


def test_an_expense_with_no_symbol_is_only_a_warning():
    """Foreign-tax rows genuinely arrive without one and are attributed
    through their description."""
    issues = check(row=row(Activity="Expense (Foreign Tax Paid)", Symbol="",
                           Quantity=None, Amount=3.46))
    assert entry.blocking(issues) == []
    assert any("Description" in i.message for i in entry.warnings(issues))


def test_a_deposit_carrying_a_ticker_is_a_warning_not_an_error():
    issues = check(row=row(Activity="Deposit", Symbol="AAPL", Quantity=None,
                           Amount=1000.0))
    assert entry.blocking(issues) == []
    assert any("normally carries CASH" in i.message
               for i in entry.warnings(issues))


# ── dates ─────────────────────────────────────────────────────────────────────

def test_a_future_trade_date_is_refused():
    future = row(**{"Trade Date": pd.Timestamp("2027-01-01")})
    assert any("in the future" in m for m in errors(check(row=future)))


def test_a_missing_trade_date_is_refused():
    assert any("trade date is required" in m
               for m in errors(check(row=row(**{"Trade Date": None}))))


def test_a_date_before_inception_is_a_warning():
    issues = check(row=row(**{"Trade Date": pd.Timestamp("2008-01-01")}))
    assert entry.blocking(issues) == []
    assert any("first deposit" in i.message for i in entry.warnings(issues))


# ── activities ────────────────────────────────────────────────────────────────

def test_an_unknown_activity_is_refused():
    assert any("not one of the Schwab" in m
               for m in errors(check(row=row(Activity="Teleport"))))


def test_a_security_transfer_warns_that_amount_is_cost_basis():
    issues = check(row=row(Activity="Credit Security", Quantity=5.0,
                           Amount=1234.0))
    assert entry.blocking(issues) == []
    assert any("cost basis" in i.message for i in entry.warnings(issues))


def test_a_reinvested_dividend_does_not_need_an_amount():
    issues = check(row=row(Activity="Income (Reinvested Qualified Dividend)",
                           Quantity=0.3, Amount=None))
    assert entry.blocking(issues) == []


def test_a_duplicate_row_warns_rather_than_blocks(tmp_db):
    store.add_transactions(tx_frame([
        {"date": "2026-10-01", "activity": "Buy", "symbol": "AAPL",
         "qty": 10.0, "amount": 3000.0}]), path=tmp_db)
    issues = entry.validate_row(row(), holdings=HELD, known_symbols=KNOWN,
                                today=TODAY,
                                existing_keys=store.existing_keys(tmp_db))
    assert entry.blocking(issues) == []
    assert any("already holds a row" in i.message for i in entry.warnings(issues))


# ── the effect preview ────────────────────────────────────────────────────────

def test_the_preview_shows_what_a_buy_will_do():
    e = entry.preview_effect(row())
    assert e["shares"] == 10.0 and e["cash"] == -3000.0
    assert e["net_purchase"] == 3000.0 and e["sleeve"] == "equity"


def test_the_preview_shows_a_split_as_shares_only():
    e = entry.preview_effect(row(Activity="Split", Quantity=116.0, Amount=None))
    assert e["shares"] == 116.0 and e["cash"] == 0.0


def test_the_preview_puts_a_bond_fund_in_fixed_income():
    e = entry.preview_effect(row(Symbol="IEF", Quantity=10.0, Amount=950.0))
    assert e["sleeve"] == "fixed_income"


# ── classification ────────────────────────────────────────────────────────────

def test_a_configured_bond_fund_needs_no_classification():
    assert not entry.needs_classification("IEF")


def test_cash_needs_no_classification():
    assert not entry.needs_classification("SNOXX")


def test_a_brand_new_symbol_needs_classification():
    assert entry.needs_classification("NEWCO")


def test_an_already_classified_symbol_does_not_ask_again():
    assert not entry.needs_classification("NEWCO", {"NEWCO": "fixed_income"})


# ── end-to-end: a saved row changes the right sleeve ─────────────────────────

@pytest.fixture
def seeded(tmp_db, calendar):
    d = [c.strftime("%Y-%m-%d") for c in calendar]
    store.add_transactions(tx_frame([
        {"date": d[0], "activity": "Deposit", "symbol": "CASH",
         "amount": 100_000.0},
        {"date": d[0], "activity": "Buy", "symbol": "AAA", "qty": 100.0,
         "amount": 5_000.0},
    ]), path=tmp_db)
    store_px = FakeStore({
        "SPY": ramp_series(calendar, 300.0, 1.0),
        "AGG": flat_series(calendar, 100.0),
        "AAA": flat_series(calendar, 50.0),
        "BBB": flat_series(calendar, 20.0),
        "NEWBOND": flat_series(calendar, 10.0),
    })
    return tmp_db, store_px, d


def build(db, px, overrides=None):
    return nav.build_book(store.read_transactions(db), px,
                          overrides=overrides or store.symbol_classes(db))


def test_adding_a_sell_reduces_the_equity_sleeve(seeded):
    db, px, d = seeded
    before = build(db, px).sleeve_value["equity"].iloc[-1]
    store.add_transactions(tx_frame([
        {"date": d[5], "activity": "Sell", "symbol": "AAA", "qty": 40.0,
         "amount": 2_000.0}]), actor="pat", path=db)
    after = build(db, px)
    assert after.sleeve_value["equity"].iloc[-1] == pytest.approx(before - 2_000.0)
    assert after.shares["AAA"].iloc[-1] == pytest.approx(60.0)


def test_adding_a_split_doubles_shares_without_touching_cash(seeded):
    db, px, d = seeded
    cash_before = build(db, px).cash.iloc[-1]
    store.add_transactions(tx_frame([
        {"date": d[5], "activity": "Split", "symbol": "AAA", "qty": 100.0}]),
        path=db)
    after = build(db, px)
    assert after.shares["AAA"].iloc[-1] == pytest.approx(200.0)
    assert after.cash.iloc[-1] == pytest.approx(cash_before)


def test_adding_a_spinoff_creates_the_child_in_the_equity_sleeve(seeded):
    db, px, d = seeded
    store.add_transactions(tx_frame([
        {"date": d[5], "activity": "Debit Security", "symbol": "AAA",
         "qty": 100.0, "amount": 5_000.0},
        {"date": d[5], "activity": "Credit Security", "symbol": "AAA",
         "qty": 100.0, "amount": 4_000.0},
        {"date": d[5], "activity": "Credit Security", "symbol": "BBB",
         "qty": 25.0, "amount": 1_000.0},
    ]), path=db)
    after = build(db, px)
    assert after.shares["AAA"].iloc[-1] == pytest.approx(100.0)
    assert after.shares["BBB"].iloc[-1] == pytest.approx(25.0)
    assert set(after.sleeve_members["equity"]) == {"AAA", "BBB"}


def test_a_new_bond_fund_lands_in_fixed_income_once_classified(seeded):
    db, px, d = seeded
    store.add_transactions(tx_frame([
        {"date": d[4], "activity": "Buy", "symbol": "NEWBOND", "qty": 100.0,
         "amount": 1_000.0}]), path=db)

    before = build(db, px)
    assert "NEWBOND" in before.sleeve_members["equity"]      # the default

    store.classify_symbol("NEWBOND", "fixed_income", actor="pat", path=db)
    after = build(db, px)
    assert "NEWBOND" in after.sleeve_members["fixed_income"]
    assert "NEWBOND" not in after.sleeve_members["equity"]
    assert after.sleeve_value["fixed_income"].iloc[-1] == pytest.approx(1_000.0)


def test_an_import_of_an_overlapping_export_only_adds_the_new_rows(seeded):
    db, px, d = seeded
    export = tx_frame([
        {"date": d[0], "activity": "Deposit", "symbol": "CASH",
         "amount": 100_000.0},
        {"date": d[0], "activity": "Buy", "symbol": "AAA", "qty": 100.0,
         "amount": 5_000.0},
        {"date": d[6], "activity": "Income (Qualified Dividend)",
         "symbol": "AAA", "amount": 30.0},
    ])
    preview = store.preview_import(export, path=db)
    assert list(preview["status"]) == ["already in ledger", "already in ledger",
                                       "new"]
    out = store.add_transactions(export, source="import:x.xlsx", path=db)
    assert out["added"] == 1 and out["skipped"] == 2
    assert build(db, px).sleeve_income["equity"].sum() == pytest.approx(30.0)


def test_deleting_a_row_removes_its_effect(seeded):
    db, px, d = seeded
    ids = store.add_transactions(tx_frame([
        {"date": d[5], "activity": "Sell", "symbol": "AAA", "qty": 40.0,
         "amount": 2_000.0}]), path=db)["added_ids"]
    assert build(db, px).shares["AAA"].iloc[-1] == pytest.approx(60.0)
    store.delete_transaction(ids[0], actor="pat", reason="keyed twice", path=db)
    assert build(db, px).shares["AAA"].iloc[-1] == pytest.approx(100.0)


# ── uploaded snapshot reconciliation ─────────────────────────────────────────

def test_a_snapshot_reconciles_position_by_position(seeded):
    db, px, _d = seeded
    book = build(db, px)
    snap = pd.DataFrame({"symbol": ["AAA"], "quantity": [100.0],
                         "price": [50.0], "value": [5_000.0]})
    out = validate.reconcile_positions(book, snap)
    aaa = out.set_index("symbol").loc["AAA"]
    assert aaa["share_diff"] == pytest.approx(0.0)
    assert bool(aaa["matches"])


def test_a_position_missing_from_the_snapshot_is_kept_not_dropped(seeded):
    """The most important row in the table is the one only one side has."""
    db, px, _d = seeded
    book = build(db, px)
    out = validate.reconcile_positions(
        book, pd.DataFrame({"symbol": ["ZZZ"], "quantity": [5.0],
                            "price": [1.0], "value": [5.0]}))
    only = out.set_index("symbol")["only_in"]
    assert only.get("AAA") == "rebuild"
    assert only.get("ZZZ") == "snapshot"


def test_a_share_difference_is_reported(seeded):
    db, px, _d = seeded
    book = build(db, px)
    out = validate.reconcile_positions(
        book, pd.DataFrame({"symbol": ["AAA"], "quantity": [90.0],
                            "price": [50.0], "value": [4_500.0]}))
    aaa = out.set_index("symbol").loc["AAA"]
    assert aaa["share_diff"] == pytest.approx(10.0)
    assert not bool(aaa["matches"])


def test_column_names_are_matched_case_insensitively(tmp_path):
    p = tmp_path / "positions.csv"
    pd.DataFrame({"Ticker": ["AAA"], "Shares": [10], "Market Value": [500]}
                 ).to_csv(p, index=False)
    out = validate.read_position_snapshot(p)
    assert list(out["symbol"]) == ["AAA"]
    assert out["quantity"].iloc[0] == 10


def test_subtotal_rows_are_not_treated_as_positions(tmp_path):
    p = tmp_path / "positions.csv"
    pd.DataFrame({"Symbol": ["AAA", "Total", "SUBTOTAL"],
                  "Quantity": [10, 0, 0], "Value": [500, 500, 500]}
                 ).to_csv(p, index=False)
    out = validate.read_position_snapshot(p)
    assert list(out["symbol"]) == ["AAA"]


def test_a_file_with_no_symbol_column_is_refused_clearly(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"Thing": ["AAA"], "Quantity": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="No symbol column"):
        validate.read_position_snapshot(p)


def test_a_cash_line_is_not_reported_as_matching_on_share_count_alone(seeded):
    """
    Cash has no share count on either side, so a share-only comparison passes
    it however far apart the dollar figures are — and cash is exactly where
    this fund's known discrepancy lives.
    """
    db, px, _d = seeded
    book = build(db, px)
    snap = pd.DataFrame({"symbol": ["AAA", "CASH"], "quantity": [100.0, np.nan],
                         "price": [50.0, np.nan], "value": [5_000.0, 1_000.0]})
    out = validate.reconcile_positions(book, snap).set_index("symbol")
    assert not bool(out.loc["CASH", "matches"])
    assert out.loc["CASH", "value_diff"] == pytest.approx(95_000.0 - 1_000.0)


def test_the_sweep_balance_appears_as_a_cash_position(seeded):
    """A custodian reports cash as a position; the engine holds it as a
    separate balance. The comparison has to put them side by side."""
    db, px, _d = seeded
    book = build(db, px)
    out = validate.reconcile_positions(
        book, pd.DataFrame({"symbol": ["AAA"], "quantity": [100.0],
                            "price": [50.0], "value": [5_000.0]}))
    cash = out.set_index("symbol").loc["CASH"]
    assert cash["rebuilt_value"] == pytest.approx(95_000.0)
    assert cash["only_in"] == "rebuild"


def test_a_value_difference_alone_fails_the_match(seeded):
    db, px, _d = seeded
    book = build(db, px)
    out = validate.reconcile_positions(
        book, pd.DataFrame({"symbol": ["AAA"], "quantity": [100.0],
                            "price": [60.0], "value": [6_000.0]}))
    aaa = out.set_index("symbol").loc["AAA"]
    assert aaa["share_diff"] == pytest.approx(0.0)
    assert not bool(aaa["matches"])


def test_a_few_cents_on_a_large_position_is_not_a_mismatch(seeded):
    """The custodian prices off a different feed. Flagging an 11-cent
    disagreement on a $16,000 position buries the lines that matter."""
    db, px, _d = seeded
    book = build(db, px)
    out = validate.reconcile_positions(
        book, pd.DataFrame({"symbol": ["AAA"], "quantity": [100.0],
                            "price": [50.0], "value": [5_000.11]}))
    assert bool(out.set_index("symbol").loc["AAA", "matches"])


def test_a_dollar_floor_stops_a_small_position_passing_on_a_percentage(seeded):
    db, px, _d = seeded
    book = build(db, px)
    out = validate.reconcile_positions(
        book, pd.DataFrame({"symbol": ["AAA"], "quantity": [100.0],
                            "price": [50.0], "value": [4_990.0]}))
    assert not bool(out.set_index("symbol").loc["AAA", "matches"])
