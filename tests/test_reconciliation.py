"""
The reconciliation the first pass achieved, asserted.

These are the checks that would catch a rule quietly breaking: a wrong sign on
one activity moves a share count, and a wrong price moves a sleeve value, long
before either shows up as an implausible return.
"""

from __future__ import annotations

import pandas as pd
import pytest

from smif import config, validate


def test_every_current_holding_reconciles_to_the_workbook(validation):
    bad = validation["share_failures"]
    assert bad.empty, "share count mismatches:\n" + bad.to_string(index=False)


def test_the_only_differences_are_sub_share_remnants_of_exited_names(validation):
    sub = validation["shares"][validation["shares"]["sub_share"]]
    assert (sub["diff"].abs() < 1.0).all()
    assert (sub["workbook"] == 0).all()


def test_the_fixed_income_sleeve_matches_the_custodian_to_the_cent(validation):
    row = validation["sleeves"].set_index("sleeve").loc["Fixed income"]
    assert abs(row["diff"]) < 0.01


def test_the_equity_sleeve_matches_the_custodian_within_twelve_dollars(validation):
    """Not to the cent: the custodian prices a handful of names off a different
    feed, which is a $11 disagreement on a $529k book — 0.002%."""
    row = validation["sleeves"].set_index("sleeve").loc["Equity"]
    assert row["custodian"] == pytest.approx(config.CUSTODIAN_EQUITY, abs=0.01)
    assert abs(row["diff"]) < 12.0


def test_the_cash_gap_is_the_known_two_thousand_dollar_discrepancy(validation):
    """
    Asserted rather than corrected. The rebuild holds MORE cash than the
    custodian, which means the export is missing an outflow. Pinning the size
    is what makes it obvious if the gap ever changes — a new gap would be a new
    bug, and a closed gap would mean the missing row has been found.
    """
    gap = validation["cash"]
    assert gap["diff"] == pytest.approx(2_041.81, abs=5.0)
    assert gap["diff"] > 0
    assert abs(gap["share_of_total"]) < 0.005


def test_almost_every_fill_lands_within_five_percent_of_the_rebuilt_close(validation):
    """
    The real test of the split un-adjust. A still-adjusted series would put the
    ratio at the split factor — 4x on Apple, 10x on Nvidia — not within 5%.
    """
    s = validation["fill_summary"]
    assert s["checked"] > 300
    assert s["share"] >= config.FILL_MATCH_TARGET, (
        f"only {s['within_tolerance']}/{s['checked']} fills matched\n"
        + s["worst"].to_string(index=False))


def test_no_priced_name_is_left_without_a_price(validation):
    assert validation["fill_summary"]["no_price"] == 0


def test_the_interpolated_names_are_exactly_the_ones_declared(analytics):
    noted = {s for s, note in analytics.book.price_notes.items()
             if "interpolation" in note or "interpolated" in note}
    ever_held = {c for c in analytics.book.shares.columns
                 if analytics.book.shares[c].abs().sum() > 0}
    assert noted == (config.NO_YAHOO_DATA & ever_held)


def test_no_interpolated_name_is_a_material_part_of_todays_book(analytics):
    """
    Five of the interpolated names still carry a sub-share remnant (0.02 of a
    Diamond Offshore, -0.01 of a Synovus), worth about two dollars in total.
    They are real in the rebuild and the custodian has swept them, so the
    honest claim is not "none are held" but "none matter": their combined value
    is under a thousandth of a percent of the fund. If an interpolated name
    ever becomes a real position this fails, which is the point — a position
    whose price is a guess has to be flagged in the UI.
    """
    still_held = validate.interpolated_positions(analytics.book)
    total = float(analytics.book.total.iloc[-1])
    assert still_held["value"].abs().sum() / total < 1e-5


def test_the_fund_never_runs_a_negative_cash_balance(analytics):
    """A negative sweep balance would mean a purchase was recorded before the
    money that paid for it."""
    assert analytics.book.cash.min() > -1.0


def test_the_snapshot_date_and_subtotals_are_the_ones_the_config_expects(analytics):
    assert analytics.custodian_as_of == config.CUSTODIAN_AS_OF
    for key, expected in config.CUSTODIAN_SUBTOTALS.items():
        assert analytics.subtotals[key] == pytest.approx(expected, abs=0.01)
