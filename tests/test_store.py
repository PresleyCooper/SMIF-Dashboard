"""The ledger store: adding, importing, de-duplicating, editing, auditing."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import tx_frame
from smif import config, store


def rows():
    return tx_frame([
        {"date": "2026-10-01", "activity": "Buy", "symbol": "AAA",
         "qty": 10.0, "amount": 500.0},
        {"date": "2026-10-02", "activity": "Income (Dividend)", "symbol": "AAA",
         "amount": 4.0},
    ])


def test_add_then_read_round_trips(tmp_db):
    store.add_transactions(rows(), actor="pat", path=tmp_db)
    df = store.read_transactions(tmp_db)
    assert len(df) == 2
    assert list(df["Activity"]) == ["Buy", "Income (Dividend)"]
    assert set(df["added_by"]) == {"pat"}


def test_rows_come_back_oldest_first_whatever_order_they_went_in(tmp_db):
    store.add_transactions(rows().iloc[::-1], path=tmp_db)
    df = store.read_transactions(tmp_db)
    assert df["Trade Date"].is_monotonic_increasing


def test_reimporting_the_same_file_adds_nothing(tmp_db):
    store.add_transactions(rows(), source="import:a.xlsx", path=tmp_db)
    out = store.add_transactions(rows(), source="import:a.xlsx", path=tmp_db)
    assert out == {"added": 0, "skipped": 2, "added_ids": [],
                   "skipped_keys": out["skipped_keys"]}
    assert len(store.read_transactions(tmp_db)) == 2


def test_an_overlapping_import_adds_only_the_new_rows(tmp_db):
    store.add_transactions(rows(), path=tmp_db)
    overlap = pd.concat([rows(), tx_frame([
        {"date": "2026-10-05", "activity": "Sell", "symbol": "AAA",
         "qty": 10.0, "amount": 540.0}])], ignore_index=True)
    out = store.add_transactions(overlap, source="import:b.xlsx", path=tmp_db)
    assert out["added"] == 1 and out["skipped"] == 2
    assert len(store.read_transactions(tmp_db)) == 3


def test_two_identical_fills_on_one_day_both_survive(tmp_db):
    """The real export contains three such pairs. De-duplicating on membership
    rather than multiplicity silently halves the position."""
    pair = tx_frame([
        {"date": "2023-02-10", "activity": "Buy", "symbol": "ZBRA",
         "qty": 2.0, "amount": 625.22},
        {"date": "2023-02-10", "activity": "Buy", "symbol": "ZBRA",
         "qty": 2.0, "amount": 625.22},
    ])
    out = store.add_transactions(pair, path=tmp_db)
    assert out["added"] == 2
    assert store.read_transactions(tmp_db)["Quantity"].sum() == pytest.approx(4.0)
    # and re-importing that same file still adds nothing
    again = store.add_transactions(pair, path=tmp_db)
    assert again["added"] == 0


def test_a_genuinely_new_third_fill_lands_even_when_two_already_exist(tmp_db):
    pair = tx_frame([
        {"date": "2023-02-10", "activity": "Buy", "symbol": "ZBRA",
         "qty": 2.0, "amount": 625.22},
        {"date": "2023-02-10", "activity": "Buy", "symbol": "ZBRA",
         "qty": 2.0, "amount": 625.22},
    ])
    store.add_transactions(pair, path=tmp_db)
    triple = pd.concat([pair, pair.iloc[[0]]], ignore_index=True)
    assert store.add_transactions(triple, path=tmp_db)["added"] == 1


def test_a_cent_of_difference_makes_it_a_different_trade(tmp_db):
    store.add_transactions(rows(), path=tmp_db)
    nudged = rows()
    nudged.loc[0, "Amount"] = 500.01
    assert store.add_transactions(nudged, path=tmp_db)["added"] == 1


def test_preview_reports_what_an_import_would_do_without_writing(tmp_db):
    store.add_transactions(rows(), path=tmp_db)
    overlap = pd.concat([rows(), tx_frame([
        {"date": "2026-10-05", "activity": "Sell", "symbol": "AAA",
         "qty": 10.0, "amount": 540.0}])], ignore_index=True)
    prev = store.preview_import(overlap, path=tmp_db)
    assert list(prev["status"]) == ["already in ledger", "already in ledger", "new"]
    assert len(store.read_transactions(tmp_db)) == 2      # nothing was written


def test_edit_changes_the_row_and_records_before_and_after(tmp_db):
    ids = store.add_transactions(rows(), path=tmp_db)["added_ids"]
    store.update_transaction(ids[0], {"Quantity": 11.0}, actor="sam", path=tmp_db)
    df = store.read_transactions(tmp_db)
    assert df.loc[df["id"] == ids[0], "Quantity"].iloc[0] == 11.0
    log = store.audit_log(tmp_db)
    edit = log[log["action"] == "edit"].iloc[0]
    assert edit["actor"] == "sam" and "11.0" in edit["detail"]


def test_edit_refuses_a_field_that_is_not_part_of_a_transaction(tmp_db):
    ids = store.add_transactions(rows(), path=tmp_db)["added_ids"]
    with pytest.raises(KeyError):
        store.update_transaction(ids[0], {"Sleeve": "equity"}, path=tmp_db)


def test_delete_is_soft_and_restorable(tmp_db):
    ids = store.add_transactions(rows(), path=tmp_db)["added_ids"]
    store.delete_transaction(ids[0], actor="sam", reason="duplicate", path=tmp_db)
    assert len(store.read_transactions(tmp_db)) == 1
    assert len(store.read_transactions(tmp_db, include_deleted=True)) == 2
    store.restore_transaction(ids[0], path=tmp_db)
    assert len(store.read_transactions(tmp_db)) == 2


def test_a_deleted_row_frees_its_key_for_reimport(tmp_db):
    ids = store.add_transactions(rows(), path=tmp_db)["added_ids"]
    store.delete_transaction(ids[0], path=tmp_db)
    assert store.add_transactions(rows(), path=tmp_db)["added"] == 1


def test_every_write_leaves_an_audit_row(tmp_db):
    ids = store.add_transactions(rows(), actor="pat", path=tmp_db)["added_ids"]
    store.update_transaction(ids[0], {"Notes": "checked"}, actor="pat", path=tmp_db)
    store.delete_transaction(ids[1], actor="pat", path=tmp_db)
    assert set(store.audit_log(tmp_db)["action"]) == {"add", "edit", "delete"}


def test_classification_is_stored_and_validated(tmp_db):
    store.classify_symbol("newbond", "fixed_income", actor="pat", path=tmp_db)
    assert store.symbol_classes(tmp_db) == {"NEWBOND": "fixed_income"}
    with pytest.raises(ValueError):
        store.classify_symbol("XYZ", "crypto", path=tmp_db)


def test_reading_an_export_rejects_an_unknown_activity(tmp_path):
    p = tmp_path / "bad.csv"
    df = rows()
    df.loc[0, "Activity"] = "Teleport"
    df[config.TX_COLUMNS].to_csv(p, index=False)
    with pytest.raises(ValueError, match="unrecognised Activity"):
        store.read_export(p)


def test_reading_an_export_rejects_a_missing_column(tmp_path):
    p = tmp_path / "short.csv"
    rows().drop(columns=["Amount"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing column"):
        store.read_export(p)


def test_a_csv_export_reads_the_same_as_an_xlsx(tmp_path):
    p = tmp_path / "ok.csv"
    rows()[config.TX_COLUMNS].to_csv(p, index=False)
    df = store.read_export(p)
    assert list(df.columns) == config.TX_COLUMNS
    assert len(df) == 2


def test_seeding_is_a_no_op_once_the_ledger_has_rows(tmp_path, tmp_db):
    p = tmp_path / "seed.csv"
    rows()[config.TX_COLUMNS].to_csv(p, index=False)
    assert store.seed_from_export(p, db_path=tmp_db)["added"] == 2
    out = store.seed_from_export(p, db_path=tmp_db)
    assert out["added"] == 0 and "already populated" in out["note"]
