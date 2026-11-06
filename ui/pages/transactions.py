"""
Transactions — the ledger, and the four ways to change it.

The canonical ledger is the SQLite database. The Schwab export is seed data,
read once and never written back to, so the file in Downloads stays exactly as
the custodian produced it.

Two choices shape this page.

VALIDATION IS LIVE, NOT ON SUBMIT. The form uses ordinary widgets rather than
st.form precisely so every keystroke re-runs the checks: "the fund holds 116
AAPL" is useful while you are typing 200, and useless after you have saved it.

NOTHING IS WRITTEN WITHOUT A PREVIEW. An import shows which rows are new
before it adds them, and a single entry shows what it will do to shares, cash
and the sleeve before it does it. Both are reversible through a soft delete,
but being able to undo a mistake is a poor substitute for seeing it coming.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from smif import config, entry, store, validate
from ui import data, theme

theme.apply_theme()
data.require_data()

a = data.analytics()
book = a.book
held = entry.current_holdings(book)
known = store.known_symbols()
overrides = store.symbol_classes()

theme.topbar(
    "Transactions",
    f"{len(a.tx):,} rows · {a.tx['Trade Date'].min():%b %Y} to "
    f"{a.tx['Trade Date'].max():%d %b %Y} · the ledger is "
    f"data/ledger.sqlite, the export is never modified")

# Who is making the change. Recorded on every row and every audit entry; the
# audit trail is worth very little if every change is by "unknown".
if "actor" not in st.session_state:
    st.session_state["actor"] = ""
actor = st.text_input(
    "Your name", key="actor", placeholder="who is making this change",
    help="Recorded against every row you add, edit or delete.")
actor = (actor or "").strip() or "unknown"

LEDGER, ADD, IMPORT, RECONCILE, AUDIT = st.tabs(
    ["Ledger", "Add a transaction", "Import an export", "Reconcile", "Audit"])


def _issue_block(issues: list[entry.Issue]) -> None:
    for i in entry.blocking(issues):
        st.error(f"**{i.field or 'Row'}** — {i.message}")
    for i in entry.warnings(issues):
        st.warning(f"**{i.field or 'Row'}** — {i.message}")


# ── the ledger ────────────────────────────────────────────────────────────────
with LEDGER:
    tx = a.tx.copy()
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    with f1:
        syms = st.multiselect("Symbol", sorted(tx["Symbol"].dropna().unique()),
                              key="flt_sym", placeholder="any symbol")
    with f2:
        acts = st.multiselect("Activity", sorted(tx["Activity"].unique()),
                              key="flt_act", placeholder="any activity")
    with f3:
        years = sorted(tx["Trade Date"].dt.year.unique(), reverse=True)
        yrs = st.multiselect("Year", years, key="flt_year",
                             placeholder="any year")
    with f4:
        manual_only = st.checkbox("Manual only", key="flt_manual",
                                  help="Rows added here rather than seeded "
                                       "from the original export")

    view = tx
    if syms:
        view = view[view["Symbol"].isin(syms)]
    if acts:
        view = view[view["Activity"].isin(acts)]
    if yrs:
        view = view[view["Trade Date"].dt.year.isin(yrs)]
    if manual_only:
        view = view[~view["source"].astype(str).str.startswith("seed")]

    st.caption(f"{len(view):,} of {len(tx):,} rows")
    shown = view.sort_values(["Trade Date", "id"], ascending=[False, False])
    display = shown[["id", "Trade Date", "Activity", "Description", "Symbol",
                     "Quantity", "Amount", "Notes", "source", "added_by"]].copy()
    display["Trade Date"] = display["Trade Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display, hide_index=True, use_container_width=True, height=430,
                 column_config={"id": st.column_config.NumberColumn("ID",
                                                                    width="small")})

    with theme.panel("Edit or delete a row"):
        if shown.empty:
            st.caption("No rows match the filters.")
        else:
            pick = st.selectbox(
                "Row", shown["id"].tolist(), key="edit_id",
                format_func=lambda i: (
                    f"#{i} · {shown.set_index('id').loc[i, 'Trade Date']:%Y-%m-%d}"
                    f" · {shown.set_index('id').loc[i, 'Activity']}"
                    f" · {shown.set_index('id').loc[i, 'Symbol'] or '—'}"))
            current = shown.set_index("id").loc[pick]

            e1, e2, e3 = st.columns(3)
            with e1:
                new_qty = st.number_input(
                    "Quantity", value=float(current["Quantity"])
                    if pd.notna(current["Quantity"]) else 0.0,
                    step=1.0, format="%.6f", key="edit_qty")
            with e2:
                new_amt = st.number_input(
                    "Amount", value=float(current["Amount"])
                    if pd.notna(current["Amount"]) else 0.0,
                    step=1.0, format="%.2f", key="edit_amt")
            with e3:
                new_notes = st.text_input("Notes",
                                          value=str(current["Notes"] or ""),
                                          key="edit_notes")

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Save the edit", key="do_edit"):
                    store.update_transaction(
                        int(pick),
                        {"Quantity": new_qty or None, "Amount": new_amt or None,
                         "Notes": new_notes}, actor=actor)
                    data.refresh()
                    st.success(f"Row #{pick} edited. Everything recomputed.")
                    st.rerun()
            with b2:
                reason = st.text_input("Reason for deleting", key="del_reason",
                                       placeholder="e.g. keyed twice")
                if st.button("Delete this row", key="do_delete"):
                    store.delete_transaction(int(pick), actor=actor,
                                             reason=reason)
                    data.refresh()
                    st.success(f"Row #{pick} deleted. It stays in the audit "
                               f"trail and can be restored.")
                    st.rerun()

# ── add one ───────────────────────────────────────────────────────────────────
with ADD:
    c1, c2 = st.columns([3, 2])

    with c1:
        d1, d2 = st.columns(2)
        with d1:
            in_date = st.date_input("Trade Date", value=a.asof.date(),
                                    max_value=config.today().date(),
                                    key="add_date")
        with d2:
            in_act = st.selectbox("Activity", config.ACTIVITIES, key="add_act")

        d3, d4 = st.columns(2)
        with d3:
            in_sym = st.text_input("Symbol", key="add_sym",
                                   placeholder="AAPL").strip().upper()
        with d4:
            in_desc = st.text_input("Description", key="add_desc",
                                    placeholder="Apple Inc")

        d5, d6 = st.columns(2)
        with d5:
            in_qty = st.text_input("Quantity", key="add_qty", placeholder="10")
        with d6:
            in_amt = st.text_input("Amount", key="add_amt", placeholder="3000.00")

        in_notes = st.text_input("Notes", key="add_notes", placeholder="optional")

        unknown = bool(in_sym) and in_sym not in known and in_sym not in config.CASH_LIKE
        confirm_new = False
        if unknown:
            confirm_new = st.checkbox(
                f"{in_sym} is a new position, not a typo", key="add_confirm")

        candidate = {"Trade Date": pd.Timestamp(in_date), "Activity": in_act,
                     "Description": in_desc, "Symbol": in_sym or None,
                     "Quantity": in_qty, "Amount": in_amt, "Notes": in_notes}
        issues = entry.validate_row(candidate, holdings=held,
                                    known_symbols=known,
                                    existing_keys=store.existing_keys(),
                                    new_symbol_confirmed=confirm_new)

        sleeve_choice = None
        if (unknown and confirm_new
                and entry.needs_classification(in_sym, overrides)):
            sleeve_choice = st.selectbox(
                f"Which sleeve does {in_sym} belong to?",
                [config.SLEEVE_LABEL[s] for s in config.SLEEVES],
                key="add_sleeve",
                help="Anything unclassified defaults to equity, which is the "
                     "right fallback and the wrong thing to do quietly to a "
                     "bond fund.")

    with c2:
        st.markdown(theme.eyebrow("What this will do"), unsafe_allow_html=True)
        effect = entry.preview_effect(candidate)
        if "error" in effect:
            st.caption("Fill in the row to see its effect.")
        else:
            rows = [
                ("Symbol", effect["symbol"] or "—"),
                ("Sleeve", config.SLEEVE_LABEL.get(effect["sleeve"], "—")),
                ("Shares", f"{effect['shares']:+,.4f}"),
                ("Cash", theme.money(effect["cash"], 2)
                 if effect["cash"] else "$0.00"),
                ("External flow", theme.money(effect["external_flow"], 2)
                 if effect["external_flow"] else "—"),
                ("Sleeve income", theme.money(effect["income"], 2)
                 if effect["income"] else "—"),
                ("Net purchase", theme.money(effect["net_purchase"], 2)
                 if effect["net_purchase"] else "—"),
            ]
            st.markdown(
                '<div class="cf-hover">'
                + "<br>".join(
                    f'<span style="color:{theme.TEXT_DIM}">{k}</span> '
                    f'<span class="cf-mono" style="color:{theme.TEXT}">{v}</span>'
                    for k, v in rows)
                + "</div>", unsafe_allow_html=True)
        if in_sym and in_sym in held.index:
            st.caption(f"The fund currently holds {held[in_sym]:,.4f} {in_sym}.")

    _issue_block(issues)

    can_save = not entry.blocking(issues)
    if st.button("Save this transaction", type="primary", disabled=not can_save,
                 key="add_save"):
        if sleeve_choice:
            inverse = {v: k for k, v in config.SLEEVE_LABEL.items()}
            store.classify_symbol(in_sym, inverse[sleeve_choice], actor=actor)
        frame = pd.DataFrame([{
            "Trade Date": pd.Timestamp(in_date), "Activity": in_act,
            "Description": in_desc or in_sym, "Symbol": in_sym or None,
            "Quantity": float(in_qty) if in_qty.strip() else None,
            "Amount": float(in_amt) if in_amt.strip() else None,
            "Notes": in_notes}])
        out = store.add_transactions(frame, source="manual", actor=actor)
        data.refresh()
        st.success(f"Saved as row #{out['added_ids'][0]}. Holdings, prices and "
                   f"returns recomputed.")
        st.rerun()

# ── bulk import ───────────────────────────────────────────────────────────────
with IMPORT:
    st.markdown(
        f'<div style="font-size:12.5px;color:{theme.TEXT_MUTED};line-height:1.6">'
        f'Upload a fresh Schwab export. Rows already in the ledger are matched '
        f'on date, activity, symbol, quantity and amount — and matched by '
        f'count, so a genuine second identical fill on the same day still '
        f'lands. Nothing is written until you have seen the preview.</div>',
        unsafe_allow_html=True)

    upload = st.file_uploader("Schwab export", type=["xlsx", "xls", "csv"],
                              key="imp_file")
    if upload is not None:
        try:
            incoming = store.read_export(upload)
        except ValueError as exc:
            st.error(f"That file cannot be read: {exc}")
        else:
            preview = store.preview_import(incoming)
            new = preview[preview["status"] == "new"]
            theme.tiles([
                {"label": "Rows in file", "value": f"{len(preview):,}"},
                {"label": "New", "value": f"{len(new):,}"},
                {"label": "Already held", "value": f"{len(preview) - len(new):,}"},
                {"label": "Date range",
                 "value": f"{incoming['Trade Date'].min():%b %Y}",
                 "sub": f"to {incoming['Trade Date'].max():%b %Y}"},
            ])
            if new.empty:
                st.info("Every row in this file is already in the ledger. "
                        "Nothing to import.")
            else:
                show = new[config.TX_COLUMNS].copy()
                show["Trade Date"] = show["Trade Date"].dt.strftime("%Y-%m-%d")
                st.dataframe(show, hide_index=True, use_container_width=True,
                             height=min(420, 36 * len(show) + 42))

                unseen = sorted({s for s in new["Symbol"].dropna().unique()
                                 if s.upper() not in known
                                 and s.upper() not in config.CASH_LIKE})
                picks = {}
                if unseen:
                    st.warning(
                        f"{len(unseen)} symbol(s) new to the ledger: "
                        f"{', '.join(unseen)}. Each needs a sleeve.")
                    for s in unseen:
                        if entry.needs_classification(s, overrides):
                            picks[s] = st.selectbox(
                                f"Sleeve for {s}",
                                [config.SLEEVE_LABEL[k] for k in config.SLEEVES],
                                key=f"imp_sleeve_{s}")

                if st.button(f"Import {len(new):,} new rows", type="primary",
                             key="imp_save"):
                    inverse = {v: k for k, v in config.SLEEVE_LABEL.items()}
                    for s, label in picks.items():
                        store.classify_symbol(s, inverse[label], actor=actor)
                    out = store.add_transactions(
                        incoming, source=f"import:{upload.name}", actor=actor)
                    data.refresh()
                    st.success(f"Added {out['added']:,} rows, skipped "
                               f"{out['skipped']:,} already held. Everything "
                               f"recomputed.")
                    st.rerun()

# ── reconcile ─────────────────────────────────────────────────────────────────
with RECONCILE:
    st.markdown(
        f'<div style="font-size:12.5px;color:{theme.TEXT_MUTED};line-height:1.6">'
        f'Upload a custodian position file to compare against the rebuild, '
        f'position by position. Column names are matched loosely — symbol / '
        f'ticker, quantity / shares, price, value / market value — so a file '
        f'does not have to be reshaped first.</div>',
        unsafe_allow_html=True)

    rec_file = st.file_uploader("Custodian positions",
                                type=["xlsx", "xls", "csv"], key="rec_file")
    rec_date = st.date_input("As of", value=a.asof.date(),
                             min_value=book.days[0].date(),
                             max_value=a.asof.date(), key="rec_date")

    if rec_file is not None:
        try:
            snap = validate.read_position_snapshot(rec_file)
        except (ValueError, KeyError) as exc:
            st.error(f"That file cannot be read: {exc}")
        else:
            cmp = validate.reconcile_positions(book, snap,
                                               as_of=pd.Timestamp(rec_date))
            matched = int(cmp["matches"].sum())
            theme.tiles([
                {"label": "Positions compared", "value": f"{len(cmp)}"},
                {"label": "Share counts match", "value": f"{matched}",
                 "sign": 1.0 if matched == len(cmp) else -1.0},
                {"label": "Rebuilt value",
                 "value": theme.money(cmp["rebuilt_value"].sum())},
                {"label": "Snapshot value",
                 "value": theme.money(np.nansum(cmp["custodian_value"]))},
            ])

            out = pd.DataFrame({
                "Symbol": cmp["symbol"],
                "Rebuilt shares": cmp["rebuilt_shares"].map("{:,.4f}".format),
                "Snapshot shares": cmp["custodian_shares"].map("{:,.4f}".format),
                "Share diff": cmp["share_diff"],
                "Rebuilt value": cmp["rebuilt_value"],
                "Snapshot value": cmp["custodian_value"],
                "Value diff": cmp["value_diff"],
                "Only in": cmp["only_in"].replace("", "—"),
            })
            numeric = out[["Share diff", "Rebuilt value", "Snapshot value",
                           "Value diff"]]
            text = out.copy()
            text["Share diff"] = numeric["Share diff"].map(
                lambda v: "—" if pd.isna(v) else f"{v:+,.4f}")
            for c in ("Rebuilt value", "Snapshot value"):
                text[c] = numeric[c].map(
                    lambda v: "—" if pd.isna(v) else f"${v:,.2f}")
            text["Value diff"] = numeric["Value diff"].map(
                lambda v: "—" if pd.isna(v) else f"{v:+,.2f}")
            css = pd.DataFrame("", index=text.index, columns=text.columns)
            for c in ("Share diff", "Value diff"):
                css[c] = numeric[c].map(
                    lambda v: f"color: {theme.TEXT_DIM}" if pd.isna(v)
                    else (f"color: {theme.TEXT_MUTED}" if abs(v) < 1e-4
                          else f"color: {theme.RED}"))
            st.dataframe(text.style.apply(lambda _: css, axis=None),
                         hide_index=True, use_container_width=True,
                         height=min(520, 36 * len(text) + 42))
            st.caption("Rows where only one side holds the position are kept "
                       "and labelled — a position the rebuild has and the "
                       "custodian does not is the most important line here.")

# ── audit ─────────────────────────────────────────────────────────────────────
with AUDIT:
    log = store.audit_log(limit=500)
    if log.empty:
        st.caption("Nothing has been changed yet.")
    else:
        st.caption(f"The {len(log):,} most recent changes, newest first.")
        st.dataframe(log.rename(columns={
            "ts": "When", "action": "Action", "tx_id": "Row",
            "actor": "By", "detail": "Detail"}), hide_index=True,
            use_container_width=True, height=440)

    deleted = store.read_transactions(include_deleted=True)
    deleted = deleted[deleted["deleted"] == 1]
    if not deleted.empty:
        with theme.panel("Deleted rows",
                         note="soft-deleted: excluded from every number, kept "
                              "as evidence"):
            show = deleted[["id", "Trade Date", "Activity", "Symbol",
                            "Quantity", "Amount", "added_by"]].copy()
            show["Trade Date"] = show["Trade Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(show, hide_index=True, use_container_width=True)
            restore = st.selectbox("Restore row", deleted["id"].tolist(),
                                   key="restore_id")
            if st.button("Restore", key="do_restore"):
                store.restore_transaction(int(restore), actor=actor)
                data.refresh()
                st.success(f"Row #{restore} restored.")
                st.rerun()
