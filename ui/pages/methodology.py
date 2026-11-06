"""
Methodology — not built yet.

Deliberately honest about what is missing rather than empty: the rules this
page is meant to document are the ones the rest of the app's numbers depend
on, so a reader who comes here and finds nothing should at least be told where
those numbers do come from in the meantime.
"""

import streamlit as st

from smif import config
from ui import data, theme

theme.apply_theme()
data.require_data()

a = data.analytics()

theme.topbar("Methodology", "not built yet — the engine behind it is")

theme.note(
    "<strong>This page is a placeholder.</strong> It is meant to document the "
    "transaction rules, the names priced by interpolation, the cash "
    "reconciliation gap and the validation results in one place. All four "
    "exist and are tested; only this view is missing.")

st.markdown(
    f"""
Until it is written, the same material is available in three places:

- **`python scripts/check.py`** prints the whole reconciliation and regression
  report — share counts against the workbook, sleeve values against the
  custodian snapshot of {config.CUSTODIAN_AS_OF:%d %b %Y}, every Buy and Sell
  fill against the rebuilt close, and the return table.
- **The Overview** carries the two findings that matter day to day: the
  ${abs(2041.81):,.2f} cash gap and the positions priced by interpolation.
- **The README** sets out the time-weighted return method, the split
  un-adjustment and the reconciliation results.

The engine modules are also written to be read: `smif/ledger.py` documents
every transaction rule, `smif/prices.py` the split un-adjustment and the
interpolation fallback, and `smif/returns.py` the flow treatment.
"""
)

theme.tiles([
    {"label": "Ledger rows", "value": f"{len(a.tx):,}"},
    {"label": "Trading days rebuilt", "value": f"{len(a.book.days):,}"},
    {"label": "Priced symbols", "value": f"{a.book.prices.shape[1]:,}"},
    {"label": "As of", "value": f"{a.asof:%d %b %Y}"},
])
