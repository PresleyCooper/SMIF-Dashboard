"""
app.py — the shell.

Navigation is declared here rather than left to Streamlit's filename-ordering
convention, so the page order is a decision in one readable place and a page can
be renamed without renumbering its neighbours.

Attribution (allocation vs selection) is deliberately not built. Its slot is
left below, commented, so adding it later is one line here plus one file — and
so nobody wonders whether it was forgotten.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui import theme  # noqa: E402

st.set_page_config(
    page_title="SMIF Portfolio Analytics",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.apply_theme()
theme.sidebar_brand()

PAGES = [
    st.Page("ui/pages/overview.py", title="Overview", icon=":material/dashboard:",
            default=True),
    st.Page("ui/pages/returns.py", title="Returns", icon=":material/trending_up:"),
    st.Page("ui/pages/correlations.py", title="Correlations",
            icon=":material/grid_on:"),
    st.Page("ui/pages/factors.py", title="Factor exposure",
            icon=":material/equalizer:"),
    # st.Page("ui/pages/attribution.py", title="Attribution",
    #         icon=":material/pie_chart:"),   # allocation vs selection — not built
    st.Page("ui/pages/transactions.py", title="Transactions",
            icon=":material/receipt_long:"),
    st.Page("ui/pages/methodology.py", title="Methodology",
            icon=":material/help_outline:"),
]

st.navigation(PAGES).run()
