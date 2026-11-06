"""Placeholder — this page is next. The engine behind it is already built and
tested; only the view is missing."""

import streamlit as st

from ui import data, theme

theme.apply_theme()
data.require_data()

theme.topbar("factors".replace("_", " ").title(), "not built yet")
st.info(
    "This page is not built yet. The Overview is live, and the engine behind "
    "every page here is already in place and covered by tests — run "
    "`python scripts/check.py` for the numbers in the meantime."
)
