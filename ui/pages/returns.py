"""
Returns — every period, every sleeve, side by side.

The matrix at the top is the page's reason to exist: four sleeves against six
periods in one glance. Two things about it are deliberate.

FIRST, "ALL TIME" IS NOT ONE WINDOW. Each sleeve is measured from its own
inception — equity from January 2010, fixed income from September 2026 — so
the column compares lifetimes, not a common period. Every other column is a
genuine like-for-like window. The table says so rather than letting a reader
assume a 0.7% fixed income figure is a seventeen-year result.

SECOND, EACH SLEEVE IS COMPARED TO ITS OWN BENCHMARK. Equity and the total
portfolio against SPY, fixed income against AGG, cash against nothing — there
is no index for a sweep account, and inventing one would imply the fund was
trying to beat something. The excess table simply leaves cash blank.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from smif import config, returns as returns_mod
from ui import charts, data, theme

theme.apply_theme()
data.require_data()

a = data.analytics()
book = a.book

SLEEVES = [("total", "Total portfolio"), ("equity", "Equity"),
           ("fixed_income", "Fixed income"), ("cash", "Cash")]
PERIOD_ORDER = ["All time", "10Y", "5Y", "3Y", "1Y", "YTD"]

theme.topbar(
    "Returns",
    f"Time-weighted, every sleeve · as of {a.asof:%d %b %Y}",
    chips=[("Net assets", theme.money(float(book.total.iloc[-1])), None),
           ("Since", f"{a.tracks['total'].base:%b %Y}", None)])


# ── shared formatting ─────────────────────────────────────────────────────────

def _colour(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"color: {theme.TEXT_DIM}"
    return f"color: {theme.GREEN if v > 0 else theme.RED if v < 0 else theme.TEXT_MUTED}"


NA = "—"


def _pct_text(v, places: int = 2, signed: bool = True) -> str:
    if v is None or pd.isna(v):
        return NA
    return f"{v * 100:{'+' if signed else ''},.{places}f}%"


def _styled(numeric: pd.DataFrame, text: pd.DataFrame,
            colour_cols: list[str] | None = None) -> pd.io.formats.style.Styler:
    """
    Render already-formatted TEXT, and carry the colour separately.

    Streamlit does not use a Styler's display values for missing entries — a
    cell that pandas renders as an em dash arrives on screen as the word
    "None", whichever of format(na_rep=...) or a formatter that handles NaN
    produced it. Handing Streamlit a frame of strings removes the question:
    there is no number left for it to reinterpret. The colours then come from
    the numeric frame through a parallel CSS table of the same shape, so the
    sign still drives the hue without the text depending on it.
    """
    cols = colour_cols if colour_cols is not None else list(numeric.columns)
    css = pd.DataFrame("", index=text.index, columns=text.columns)
    for c in cols:
        if c in numeric.columns:
            css[c] = numeric[c].map(_colour)
    return text.style.apply(lambda _: css, axis=None)


def _show(df: pd.DataFrame, *, places: int = 2, colour: bool = True,
          height: int | None = None) -> None:
    """A percentage matrix, green/red by sign, em dash where there is no number."""
    numeric = df.apply(pd.to_numeric, errors="coerce")
    text = numeric.map(lambda v: _pct_text(v, places))
    styler = _styled(numeric, text, list(numeric.columns) if colour else [])
    st.dataframe(styler, use_container_width=True,
                 height=height or (38 * len(df) + 40))


# ── the matrix ────────────────────────────────────────────────────────────────

tables = {key: data.period_table(key).set_index("period")
          for key, _ in SLEEVES}

with theme.panel("Returns by sleeve and period",
                 note="time-weighted · external flows neutralised"):
    mode = st.segmented_control(
        "Basis", ["Cumulative", "Annualized", "Excess vs benchmark"],
        default="Cumulative", key="basis", label_visibility="collapsed") \
        or "Cumulative"
    field = {"Cumulative": "cum", "Annualized": "ann",
             "Excess vs benchmark": "excess"}[mode]

    def cell(t: pd.DataFrame, period: str, column: str):
        """
        The figure for one period, or None when the sleeve is younger than it.

        A sleeve that did not exist ten years ago has its 10Y window clamped to
        its own inception, which is arithmetically fine and editorially awful:
        fixed income's six-week +0.75% would appear under a "10Y" heading and
        read as a decade's work. Blank is the honest cell.
        """
        if period not in t.index:
            return None
        if period != "All time" and t.loc[period, "start"] == t.loc["All time", "start"]:
            return None
        return t.loc[period, column]

    rows = {}
    for key, label in SLEEVES:
        rows[label] = {p: cell(tables[key], p, field) for p in PERIOD_ORDER}
    matrix = pd.DataFrame(rows).T[PERIOD_ORDER]

    if field != "excess":
        # Benchmark rows ride on the track whose history they actually span:
        # SPY over the fund's whole life, AGG only since fixed income existed.
        bench_rows = {}
        col = "bench_ann" if field == "ann" else "bench_cum"
        for src, name in (("total", "SPY"), ("fixed_income", "AGG")):
            bench_rows[name] = {p: cell(tables[src], p, col)
                                for p in PERIOD_ORDER}
        matrix = pd.concat([matrix, pd.DataFrame(bench_rows).T[PERIOD_ORDER]])

    _show(matrix)

    starts = " · ".join(
        f"{label} from {tables[key].loc['All time', 'start']:%b %Y}"
        for key, label in SLEEVES if "All time" in tables[key].index)
    theme.hover_strip(
        "<strong>All time is each sleeve's own lifetime</strong>, not a shared "
        f"window — {starts}. Every other column is the same window for every "
        "row, and is left blank where a sleeve is younger than the period. "
        + ("Annualized figures are shown only where the window is a year or "
           "longer; shorter ones would exaggerate."
           if field == "ann" else
           "Cash has no benchmark: there is no index for a sweep account."
           if field == "excess" else
           "SPY and AGG are measured over the fund's full history.")
    )

# ── growth, all sleeves ───────────────────────────────────────────────────────

with theme.panel("Every sleeve from its own start",
                 note="cumulative %, each rebased to its first day"):
    st.plotly_chart(charts.multi_sleeve_chart(a.tracks, height=360),
                    use_container_width=True, config=theme.PLOTLY_CONFIG,
                    key="sleeves_growth")
    theme.hover_strip(
        "Each line starts at 0% on its own first day, so the lines are not "
        "comparable in level — fixed income's six weeks sit beside equity's "
        "seventeen years. Read the shapes, not the endpoints; the matrix above "
        "is where the comparable numbers are.")

# ── one sleeve in detail ──────────────────────────────────────────────────────

with theme.panel("One sleeve in detail"):
    pick = st.segmented_control(
        "Sleeve", [label for _, label in SLEEVES], default="Total portfolio",
        key="detail_sleeve", label_visibility="collapsed") or "Total portfolio"
    key = dict((label, k) for k, label in SLEEVES)[pick]
    track = a.tracks[key]
    detail = tables[key].reset_index()

    out = pd.DataFrame({
        "Period": detail["period"],
        "From": detail["start"].dt.strftime("%d %b %Y"),
        "To": detail["end"].dt.strftime("%d %b %Y"),
        # as text, not a rounded float: Streamlit picks its own precision for a
        # numeric column and renders 17.06 as 17.060000
        "Years": detail["years"].map(lambda v: f"{v:.2f}"),
        "Cumulative": detail["cum"],
        "Annualized": detail["ann"],
        track.benchmark_name or "Benchmark": detail["bench_cum"],
        "Excess": detail["excess"],
        "Max drawdown": detail["mdd"],
        "Volatility": detail["vol"],
    })
    bench_col = track.benchmark_name or "Benchmark"
    signed_cols = ["Cumulative", "Annualized", bench_col, "Excess",
                   "Max drawdown"]
    numeric = out[signed_cols + ["Volatility"]].apply(pd.to_numeric,
                                                      errors="coerce")
    text = out.copy()
    for c in signed_cols:
        text[c] = numeric[c].map(_pct_text)
    text["Volatility"] = numeric["Volatility"].map(
        lambda v: _pct_text(v, signed=False))
    st.dataframe(_styled(numeric, text,
                         ["Cumulative", "Annualized", bench_col, "Excess"]),
                 hide_index=True, use_container_width=True)

# ── custom range ──────────────────────────────────────────────────────────────

with theme.panel("Custom date range",
                 note="both ends snap to the last close on or before the date"):
    first, last = book.days[0].date(), a.asof.date()
    c1, c2 = st.columns([1, 3])
    with c1:
        picked = st.date_input(
            "Range", value=(max(first, (a.asof - pd.DateOffset(years=2)).date()),
                            last),
            min_value=first, max_value=last, key="custom_range",
            label_visibility="collapsed")

    if isinstance(picked, tuple) and len(picked) == 2:
        start, end = (pd.Timestamp(picked[0]), pd.Timestamp(picked[1]))
        rows = []
        for k, label in SLEEVES:
            s = returns_mod.range_stats(a.tracks[k], start, end)
            if s is None:
                rows.append({"Sleeve": label, "From": "—", "To": "—",
                             "Cumulative": None, "Annualized": None,
                             "Benchmark": None, "Excess": None,
                             "Max drawdown": None, "Volatility": None,
                             "note": "not held over this window"})
                continue
            rows.append({
                "Sleeve": label,
                "From": f"{s['start']:%d %b %Y}", "To": f"{s['end']:%d %b %Y}",
                "Cumulative": s["cum"], "Annualized": s["ann"],
                "Benchmark": s["bench_cum"], "Excess": s["excess"],
                "Max drawdown": s["mdd"], "Volatility": s["vol"],
                "note": s["bench"] or "no benchmark"})
        cr = pd.DataFrame(rows)
        pct_cols = ["Cumulative", "Annualized", "Benchmark", "Excess",
                    "Max drawdown", "Volatility"]
        numeric = cr[pct_cols].apply(pd.to_numeric, errors="coerce")
        text = cr.copy()
        for c in pct_cols:
            text[c] = numeric[c].map(_pct_text)
        text = text.rename(columns={"note": "Against"})
        st.dataframe(_styled(numeric, text,
                             ["Cumulative", "Annualized", "Benchmark", "Excess"]),
                     hide_index=True, use_container_width=True)
    else:
        st.caption("Pick an end date to see the window.")

# ── calendar years ────────────────────────────────────────────────────────────

with theme.panel(f"Calendar years — {pick.lower()}",
                 note="the first and last years are partial"):
    cal = data.calendar_table(key)
    if cal.empty:
        st.caption("Not enough history for a calendar year yet.")
    else:
        st.plotly_chart(
            charts.calendar_bars(cal, benchmark_name=track.benchmark_name or "—",
                                 height=300),
            use_container_width=True, config=theme.PLOTLY_CONFIG,
            key=f"cal_{key}")

        table = pd.DataFrame({
            "Year": cal["year"].astype(str) + np.where(cal["partial"], " *", ""),
            pick: cal["portfolio"],
            track.benchmark_name or "Benchmark": cal["benchmark"],
            "Excess": cal["portfolio"] - cal["benchmark"],
        })
        num_cols = [c for c in table.columns if c != "Year"]
        numeric = table[num_cols].apply(pd.to_numeric, errors="coerce")
        text = table.copy()
        for c in num_cols:
            text[c] = numeric[c].map(_pct_text)
        st.dataframe(_styled(numeric, text, num_cols), hide_index=True,
                     use_container_width=True, height=38 * len(table) + 40)
        st.caption("* partial year — the fund did not start on 1 January, and "
                   "the current year is not finished.")
