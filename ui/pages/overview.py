"""
Overview — the headline numbers, the growth path, and how much is in cash.

One decision shapes this page: THE TOGGLE IS NOT A FILTER, IT CHANGES THE
QUESTION. "Total portfolio" is what the fund earned on everything it was given,
cash drag included — the number a donor or an advisory board is owed.
"Equity only" is what the stock picks did, measured against the benchmark those
picks were chosen from. Both are correct and they differ by roughly 120
percentage points over the fund's life, so the page says which one is on screen
rather than leaving a reader to guess.

The cash weight chart sits directly under the growth chart for the same reason:
it is the explanation for most of the gap between the two.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from smif import config, returns as returns_mod
from ui import charts, data, theme

theme.apply_theme()
data.require_data()

a = data.analytics()
book = a.book

VIEWS = {
    "Total portfolio": "total",
    "Equity only": "equity",
}
# Window label -> the period row in the returns table that describes it, so the
# summary strip under the chart always matches the window on screen.
WINDOW_PERIOD = {"All": "All time", "10Y": "10Y", "5Y": "5Y", "3Y": "3Y",
                 "1Y": "1Y", "YTD": "YTD"}
WINDOWS = {
    "All": None,
    "10Y": pd.DateOffset(years=10),
    "5Y": pd.DateOffset(years=5),
    "3Y": pd.DateOffset(years=3),
    "1Y": pd.DateOffset(years=1),
    "YTD": "ytd",
}

# ── topbar ────────────────────────────────────────────────────────────────────
nav = float(book.total.iloc[-1])
cash_w = float(book.cash_weight().iloc[-1])
one_day = float(returns_mod.total_daily_returns(book).iloc[-1])

theme.topbar(
    "Overview",
    f"KSU Student Managed Investment Fund · rebuilt from "
    f"{len(a.tx):,} transactions · as of {a.asof:%d %b %Y}",
    chips=[
        ("Net assets", theme.money(nav), None),
        ("1 day", theme.pct(one_day), one_day),
        ("Cash", f"{cash_w * 100:.1f}%", None),
    ])

# ── the view toggle ───────────────────────────────────────────────────────────
left, right = st.columns([2, 3], vertical_alignment="center")
with left:
    label = st.segmented_control(
        "View", list(VIEWS), default="Total portfolio", key="view",
        label_visibility="collapsed")
which = VIEWS.get(label or "Total portfolio", "total")
track = a.tracks[which]

with right:
    st.markdown(
        f'<div style="font-size:11.5px;color:{theme.TEXT_DIM};line-height:1.5">'
        + ("Everything the fund holds, cash included — the return on all capital "
           "entrusted to it. Deposits and withdrawals are neutralised; dividends "
           "are not, because the fund earned them."
           if which == "total" else
           "The stock sleeve on its own, measured against the benchmark its "
           "names were chosen from. Excludes the cash drag that the total "
           "portfolio carries.")
        + "</div>", unsafe_allow_html=True)

# ── headline tiles ────────────────────────────────────────────────────────────
periods = data.period_table(which)
bench = track.benchmark_name or "benchmark"

tiles = []
for _, r in periods.iterrows():
    excess = (None if r["bench_cum"] is None or pd.isna(r["bench_cum"])
              else r["cum"] - r["bench_cum"])
    sub = (f'{bench} {theme.pct(r["bench_cum"], 1)}' if excess is not None
           else f'from {r["start"]:%b %Y}')
    tiles.append({"label": r["period"], "value": theme.pct(r["cum"], 1),
                  "sign": r["cum"], "sub": sub})
theme.tiles(tiles)

ann = periods.set_index("period")
all_time = ann.loc["All time"]
theme.tiles([
    {"label": "Annualized, all time", "value": theme.pct(all_time["ann"], 2),
     "sign": all_time["ann"],
     "sub": f'{bench} {theme.pct(all_time["bench_ann"], 2)}'},
    {"label": "Max drawdown", "value": theme.pct(all_time["mdd"], 1),
     "sign": all_time["mdd"], "sub": "peak to trough, daily closes"},
    {"label": "Volatility", "value": f'{all_time["vol"] * 100:.1f}%',
     "sub": "annualized, daily returns"},
    {"label": "Since", "value": f'{track.base:%d %b %Y}',
     "sub": ("first deposit" if which == "total"
             else "close before the first purchase")},
])

# ── growth ────────────────────────────────────────────────────────────────────
with theme.panel("Growth of the fund vs benchmark",
                 note="cumulative %, rebased to the start of the window"):
    wl, wr = st.columns([3, 2], vertical_alignment="center")
    with wl:
        window = st.segmented_control(
            "Window", list(WINDOWS), default="All", key="window",
            label_visibility="collapsed") or "All"
    with wr:
        show_dd = st.checkbox("Shade the deepest drawdown", value=True,
                              key="show_dd")

    base = returns_mod.period_base(track.index, window, WINDOWS[window], a.asof)
    series = {track.name: track.index.loc[base:]}
    if track.benchmark is not None:
        series[bench] = track.benchmark.loc[base:]

    dd = None
    if show_dd:
        peak, trough = returns_mod.drawdown_window(track.index.loc[base:])
        dd = (peak, trough)

    st.plotly_chart(
        charts.growth_chart(series, highlight=track.name, drawdown=dd),
        use_container_width=True, config=theme.PLOTLY_CONFIG,
        key=f"growth_{which}_{window}")

    by_period = periods.set_index("period")
    win = by_period.loc[WINDOW_PERIOD.get(window, "All time")]
    excess = (None if win["bench_cum"] is None or pd.isna(win["bench_cum"])
              else win["cum"] - win["bench_cum"])
    theme.hover_strip(
        f'<strong>{track.name}</strong> <span class="cf-mono">'
        f'{theme.pct(win["cum"])}</span> vs <strong>{bench}</strong> '
        f'<span class="cf-mono">{theme.pct(win["bench_cum"])}</span>'
        + (f' · <span class="cf-mono">{theme.pct(excess)}</span> excess'
           if excess is not None else "")
        + f' · from {win["start"]:%d %b %Y} to {win["end"]:%d %b %Y}'
        + f' · worst drawdown <span class="cf-mono">'
          f'{theme.pct(win["mdd"], 1)}</span>')

# ── cash ──────────────────────────────────────────────────────────────────────
with theme.panel("Cash weight",
                 note="sweep cash plus the SNOXX money market, as a share of "
                      "net assets"):
    cw = book.cash_weight()
    st.plotly_chart(charts.weight_chart(cw.loc[base:]),
                    use_container_width=True, config=theme.PLOTLY_CONFIG,
                    key=f"cashw_{window}")
    st.markdown(
        f'<div style="font-size:11.5px;color:{theme.TEXT_DIM};line-height:1.6">'
        f'Now <span class="cf-mono" style="color:{theme.TEXT}">'
        f'{cash_w * 100:.1f}%</span> · '
        f'median over the window <span class="cf-mono">'
        f'{cw.loc[base:].median() * 100:.1f}%</span> · '
        f'highest <span class="cf-mono">{cw.loc[base:].max() * 100:.1f}%</span> '
        f'on {cw.loc[base:].idxmax():%d %b %Y}. '
        f'Cash earns the money-market rate, so a high weight is the main reason '
        f'the total portfolio trails the equity sleeve.</div>',
        unsafe_allow_html=True)

# ── sleeves at a glance ───────────────────────────────────────────────────────
with theme.panel("Sleeves today"):
    rows = []
    for key in ("equity", "fixed_income", "cash"):
        v = float(book.sleeve_value[key].iloc[-1])
        t = a.tracks.get(key)
        rows.append({
            "Sleeve": config.SLEEVE_LABEL[key],
            "Value": theme.money(v),
            "Weight": f"{v / nav * 100:.1f}%",
            # by value, against the shared materiality floor: a dozen
            # long-exited names carry sub-share remnants worth a dollar or two
            "Holdings": int((book.mv[book.sleeve_members[key]].iloc[-1].abs()
                             >= config.MATERIAL_POSITION_VALUE).sum())
            if book.sleeve_members[key] else 0,
            "Since": f"{t.base:%b %Y}" if t else "—",
            "Return since": theme.pct(t.index.iloc[-1] - 1, 2) if t else "—",
            "Benchmark": (t.benchmark_name or "—") if t else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ── the reconciliation note ───────────────────────────────────────────────────
gap = data.cash_gap()
if gap and abs(gap["diff"]) > 1.0:
    theme.note(
        f'<strong>Known reconciliation gap.</strong> On '
        f'{gap["as_of"]:%d %b %Y} the rebuilt cash balance is '
        f'<span class="cf-mono">{theme.money(gap["rebuilt"], 2)}</span> against '
        f'the custodian\'s <span class="cf-mono">'
        f'{theme.money(gap["custodian"], 2)}</span> — a difference of '
        f'<span class="cf-mono">{theme.money(gap["diff"], 2)}</span>, '
        f'{abs(gap["share_of_total"]) * 100:.2f}% of net assets. The rebuild '
        f'holds <em>more</em> cash than the custodian, which means the export is '
        f'missing an outflow. It is reported rather than plugged: a balancing '
        f'entry would make this line green and the history wrong. Every other '
        f'sleeve reconciles — see Methodology.')
