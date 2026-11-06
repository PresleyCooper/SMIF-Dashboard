"""
Correlations — what is actually diversifying what.

Three views, because the word means three different questions at three levels:
which names move together, which sectors do, and whether the sleeves are doing
different jobs. The window and the frequency are the reader's choice because
they change the answer: daily correlations understate genuine co-movement, and
a monthly three-year window is only 36 observations.

The scale is diverging and centred on zero, in teal-to-coral rather than
red-to-green. Elsewhere in this app green means favourable; a -0.4 correlation
is not bad news, it is the diversification you were looking for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from smif import config, correlations as corr_mod
from ui import charts, data, theme

theme.apply_theme()
data.require_data()

a = data.analytics()
book = a.book

WINDOWS = {"1Y": 1, "3Y": 3, "5Y": 5, "Custom": None}
VIEWS = ["Holdings", "Sectors", "Sleeves vs benchmarks"]

theme.topbar(
    "Correlations",
    f"Total-return correlations of what the fund holds today · "
    f"as of {a.asof:%d %b %Y}")

# ── controls ──────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([2, 2, 3], vertical_alignment="center")
with c1:
    view = st.segmented_control("View", VIEWS, default="Holdings", key="corr_view",
                                label_visibility="collapsed") or "Holdings"
with c2:
    window = st.segmented_control("Window", list(WINDOWS), default="3Y",
                                  key="corr_window",
                                  label_visibility="collapsed") or "3Y"
with c3:
    freq = st.segmented_control("Frequency", list(corr_mod.FREQUENCIES),
                                default="Daily", key="corr_freq",
                                label_visibility="collapsed") or "Daily"

end = a.asof
if WINDOWS[window] is None:
    picked = st.date_input(
        "Custom window",
        value=((end - pd.DateOffset(years=3)).date(), end.date()),
        min_value=book.days[0].date(), max_value=end.date(),
        key="corr_range", label_visibility="collapsed")
    if isinstance(picked, tuple) and len(picked) == 2:
        start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
    else:
        st.caption("Pick an end date.")
        st.stop()
else:
    start = end - pd.DateOffset(years=WINDOWS[window])

_rule, min_obs = corr_mod.FREQUENCIES[freq]

# ── returns for the chosen view ───────────────────────────────────────────────
holdings = book.holdings_on(min_value=config.MATERIAL_POSITION_VALUE)
equity_syms = [s for s in holdings.index if holdings.loc[s, "sleeve"] == "equity"]
weights = corr_mod.current_weights(book, sleeve="equity")

holding_rets, dropped = corr_mod.price_returns(a.store, equity_syms, start, end,
                                               freq)

if view == "Holdings":
    rets = holding_rets
    caption = "current equity holdings"
elif view == "Sectors":
    sectors = {}
    try:
        from smif import workbook
        sectors = workbook.sector_map()
    except Exception:
        sectors = {}
    rets = corr_mod.sector_returns(holding_rets, weights, sectors)
    caption = "GICS sectors, dollar-weighted from the fund's own holdings"
else:
    rets = corr_mod.sleeve_returns(a.tracks, a.store, start, end, freq)
    caption = "sleeves against their benchmarks"

if rets.empty or rets.shape[1] < 2:
    st.warning(
        f"Not enough {freq.lower()} data in this window to correlate anything. "
        f"Try a longer window or a higher frequency.")
    st.stop()

corr = corr_mod.correlation(rets, min_obs=min_obs)
order = corr_mod.cluster_order(corr)
corr = corr.loc[order, order]

# ── summary ───────────────────────────────────────────────────────────────────
avg = corr_mod.average_pair_correlation(corr)
w_for_stats = weights if view == "Holdings" else pd.Series(
    1.0 / corr.shape[0], index=corr.columns)
eff = corr_mod.effective_holdings(corr, w_for_stats)

theme.tiles([
    {"label": "Average pair", "value": theme.num(avg, 2),
     "sub": f"{corr.shape[0]} series, {freq.lower()}"},
    {"label": "Effective bets", "value": theme.num(eff, 1),
     "sub": f"out of {corr.shape[0]}"},
    {"label": "Observations", "value": f"{len(rets):,}",
     "sub": f"{start:%b %Y} – {end:%b %Y}"},
    {"label": "Frequency", "value": freq, "sub": caption},
])

# ── heatmap ───────────────────────────────────────────────────────────────────
with theme.panel(f"Correlation matrix — {caption}",
                 note=("clustered: names that move together sit together"
                       if corr_mod.clustering_available()
                       else "alphabetical — install SciPy to cluster")):
    st.plotly_chart(charts.correlation_heatmap(corr), use_container_width=True,
                    config=theme.PLOTLY_CONFIG, key=f"heat_{view}_{window}_{freq}")
    if not corr_mod.clustering_available():
        theme.note(
            "<strong>SciPy is not installed</strong>, so the matrix is in "
            "alphabetical order rather than clustered. The numbers are "
            "unaffected; only the ordering is.")

# ── the sortable table ────────────────────────────────────────────────────────
market = rets.get("SPY")
if market is None:
    bench = corr_mod.benchmark_returns(a.store, start, end, freq)
    market = bench.get("SPY", pd.Series(dtype=float))

with theme.panel("Each series against SPY",
                 note=f"beta, correlation and volatility from the same "
                      f"{freq.lower()} returns; volatility is annualized"):
    table = corr_mod.beta_table(
        rets, market, weights=weights if view == "Holdings" else None,
        min_obs=min_obs)
    table = table.reindex(order)

    out = pd.DataFrame({
        "Series": table.index,
        "Weight": table.get("weight"),
        "Corr to SPY": table["corr_spy"],
        "Beta": table["beta"],
        "R²": table["r2"],
        # annualized, so it must not be labelled by the sampling frequency:
        # "Daily volatility 15.4%" would be a startling thing to claim
        "Volatility (ann.)": table["vol"] * np.sqrt(
            {"Daily": 252, "Weekly": 52, "Monthly": 12}[freq]),
        "Obs": table["obs"],
    })
    if out["Weight"].isna().all():
        out = out.drop(columns=["Weight"])

    numeric = out.drop(columns=["Series", "Obs"]).apply(pd.to_numeric,
                                                         errors="coerce")
    text = out.copy()
    for col in numeric.columns:
        if col in ("Weight", "Volatility (ann.)"):
            text[col] = numeric[col].map(
                lambda v: "—" if pd.isna(v) else f"{v * 100:,.1f}%")
        else:
            text[col] = numeric[col].map(
                lambda v: "—" if pd.isna(v) else f"{v:,.2f}")
    css = pd.DataFrame("", index=text.index, columns=text.columns)
    for col in ("Corr to SPY", "Beta"):
        css[col] = numeric[col].map(
            lambda v: f"color: {theme.TEXT_DIM}" if pd.isna(v)
            else f"color: {theme.GREEN if v > 0 else theme.RED}")
    st.dataframe(text.style.apply(lambda _: css, axis=None), hide_index=True,
                 use_container_width=True, height=min(620, 36 * len(text) + 42))

    theme.hover_strip(
        "<strong>Beta without R² is a slope with no claim attached.</strong> "
        "A beta of 1.4 that explains 9% of a name's variance is not a market "
        "bet, it is noise that happens to lean. Read the two together."
        + (f" {len(dropped)} name(s) excluded: "
           + ", ".join(f"{k} ({v})" for k, v in sorted(dropped.items()))
           if dropped and view == "Holdings" else ""))

# ── beta scatter ──────────────────────────────────────────────────────────────
if view == "Holdings" and table["beta"].notna().any():
    with theme.panel("Beta against correlation",
                     note="bubble size is the position's weight in the sleeve"):
        scatter = table.dropna(subset=["beta", "corr_spy"]).copy()
        if "weight" not in scatter.columns:
            scatter["weight"] = 1.0
        scatter["weight"] = scatter["weight"].fillna(0.0)
        st.plotly_chart(charts.beta_scatter(scatter), use_container_width=True,
                        config=theme.PLOTLY_CONFIG, key=f"beta_{window}_{freq}")
        theme.hover_strip(
            "Up and to the right is a name that moves with the market and does "
            "so reliably. High beta with low correlation — bottom right — is "
            "idiosyncratic volatility being mistaken for market exposure.")
