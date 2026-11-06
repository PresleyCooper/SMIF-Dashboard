"""
Factor exposure — what kind of book this is, against the S&P 500.

The chart answers "what do I own", in standard deviations from the index. It
does not answer "how much market am I taking" — that is beta, and it is on the
page underneath, because a style chart with no market sensitivity on it invites
exactly the wrong conclusion from a book that is 90% long equities.

Two things the page is careful to say out loud:

  COVERAGE IS NOT 100% AND SHOULD NOT PRETEND TO BE. Funds have no
  fundamentals, and some companies are missing a metric. Missing data is
  excluded and the remaining weight renormalised, never imputed as zero —
  imputing zero would be a fabricated claim that a holding is exactly average.
  The share of weight actually scored is shown beside every number.

  THE SNAPSHOT HAS A DATE. Fundamentals are crawled, not live. A chart built
  from a month-old crawl is fine; one built from a year-old crawl is fiction,
  so the age is on the page and goes amber when it is stale.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from smif import config, correlations as corr_mod, factors as fx
from ui import charts, data, theme

theme.apply_theme()
data.require_data()

a = data.analytics()
book = a.book

theme.topbar("Factor exposure",
             f"Characteristics of what the fund holds, z-scored against the "
             f"S&P 500 · holdings as of {a.asof:%d %b %Y}")

snapshot = fx.load_snapshot()
universe = snapshot.get("metrics", {})

# ── nothing to show yet ───────────────────────────────────────────────────────
if len(universe) < config.FACTOR_MIN_UNIVERSE:
    theme.note(
        "<strong>The reference universe has not been built yet.</strong> "
        "Factor scores are relative to the S&P 500, so there is nothing to "
        "score against until the constituents' fundamentals have been "
        "crawled. It takes several minutes and is then cached for a month.")
    st.code("python scripts/build_factors.py", language="bash")
    st.stop()

# ── weights: companies vs funds ───────────────────────────────────────────────
holdings = book.holdings_on(min_value=config.MATERIAL_POSITION_VALUE)
total_value = float(holdings["value"].sum())

funds = [s for s in holdings.index if fx.is_fund(s, snapshot)]
stocks = [s for s in holdings.index if s not in funds]

stock_weights = {s: float(holdings.loc[s, "value"]) / total_value for s in stocks}
fund_pct = sum(float(holdings.loc[s, "value"]) for s in funds) / total_value * 100

absolute = fx.exposure(stock_weights, fx.holding_metrics(snapshot, stocks),
                       universe)

# The index's own scores, at its real capitalisation weights. Without these the
# chart invites the reader to treat "vs the average constituent" as "vs the
# S&P 500", which on Size alone is the difference between +1.78 and +0.03.
index_weights = {}
try:
    from smif import workbook
    sp = workbook.read_sp500()
    index_weights = {r.symbol: float(r.weight) for r in sp.itertuples()
                     if r.symbol in universe}
except Exception:
    index_weights = {}

benchmark = (fx.benchmark_exposure(universe, index_weights)
             if len(index_weights) >= config.FACTOR_MIN_UNIVERSE else None)

BASIS = {
    "Active — vs the S&P 500": "active",
    "Absolute — vs the average constituent": "absolute",
}
basis = "absolute"
if benchmark is not None:
    label = st.segmented_control(
        "Compared to", list(BASIS), default="Active — vs the S&P 500",
        key="fx_basis", label_visibility="collapsed")
    basis = BASIS.get(label or "Active — vs the S&P 500", "active")

result = (fx.active_exposure(absolute, benchmark)
          if (basis == "active" and benchmark is not None) else absolute)

age = fx.snapshot_age_days(snapshot)
stale = fx.is_stale(snapshot)
asof_text = (pd.Timestamp(snapshot["asof"]).strftime("%d %b %Y")
             if snapshot.get("asof") else "unknown")

theme.tiles([
    {"label": "Stock weight covered", "value": f"{result['coverage']:.0f}%",
     "sub": f"{len(stocks)} companies scored"},
    {"label": "In funds, not scored", "value": f"{fund_pct:.0f}%",
     "sub": ", ".join(funds) if funds else "none"},
    {"label": "Universe", "value": f"{result['universeN']}",
     "sub": "S&P 500 constituents"},
    {"label": "Fundamentals as of", "value": asof_text,
     "sub": (f"{age:.0f} days old — stale" if stale and age is not None
             else f"{age:.0f} days old" if age is not None else "unknown")},
])

if snapshot.get("status") == "partial":
    # build_snapshot() checkpoints every 25 names so a crawl that dies at
    # minute seven does not lose everything. The side effect is that a crawl
    # still running is perfectly readable — and would otherwise be presented
    # as a finished one, with a coverage figure that is really just "how far
    # the alphabet got".
    theme.note(
        f"<strong>This crawl is incomplete.</strong> {snapshot.get('n', 0)} "
        f"names are in the snapshot so far, written as a checkpoint while the "
        f"crawl is still running. The scores below are computed only from the "
        f"holdings already covered, so the coverage figure reflects the "
        f"crawl's progress rather than the data's true availability. Let it "
        f"finish and reload.")
elif stale:
    theme.note(
        f"<strong>This snapshot is {age:.0f} days old.</strong> Fundamentals "
        f"are crawled, not live, and the scores above are only as current as "
        f"the crawl. Refresh below, or run "
        f"<code>python scripts/build_factors.py</code>.")

# ── the chart ─────────────────────────────────────────────────────────────────
with theme.panel(
        "Factor exposure — vs S&P 500",
        note=f"standard deviations · {result['coverage']:.0f}% of stock weight "
             f"covered · {fund_pct:.0f}% in funds, not scored"):
    show_metrics = st.checkbox("Show the underlying metrics", value=True,
                               key="fx_metrics")
    svg = charts.factor_exposure_svg(result, show_metrics=show_metrics)
    st.markdown(f'<div style="overflow-x:auto">{svg}</div>',
                unsafe_allow_html=True)
    if basis == "active":
        basis_note = (
            "<strong>Zero is the S&amp;P 500 itself</strong>, at its real "
            "capitalisation weights. What the z-scores measure on their own is "
            "distance from the AVERAGE CONSTITUENT, and the index is nothing "
            "like its average constituent — it scores "
            f"{benchmark['factors'][0]['z']:+.2f} on Size by that yardstick. "
            "These bars subtract it, so what is left is the fund's tilt "
            "against the thing it is benchmarked to.")
    else:
        basis_note = (
            "<strong>Zero is the average S&amp;P 500 constituent</strong>, not "
            "the index. Each constituent is one observation in the "
            "distribution, so the cap-weighted index itself is not at zero — "
            f"it sits at {benchmark['factors'][0]['z']:+.2f} on Size. Switch to "
            "Active to measure against the index instead."
            if benchmark is not None else
            "<strong>Zero is the average S&amp;P 500 constituent.</strong> The "
            "index's own weights are unavailable, so the cap-weighted "
            "comparison cannot be shown.")
    theme.hover_strip(
        "Up means larger, higher quality, faster growing, cheaper and stronger "
        "momentum — the intuitive convention, not the academic one (the "
        "academic size factor is the inverse of this). " + basis_note
        + " <strong>Hover any bar</strong> for its definition, its coverage and "
        "the five holdings contributing most. Size and Momentum are coloured "
        "neutrally: bigger and faster are not virtues.")

# ── the numbers behind the chart ──────────────────────────────────────────────
with theme.panel("Metric by metric",
                 note="weighted z, the coverage behind it, and what drives it"):
    rows = []
    for m in result["metrics"]:
        top = " · ".join(f"{c['ticker']} {c['z']:+.1f}" for c in m["top"])
        rows.append({
            "Factor": m["factor"], "Metric": m["label"],
            "z": m["z"],
            "Fund": m.get("absolute_z", m["z"]),
            "S&P 500": m.get("benchmark_z"),
            "Coverage": (m["coverage"] or 0) / 100.0,
            "Names": m["n"], "Biggest contributors": top or "—",
        })
    table = pd.DataFrame(rows)
    num_cols = ["z", "Fund", "S&P 500", "Coverage"]
    numeric = table[num_cols].apply(pd.to_numeric, errors="coerce")
    text = table.copy()
    for c in ("z", "Fund", "S&P 500"):
        text[c] = numeric[c].map(lambda v: "—" if pd.isna(v) else f"{v:+.2f}")
    text = text.rename(columns={"z": "Active" if basis == "active" else "Score"})
    text["Coverage"] = numeric["Coverage"].map(
        lambda v: "—" if pd.isna(v) else f"{v * 100:.0f}%")
    css = pd.DataFrame("", index=text.index, columns=text.columns)
    css[text.columns[2]] = numeric["z"].map(
        lambda v: f"color: {theme.TEXT_DIM}" if pd.isna(v)
        else f"color: {theme.GREEN if v > 0 else theme.RED}")
    css["Coverage"] = numeric["Coverage"].map(
        lambda v: f"color: {theme.YELLOW}" if (not pd.isna(v) and v < 0.9)
        else f"color: {theme.TEXT_MUTED}")
    st.dataframe(text.style.apply(lambda _: css, axis=None), hide_index=True,
                 use_container_width=True, height=36 * len(text) + 42)
    st.caption("Coverage below 90% is shown in amber: that metric's score "
               "speaks for less of the book than the others.")

# ── market sensitivity ────────────────────────────────────────────────────────
with theme.panel("Beta to SPY",
                 note="three years of daily returns — the style chart above "
                      "says nothing about market exposure"):
    end = a.asof
    start = end - pd.DateOffset(years=3)
    sleeve_rets = corr_mod.sleeve_returns(a.tracks, a.store, start, end, "Daily")
    spy = sleeve_rets.get("SPY")

    if spy is None or spy.dropna().empty:
        st.caption("SPY returns are not available for this window.")
    else:
        betas = corr_mod.beta_table(
            sleeve_rets[[c for c in sleeve_rets.columns if c != "SPY"]],
            spy.dropna())
        tiles = []
        for label in ("Total portfolio", "Equity", "Fixed income"):
            if label in betas.index and not pd.isna(betas.loc[label, "beta"]):
                r = betas.loc[label]
                tiles.append({
                    "label": f"{label} beta", "value": f"{r['beta']:.2f}",
                    "sub": f"R² {r['r2']:.2f} · {int(r['obs'])} days"})
        if tiles:
            theme.tiles(tiles)
        theme.hover_strip(
            "The equity sleeve's beta is the honest read on market exposure; "
            "the total portfolio's is lower only because cash has no beta. "
            "R² says how much of each sleeve's movement SPY actually explains "
            "— a beta without it is a slope with no claim attached.")

# ── refresh ───────────────────────────────────────────────────────────────────
with theme.panel("Refresh fundamentals"):
    st.markdown(
        f'<div style="font-size:12px;color:{theme.TEXT_MUTED};line-height:1.6">'
        f'Re-crawls the S&amp;P 500 constituents and the fund\'s own holdings: '
        f'a price panel for momentum, then one fundamentals call per name. '
        f'Several minutes, checkpointed, then cached for '
        f'{config.FUNDAMENTALS_TTL_DAYS} days.</div>',
        unsafe_allow_html=True)

    if st.button("Refresh fundamentals", type="primary", key="fx_refresh"):
        from smif import workbook
        sp = workbook.read_sp500()
        tickers = sorted(set(sp["symbol"]) | set(stocks))
        bar = st.progress(0.0, text="Downloading the price panel…")
        panel = fx.fetch_universe_prices(
            tickers, end=a.asof,
            progress=lambda i, n, t: bar.progress(
                min(i / max(n, 1) * 0.3, 0.3), text=f"Prices: {i}/{n} ({t})"))
        out = fx.build_snapshot(
            tickers, price_panel=panel,
            progress=lambda i, n, t: bar.progress(
                0.3 + 0.7 * i / max(n, 1), text=f"Fundamentals: {i}/{n} ({t})"))
        bar.empty()
        st.success(f"Crawled {out['n']} names "
                   f"({out['fetched']} fetched, {out['failed']} failed).")
        data.refresh()
        st.rerun()

    cov = fx.coverage_table(snapshot)
    if not cov.empty:
        cov_text = cov.copy()
        cov_text["share"] = cov["share"].map(
            lambda v: "—" if pd.isna(v) else f"{v * 100:.0f}%")
        st.dataframe(cov_text.rename(columns={
            "metric": "Metric", "factor": "Factor", "covered": "Names",
            "share": "Universe coverage"}), hide_index=True,
            use_container_width=True, height=36 * len(cov_text) + 42)
