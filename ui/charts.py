"""
ui/charts.py — the figures, all starting from one layout.

Ported from Confluence's chart components (`PortfolioCharts.jsx`,
`FactorExposure.jsx`) with their colour discipline intact: the portfolio line
carries the coral accent because it is the one thing on the page you came to
look at, and the benchmark is a neutral grey reference. "Us" versus "the
market", stated in colour.

The factor chart is hand-written SVG rather than a Plotly bar chart, for the
same reason the React version is: it needs group dividers, two bar widths, a
thick zero line through the middle and a bold score floating above each
composite bar. Plotly can be bent into that shape but not cheaply, and the
result would not match the rest of the fund's reporting.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme as T


# ── growth ────────────────────────────────────────────────────────────────────

def growth_chart(series: dict[str, pd.Series], *, height: int = 400,
                 highlight: str | None = None,
                 drawdown: tuple | None = None) -> go.Figure:
    """
    Cumulative percent growth, every series rebased to 0% on the first common
    day.

    The highlighted series gets the coral accent and a heavier line; everything
    else is a comparison. Rebasing happens here rather than upstream so a chart
    can never show two series measured from different days.
    """
    fig = go.Figure()
    first_key = highlight or next(iter(series), None)
    for name, s in series.items():
        if s is None or s.dropna().empty:
            continue
        s = s.dropna()
        y = (s / s.iloc[0] - 1) * 100
        is_lead = name == first_key
        fig.add_trace(go.Scatter(
            x=y.index, y=y.values, name=name, mode="lines",
            line=dict(color=T.ACCENT if is_lead else T.GRAY,
                      width=2.1 if is_lead else 1.7),
            hovertemplate="%{y:+.2f}%<extra>" + name + "</extra>"))

    shapes, annotations = [], []
    if drawdown and drawdown[0] is not None and drawdown[0] != drawdown[1]:
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=drawdown[0], x1=drawdown[1], y0=0, y1=1,
                           fillcolor="rgba(148,163,184,0.13)",
                           line=dict(width=0), layer="below"))
        annotations.append(dict(x=drawdown[0], xref="x", y=1, yref="paper",
                                yanchor="bottom", xanchor="left",
                                text="max drawdown", showarrow=False,
                                font=dict(size=9.5, color="#e0913a")))

    fig.update_layout(**T.plotly_layout(
        height=height, shapes=shapes, annotations=annotations,
        xaxis=dict(tickformat="%b '%y"),
        yaxis=dict(ticksuffix="%")))
    return fig


def multi_sleeve_chart(tracks: dict, *, height: int = 400) -> go.Figure:
    """Every sleeve on one axis, each in its own hue, benchmark in grey."""
    fig = go.Figure()
    for key, track in tracks.items():
        s = track.index.dropna()
        if s.empty:
            continue
        y = (s / s.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(
            x=y.index, y=y.values, name=track.name, mode="lines",
            line=dict(color=T.SLEEVE_COLOR.get(key, T.GRAY), width=1.9),
            hovertemplate="%{y:+.2f}%<extra>" + track.name + "</extra>"))
    fig.update_layout(**T.plotly_layout(
        height=height, xaxis=dict(tickformat="%b '%y"),
        yaxis=dict(ticksuffix="%")))
    return fig


def weight_chart(weight: pd.Series, *, label: str = "Cash weight",
                 height: int = 220) -> go.Figure:
    """A filled area for a share-of-portfolio series."""
    w = weight.dropna() * 100
    fig = go.Figure(go.Scatter(
        x=w.index, y=w.values, name=label, mode="lines",
        line=dict(color=T.YELLOW, width=1.6),
        fill="tozeroy", fillcolor="rgba(251,191,36,0.13)",
        hovertemplate="%{y:.1f}%<extra>" + label + "</extra>"))
    fig.update_layout(**T.plotly_layout(
        height=height, margin=dict(l=54, r=18, t=8, b=30),
        xaxis=dict(tickformat="%b '%y"),
        yaxis=dict(ticksuffix="%", rangemode="tozero")))
    return fig


def calendar_bars(cal: pd.DataFrame, *, benchmark_name: str = "SPY",
                  height: int = 300) -> go.Figure:
    """
    Calendar-year returns as paired bars.

    Partial years (the fund's first and the current one) are drawn with a
    lighter fill and said so in the hover, rather than presented as if they
    were full years.
    """
    if cal.empty:
        return go.Figure()
    opacity = [0.55 if p else 1.0 for p in cal["partial"]]
    tag = ["  (partial year)" if p else "" for p in cal["partial"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cal["year"], y=cal["portfolio"] * 100, name="Fund",
        marker=dict(color=T.ACCENT, opacity=opacity),
        customdata=tag,
        hovertemplate="%{y:+.2f}%%{customdata}<extra>Fund</extra>"))
    if cal["benchmark"].notna().any():
        fig.add_trace(go.Bar(
            x=cal["year"], y=cal["benchmark"] * 100, name=benchmark_name,
            marker=dict(color=T.GRAY, opacity=opacity),
            customdata=tag,
            hovertemplate="%{y:+.2f}%%{customdata}<extra>"
                          + benchmark_name + "</extra>"))
    fig.update_layout(**T.plotly_layout(
        height=height, barmode="group", bargap=0.26, bargroupgap=0.08,
        hovermode="x unified",
        xaxis=dict(showspikes=False, type="category", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(ticksuffix="%")))
    return fig


# ── correlation ───────────────────────────────────────────────────────────────

def correlation_heatmap(matrix: pd.DataFrame, *, height: int | None = None
                        ) -> go.Figure:
    """
    A diverging heatmap centred on zero.

    zmid is set explicitly rather than letting Plotly infer the midpoint from
    the data: a matrix whose weakest pair is +0.2 would otherwise colour that
    as "negative", which is exactly the misreading a centred scale exists to
    prevent.
    """
    labels = list(matrix.columns)
    n = len(labels)
    h = height or max(360, n * 21 + 110)
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=labels, y=labels,
        zmin=-1, zmax=1, zmid=0,
        colorscale=T.correlation_scale(),
        hovertemplate="%{y} x %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=10, tickfont=dict(size=9, color=T.TEXT_MUTED),
                      outlinewidth=0, len=0.82)))
    fig.update_layout(**T.plotly_layout(
        height=h, hovermode="closest",
        margin=dict(l=76, r=10, t=10, b=76),
        xaxis=dict(tickfont=dict(size=9), showspikes=False, side="bottom",
                   gridcolor="rgba(0,0,0,0)", tickangle=-90),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed",
                   gridcolor="rgba(0,0,0,0)", zeroline=False)))
    return fig


def beta_scatter(table: pd.DataFrame, *, height: int = 380) -> go.Figure:
    """
    Each holding's beta to SPY against its correlation, bubble-sized by weight.

    The two together say something neither says alone: a high beta on a low
    correlation is mostly idiosyncratic volatility, not market exposure.
    """
    if table.empty:
        return go.Figure()
    size = table["weight"].fillna(0)
    size = 8 + 34 * (size / size.max() if size.max() else size)
    fig = go.Figure(go.Scatter(
        x=table["beta"], y=table["corr_spy"], mode="markers+text",
        text=table.index, textposition="top center",
        textfont=dict(size=8.5, color=T.TEXT_DIM),
        marker=dict(size=size, color=T.ACCENT, opacity=0.62,
                    line=dict(width=0.8, color=T.BORDER_2)),
        customdata=np.c_[table["weight"] * 100],
        hovertemplate="<b>%{text}</b><br>beta %{x:.2f}<br>"
                      "corr %{y:.2f}<br>weight %{customdata[0]:.1f}%"
                      "<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=T.ZERO, width=1))
    fig.add_vline(x=1, line=dict(color=T.ZERO, width=1, dash="dot"))
    fig.update_layout(**T.plotly_layout(
        height=height, hovermode="closest", showlegend=False,
        xaxis=dict(title="beta to SPY", showspikes=False),
        yaxis=dict(title="correlation to SPY", zeroline=False)))
    return fig


# ── factor exposure ───────────────────────────────────────────────────────────

_W, _H = 960, 310
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 46, 14, 30, 50
_Z_MAX = 2.5

# Quality, Growth and Value carry a verdict: higher IS better, so green/red is
# honest. Size and Momentum do not — "bigger" and "more momentum" are not
# virtues, and colouring them green would state an opinion the chart is not
# making. They get a neutral cool/warm pair instead.
_VERDICT = {"Quality", "Growth", "Value"}
_NEUTRAL_POS, _NEUTRAL_NEG = T.BLUE, T.AMBER
_DIRECTION = {"Size": "larger ↑", "Value": "cheaper ↑",
              "Momentum": "stronger ↑", "Quality": "higher ↑",
              "Growth": "faster ↑"}


def _bar_color(factor: str, z: float) -> str:
    if factor in _VERDICT:
        return T.GREEN if z >= 0 else T.RED
    return _NEUTRAL_POS if z >= 0 else _NEUTRAL_NEG


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def factor_exposure_svg(payload: dict, *, show_metrics: bool = True) -> str:
    """
    The five-group factor chart, in standard deviations against the S&P 500.

    Hover detail rides on SVG <title> elements: the browser renders them as
    native tooltips, which keeps the whole figure a single self-contained
    string with no callback round-trip. Each bar's tooltip carries the metric's
    definition, its coverage and the five holdings contributing most.
    """
    groups = payload.get("factors") or []
    metrics = payload.get("metrics") or []
    if not groups:
        return ""

    plot_w = _W - _PAD_L - _PAD_R
    plot_h = _H - _PAD_T - _PAD_B
    zero_y = _PAD_T + plot_h / 2
    group_w = plot_w / len(groups)

    def y_of(z: float) -> float:
        z = max(-_Z_MAX, min(_Z_MAX, z))
        return zero_y - (z / _Z_MAX) * (plot_h / 2)

    out = [f'<svg viewBox="0 0 {_W} {_H}" style="width:100%;min-width:620px" '
           f'role="img" aria-label="Portfolio factor exposure versus the '
           f'S&amp;P 500" xmlns="http://www.w3.org/2000/svg">']

    # sigma gridlines
    for g in (-2, -1, 1, 2):
        y = y_of(g)
        out.append(
            f'<line x1="{_PAD_L}" x2="{_W - _PAD_R}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="{T.BORDER}" stroke-dasharray="3 4" stroke-width="1"/>'
            f'<text x="{_PAD_L - 7}" y="{y + 3.5:.1f}" text-anchor="end" '
            f'font-size="9.5" fill="{T.TEXT_DIM}" '
            f'font-family="DM Mono, monospace">{g:+d}&#963;</text>')

    # direction hints and the zero line
    out.append(
        f'<text x="{_PAD_L - 7}" y="{_PAD_T + 9}" text-anchor="end" '
        f'font-size="9" fill="{T.TEXT_DIM}">more</text>'
        f'<text x="{_PAD_L - 7}" y="{_H - _PAD_B - 2}" text-anchor="end" '
        f'font-size="9" fill="{T.TEXT_DIM}">less</text>'
        f'<line x1="{_PAD_L}" x2="{_W - _PAD_R}" y1="{zero_y}" y2="{zero_y}" '
        f'stroke="{T.TEXT_MUTED}" stroke-width="2.5"/>')

    for gi, g in enumerate(groups):
        x0 = _PAD_L + gi * group_w
        factor = g["factor"]
        mine = [m for m in metrics
                if m["factor"] == factor and m.get("z") is not None] \
            if show_metrics else []
        slot = group_w / (len(mine) + 2)
        comp_w = min(34.0, slot * 0.8)
        met_w = min(15.0, slot * 0.62)

        if gi > 0:
            out.append(
                f'<line x1="{x0:.1f}" x2="{x0:.1f}" y1="{_PAD_T - 6}" '
                f'y2="{_H - _PAD_B + 6}" stroke="{T.TEXT_MUTED}" '
                f'stroke-width="2.5"/>')

        # the composite bar
        if g.get("z") is not None:
            z = g["z"]
            cx = x0 + slot
            y = y_of(z)
            tip = (f'{factor} composite: {z:+.2f}σ\n'
                   f'Mean of {g.get("nMetrics", 0)} metric z-scores.\n'
                   f'{g.get("coverage", 0):.0f}% of stock weight covered.')
            out.append(
                f'<g><title>{_esc(tip)}</title>'
                f'<rect x="{cx - comp_w / 2:.1f}" y="{min(y, zero_y):.1f}" '
                f'width="{comp_w:.1f}" height="{max(2, abs(y - zero_y)):.1f}" '
                f'fill="{_bar_color(factor, z)}" opacity="0.9" rx="1.5"/>'
                f'<text x="{cx:.1f}" y="{(y - 6) if z >= 0 else (y + 13):.1f}" '
                f'text-anchor="middle" font-size="11.5" font-weight="700" '
                f'fill="{T.TEXT}" font-family="DM Mono, monospace">'
                f'{z:+.2f}</text></g>')

        # the per-metric bars
        for mi, m in enumerate(mine):
            z = m["z"]
            cx = x0 + slot * (mi + 2)
            y = y_of(z)
            top = " · ".join(f'{c["ticker"]} {c["z"]:+.2f}'
                             for c in (m.get("top") or []))
            tip = (f'{m["label"]}: {z:+.2f}σ\n{m["desc"]}\n'
                   f'{m.get("coverage", 0):.0f}% of stock weight covered.')
            if top:
                tip += f'\nLargest contributors: {top}'
            out.append(
                f'<g style="cursor:help"><title>{_esc(tip)}</title>'
                f'<rect x="{cx - met_w / 2:.1f}" y="{min(y, zero_y):.1f}" '
                f'width="{met_w:.1f}" height="{max(2, abs(y - zero_y)):.1f}" '
                f'fill="{_bar_color(factor, z)}" opacity="0.42" rx="1"/>'
                f'<text x="{cx:.1f}" y="{(y - 4) if z >= 0 else (y + 11):.1f}" '
                f'text-anchor="middle" font-size="9" fill="{T.TEXT_MUTED}" '
                f'font-family="DM Mono, monospace">{z:+.1f}</text></g>')

        out.append(
            f'<text x="{x0 + group_w / 2:.1f}" y="{_H - _PAD_B + 22}" '
            f'text-anchor="middle" font-size="12" font-weight="600" '
            f'fill="{T.TEXT}">{_esc(factor)}</text>')
        if g.get("z") is not None:
            out.append(
                f'<text x="{x0 + group_w / 2:.1f}" y="{_H - _PAD_B + 35}" '
                f'text-anchor="middle" font-size="9" fill="{T.TEXT_DIM}">'
                f'{_DIRECTION.get(factor, "higher ↑")}</text>')

    out.append("</svg>")
    return "".join(out)
