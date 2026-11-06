"""
ui/theme.py — Confluence's design language, ported to Streamlit.

Confluence is a Flask API with a React front end, so there are no components to
import; what carries over is the design system itself, lifted from
`confluence_saniti_theme.css` and `theme.jsx`:

  · a near-black #0b0b0b canvas with one tonal step (#161616) for panels
  · graphite hairlines, no lifted shadows, sharp 6px radii
  · ONE coral accent (#f36458) that belongs to the interactive layer only
  · mono uppercase "eyebrow" labels for section titles
  · IBM Plex Sans for text, DM Mono for every number

The one rule worth repeating because breaking it is so easy: COLOUR THAT MEANS
SOMETHING IS NOT CHROME. Coral is navigation, buttons and the one highlighted
series in a chart. Green and red mean favourable and unfavourable and are never
used for decoration — which is why the correlation heatmap below is built on a
cool/warm pair rather than on green/red, and why the Size and Momentum factor
bars are deliberately neutral: "bigger" and "more momentum" are not verdicts.

Plotly layouts take their colours from literal values here rather than reading
CSS custom properties, for the same reason the React app does: a figure is
built before the stylesheet can be queried, and a chart one toggle behind the
page is worse than a duplicated hex code.
"""

from __future__ import annotations

from contextlib import contextmanager

import pandas as pd
import streamlit as st

# ── tokens ────────────────────────────────────────────────────────────────────
BG = "#0b0b0b"
SURFACE = "#161616"
SURFACE_2 = "#202020"
BORDER = "#2b2b2b"
BORDER_2 = "#383838"
TEXT = "#ffffff"
TEXT_MUTED = "#b9b9b9"
TEXT_DIM = "#8a8a8a"

ACCENT = "#f36458"          # the Saniti coral — interactive layer only
ACCENT_HOVER = "#ff7266"
ACCENT_BG = "rgba(243,100,88,0.14)"

GREEN = "#4ade80"           # favourable
RED = "#f87171"             # unfavourable
CYAN = "#22d3ee"            # regime / neutral-positive
AMBER = "#fb923c"           # neutral-negative
YELLOW = "#fbbf24"          # caution: defaults, interpolated values
BLUE = "#38bdf8"
GRAY = "#8f8f8f"            # benchmarks and comparison series

FONT_BODY = "'IBM Plex Sans', system-ui, sans-serif"
FONT_MONO = "'DM Mono', 'Courier New', monospace"
RADIUS = "6px"

GRID = "rgba(255,255,255,0.07)"
ZERO = "rgba(255,255,255,0.16)"
SPIKE = "#8a8a8a"

# One hue per sleeve. Total keeps the coral accent because it is the fund's
# headline number; equity gets its own blue rather than sharing coral, since
# the two lines sit within a few percent of each other for most of the history
# and in one colour they are simply indistinguishable.
SLEEVE_COLOR = {
    "total": ACCENT,
    "equity": BLUE,
    "fixed_income": CYAN,
    "cash": YELLOW,
    "benchmark": GRAY,
}


def _css() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
:root {{
  --bg: {BG}; --surface: {SURFACE}; --surface-2: {SURFACE_2};
  --border: {BORDER}; --border-2: {BORDER_2};
  --text: {TEXT}; --text-muted: {TEXT_MUTED}; --text-dim: {TEXT_DIM};
  --accent: {ACCENT}; --accent-bg: {ACCENT_BG};
  --green: {GREEN}; --red: {RED}; --yellow: {YELLOW};
  --font-body: {FONT_BODY}; --font-mono: {FONT_MONO};
  --radius: {RADIUS};
}}

html, body, [class*="st-"], .stMarkdown, button, input, select, textarea {{
  font-family: var(--font-body);
}}
/* ...but NOT the icon spans. Streamlit draws its nav icons as ligatures in the
   Material Symbols font, so a blanket body-font override turns every icon back
   into the literal word "dashboard" or "trending_up", overlapping the label. */
span[data-testid="stIconMaterial"], .material-icons, .material-icons-outlined,
[class*="material-symbols"], [data-testid*="stIconMaterial"] {{
  font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}}
.stApp {{ background: var(--bg); }}

/* content column: Confluence's cf-content measure, so wide tables scroll
   inside the page rather than stretching it */
.block-container {{
  max-width: 1720px; padding: 1.1rem 1.9rem 4rem; }}

/* ── sidebar: near-black in both themes, coral brand mark ─────────────── */
section[data-testid="stSidebar"] {{
  background: #0b0b0b; border-right: 1px solid rgba(255,255,255,0.06); }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.1rem; }}

.cf-brand {{ display: flex; align-items: center; gap: 9px; padding: 2px 4px 14px; }}
.cf-brand-mark {{
  width: 27px; height: 27px; border-radius: 6px; flex: none;
  display: grid; place-items: center; background: var(--accent);
  color: #0b0b0b; font-size: 13px; font-weight: 700;
  font-family: var(--font-mono); }}
.cf-brand-name {{
  font-weight: 800; letter-spacing: -0.2px; color: #fff;
  font-size: 15px; line-height: 1.1; }}
.cf-brand-sub {{
  font-size: 9.5px; color: #797979; font-family: var(--font-mono);
  letter-spacing: 0.02em; }}

/* active nav item: a coral edge marker and a coral label, not a filled pill */
section[data-testid="stSidebarNav"] ul li a {{
  border-radius: var(--radius); font-size: 13.5px; font-weight: 500; }}
section[data-testid="stSidebarNav"] ul li a[aria-current="page"] {{
  background: var(--accent-bg); color: #fff; font-weight: 650;
  box-shadow: inset 3px 0 0 0 var(--accent); }}

/* ── the editorial card: flat, 1px hairline, no shadow ───────────────── */
/* Applied to Streamlit's own bordered container rather than to a hand-written
   <div>: each st.markdown call lands in its own DOM container, so an unclosed
   wrapper div auto-closes and nests nothing. */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 16px 6px; }}
.cf-panel {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 16px 18px; margin-bottom: 14px; }}

/* the mono "eyebrow" — Confluence's technical-label signal */
.cf-eyebrow {{
  font-family: var(--font-mono); font-size: 10.5px; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--text-muted); margin-bottom: 12px;
  display: flex; align-items: baseline; gap: 10px; }}
.cf-eyebrow .note {{
  margin-left: auto; text-transform: none; letter-spacing: 0;
  font-family: var(--font-body); font-size: 11.5px; color: var(--text-dim);
  font-weight: 400; }}

/* ── headline tiles (Confluence's cf-verdict-nums) ───────────────────── */
.cf-tiles {{
  display: flex; border: 1px solid var(--border);
  border-radius: var(--radius); overflow: hidden; margin-bottom: 14px;
  background: var(--surface); flex-wrap: wrap; }}
.cf-tile {{
  flex: 1 1 128px; padding: 12px 14px; text-align: center;
  border-right: 1px solid var(--border); min-width: 0; }}
.cf-tile:last-child {{ border-right: none; }}
.cf-tile-label {{
  font-size: 10px; color: var(--text-dim); font-family: var(--font-mono);
  text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 5px; }}
.cf-tile-value {{
  font-size: 22px; font-family: var(--font-mono); font-weight: 600;
  line-height: 1.1; }}
.cf-tile-sub {{
  font-size: 10.5px; color: var(--text-dim); margin-top: 4px;
  font-family: var(--font-mono); }}
.cf-up {{ color: var(--green); }}
.cf-down {{ color: var(--red); }}
.cf-flat {{ color: var(--text-muted); }}

/* ── topbar ──────────────────────────────────────────────────────────── */
.cf-page-title {{
  font-size: 19px; font-weight: 750; letter-spacing: -0.3px;
  margin-bottom: 1px; }}
.cf-page-sub {{ font-size: 11.5px; color: var(--text-muted); }}
.cf-topbar {{
  display: flex; align-items: flex-end; gap: 14px;
  border-bottom: 1px solid var(--border); padding-bottom: 12px;
  margin-bottom: 18px; flex-wrap: wrap; }}
.cf-idx-row {{ margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }}
.cf-idx {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 5px 12px; text-align: center;
  min-width: 86px; }}
.cf-idx .k {{
  font-size: 9.5px; color: var(--text-dim); font-weight: 600;
  font-family: var(--font-mono); text-transform: uppercase;
  letter-spacing: 0.05em; }}
.cf-idx .v {{
  font-size: 12.5px; font-weight: 700; margin-top: 2px;
  font-family: var(--font-mono); }}

/* ── the hover strip under a chart (Confluence's cf-fx-hover) ────────── */
.cf-hover {{
  min-height: 44px; margin-top: 8px; padding: 8px 11px;
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 4px; font-size: 11.5px; color: var(--text-muted);
  line-height: 1.5; }}
.cf-hover strong {{ color: var(--text); }}
.cf-mono {{ font-family: var(--font-mono); }}

/* ── notes: caution yellow for anything estimated or unreconciled ─────── */
.cf-note {{
  border-left: 2px solid var(--yellow); background: rgba(251,191,36,0.07);
  padding: 9px 13px; border-radius: 0 4px 4px 0; font-size: 12.5px;
  color: var(--text-muted); margin-bottom: 14px; line-height: 1.55; }}
.cf-note strong {{ color: var(--yellow); }}
.cf-note.ok {{
  border-left-color: var(--green); background: rgba(74,222,128,0.06); }}
.cf-note.ok strong {{ color: var(--green); }}

/* ── Streamlit widget chrome, flattened to the Saniti language ────────── */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
  border-radius: 4px; font-size: 12px; font-weight: 500;
  border: 1px solid var(--border-2); background: var(--surface);
  color: var(--text-muted); }}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: var(--text-muted); color: var(--text); }}
.stButton > button[kind="primary"], .stFormSubmitButton > button {{
  background: var(--accent); border-color: var(--accent);
  color: #0b0b0b; font-weight: 600; }}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {{
  background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; color: #0b0b0b; }}

div[data-baseweb="tab-list"] {{
  gap: 4px; border-bottom: 1px solid var(--border); }}
button[data-baseweb="tab"] {{ font-size: 13.5px; font-weight: 500; }}

div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
  border: 1px solid var(--border); border-radius: var(--radius); }}
div[data-testid="stDataFrame"] {{ font-family: var(--font-mono); }}

div[data-testid="stMetricValue"] {{
  font-family: var(--font-mono); font-weight: 600; }}

/* radios and segmented controls read as Confluence's chip row */
div[data-testid="stSegmentedControl"] button {{
  font-size: 12px; border-radius: 999px; }}

footer, #MainMenu, header[data-testid="stHeader"] {{ visibility: hidden; }}
</style>
"""


def apply_theme() -> None:
    """
    Inject the design system. Called once per page, before anything renders.

    Blank lines are stripped first, and this is not cosmetic: st.markdown parses
    its input as Markdown before honouring the HTML, and in Markdown a blank
    line ENDS a raw-HTML block. Leave them in and everything after the first one
    is treated as prose — the stylesheet renders as visible text on the page and
    only the rules above that first gap take effect. Fonts come in through an
    @import rather than <link> tags for the same reason: the sanitizer keeps
    <style>, but drops <link>.
    """
    css = "\n".join(line for line in _css().splitlines() if line.strip())
    st.markdown(css, unsafe_allow_html=True)


def sidebar_brand(subtitle: str = "KSU Student Managed Investment Fund") -> None:
    st.sidebar.markdown(
        f'<div class="cf-brand"><div class="cf-brand-mark">S</div>'
        f'<div><div class="cf-brand-name">SMIF</div>'
        f'<div class="cf-brand-sub">{subtitle}</div></div></div>',
        unsafe_allow_html=True)


# ── small render helpers ──────────────────────────────────────────────────────

def topbar(title: str, subtitle: str = "", chips: list[tuple[str, str, float | None]]
           | None = None) -> None:
    """
    Page title on the left, index chips on the right.

    `chips` are (label, value, signed change) — the change decides the colour,
    so a chip never has to be told whether it is good news.
    """
    html = ['<div class="cf-topbar"><div>',
            f'<div class="cf-page-title">{title}</div>']
    if subtitle:
        html.append(f'<div class="cf-page-sub">{subtitle}</div>')
    html.append("</div>")
    if chips:
        html.append('<div class="cf-idx-row">')
        for label, value, change in chips:
            cls = sign_class(change)
            html.append(f'<div class="cf-idx"><div class="k">{label}</div>'
                        f'<div class="v {cls}">{value}</div></div>')
        html.append("</div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def sign_class(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "cf-flat"
    if x > 0:
        return "cf-up"
    if x < 0:
        return "cf-down"
    return "cf-flat"


def eyebrow(title: str, note: str = "") -> str:
    tail = f'<span class="note">{note}</span>' if note else ""
    return f'<div class="cf-eyebrow">{title}{tail}</div>'


@contextmanager
def panel(title: str = "", note: str = ""):
    """
    A Confluence card. Use as `with theme.panel("Growth"): ...`.

    Streamlit's bordered container does the nesting; the CSS above gives it the
    flat hairline look, and the mono eyebrow goes in as its first child.
    """
    with st.container(border=True):
        if title:
            st.markdown(eyebrow(title, note), unsafe_allow_html=True)
        yield


def tiles(items: list[dict]) -> None:
    """
    A row of headline figures.

    Each item: {"label", "value", "sub" (optional), "sign" (optional float)}.
    `sign` colours the value; leave it out for a figure that is neither good nor
    bad, like a cash weight.
    """
    html = ['<div class="cf-tiles">']
    for it in items:
        cls = sign_class(it["sign"]) if "sign" in it else "cf-flat"
        sub = f'<div class="cf-tile-sub">{it["sub"]}</div>' if it.get("sub") else ""
        html.append(f'<div class="cf-tile">'
                    f'<div class="cf-tile-label">{it["label"]}</div>'
                    f'<div class="cf-tile-value {cls}">{it["value"]}</div>'
                    f'{sub}</div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def note(text: str, kind: str = "caution") -> None:
    cls = "cf-note ok" if kind == "ok" else "cf-note"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def hover_strip(text: str) -> None:
    st.markdown(f'<div class="cf-hover">{text}</div>', unsafe_allow_html=True)


# ── formatting ────────────────────────────────────────────────────────────────

def pct(x, places: int = 2, signed: bool = True) -> str:
    if x is None or pd.isna(x):
        return "—"
    sign = "+" if (signed and x > 0) else ""
    return f"{sign}{x * 100:.{places}f}%"


def money(x, places: int = 0) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"${x:,.{places}f}"


def num(x, places: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{places}f}"


# ── Plotly ────────────────────────────────────────────────────────────────────

def plotly_layout(height: int = 380, **overrides) -> dict:
    """The shared chart chrome: every figure in the app starts from this."""
    layout = dict(
        template="plotly_dark",
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        height=height,
        font=dict(size=11, color=TEXT_MUTED, family="IBM Plex Sans, sans-serif"),
        margin=dict(l=54, r=18, t=12, b=38),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER,
                        font=dict(family="DM Mono, monospace", size=11.5)),
        legend=dict(orientation="v", x=0.012, y=0.98, xanchor="left",
                    yanchor="top", bgcolor="rgba(11,11,11,0.65)",
                    font=dict(size=11.5), borderwidth=0),
        xaxis=dict(gridcolor=GRID, showspikes=True, spikecolor=SPIKE,
                   spikethickness=1, spikemode="across", spikedash="dot",
                   spikesnap="cursor", linecolor=BORDER),
        yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=ZERO,
                   zerolinewidth=1, linecolor=BORDER),
    )
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(layout.get(k), dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout


PLOTLY_CONFIG = {"displaylogo": False, "displayModeBar": False,
                 "responsive": True}


def correlation_scale() -> list:
    """
    A diverging scale centred on zero: cool for negative, grey at zero, coral
    for positive.

    Not green/red — in this app those two mean favourable and unfavourable, and
    a -0.4 correlation is not bad news. Grey at the centre also makes "near
    zero" read as absence rather than as a colour of its own.
    """
    return [
        [0.0, "#1e4e5f"],      # -1: deep teal
        [0.25, "#2f7287"],
        [0.5, "#3a3a3a"],      #  0: neutral graphite
        [0.75, "#b8483c"],
        [1.0, ACCENT],         # +1: the coral accent
    ]


def heat_color(value: float, cap: float = 0.25) -> str:
    """Finviz-style return heat: slate at zero, saturating to green or red."""
    if value is None or pd.isna(value):
        return "#4b5563"
    x = max(-1.0, min(1.0, value / cap))
    mid = (71, 77, 94)
    target = (34, 163, 74) if x >= 0 else (210, 50, 55)
    a = abs(x)
    rgb = tuple(round(m + (t - m) * a) for m, t in zip(mid, target))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
