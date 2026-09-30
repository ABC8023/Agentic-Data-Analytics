"""
The look of the app: the custom CSS layer, the Plotly template, and the
small layout helpers. safe() escapes anything interpolated into raw HTML.

Design read: a working tool for analysts, and for anyone checking how an
answer was reached. Calm and precise rather than decorative. Warm paper,
ink text, one deep-teal accent, Geist for text and Geist Mono for figures.
Native widgets are styled by the Streamlit theme in .streamlit/config.toml;
this file only styles what the app renders as its own HTML, plus a few
refinements the theme has no setting for.
"""

import contextlib
import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ──────────────────────────────────────────────────────────────
#  TOKENS
#  Kept in step with .streamlit/config.toml.
# ──────────────────────────────────────────────────────────────

PAPER = "#F6F5F1"
SURFACE = "#FFFFFF"
INK = "#1B1D1C"
MUTED = "#5F625E"
LINE = "#DAD6CC"
LINE_SOFT = "#E7E3DA"
ACCENT = "#0F6E63"
BRICK = "#A64B35"
OCHRE = "#8A6414"
SLATE = "#4C6A8C"

THEME_CSS = f"""
<style>
:root {{
    --paper: {PAPER};
    --surface: {SURFACE};
    --ink: {INK};
    --muted: {MUTED};
    --line: {LINE};
    --line-soft: {LINE_SOFT};
    --accent: {ACCENT};
    --accent-soft: rgba(15, 110, 99, 0.08);
    --act: {BRICK};
    --watch: {OCHRE};
    --r-panel: 12px;
    --r-control: 8px;
    --r-inner: 4px;
    --mono: 'Geist Mono', ui-monospace, monospace;
}}

/* ---------- page frame ----------
   Streamlit's own header strip is hidden: the app's bar takes the top of
   the page, and the strip would sit over its right-hand button. */
[data-testid="stHeader"], [data-testid="stDecoration"] {{ display: none !important; }}
.block-container {{ max-width: 1320px; padding: 0 2.4rem 5rem 2.4rem; }}
@media (max-width: 720px) {{ .block-container {{ padding: 0 1rem 4rem 1rem; }} }}

/* Heading anchor icons are noise in a tool. */
[data-testid="stHeaderActionElements"] {{ display: none !important; }}

/* Figures line up in columns instead of jittering between reruns. */
[data-testid="stMetricValue"], [data-testid="stMetricDelta"],
[data-testid="stDataFrame"], [data-testid="stTable"] td, .num {{
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum" 1;
}}

/* ---------- site header ----------
   The site-header pattern: a white bar across the full width of the
   window, a hairline under it, pinned to the top while the page scrolls.
   The content inside stays on the page column.

   Full width without widening the page: the white and the hairline are a
   border-image pushed out 100vmax on each side. Border-image outset is ink
   overflow, so it paints edge to edge but adds no horizontal scroll. */
/* Streamlit wraps each container in a box exactly its own height, and a
   sticky element can only move inside its parent. So the wrapper is what
   sticks, and the bar inside it carries the look. */
[data-testid="stLayoutWrapper"]:has(> .st-key-topbar),
.st-key-topbar:not([data-testid="stLayoutWrapper"] > .st-key-topbar) {{
    position: sticky; top: 0; z-index: 100;
}}
[data-testid="stMainBlockContainer"], [data-testid="stMain"] {{ padding-top: 0 !important; }}
/* The element carrying this stylesheet has no height, but Streamlit still
   spaces it 15px from the bar, leaving a gap above the header. A style
   sheet applies whether or not its element is displayed. */
.stElementContainer:has(style) {{ display: none !important; }}
.st-key-topbar p {{ margin: 0 !important; }}
.st-key-brand_lockup {{ flex-wrap: nowrap !important; align-items: center; }}
/* The name and its line never wrap or shrink; the menus and the table
   name give way first, and on narrow screens they hide instead. */
.st-key-topbar .brand, .st-key-topbar .brand-note,
.st-key-brand_home button p {{ white-space: nowrap !important; }}
.st-key-brand_lockup > *, .st-key-topbar_left > .stElementContainer,
.st-key-topbar_left > [data-testid="stLayoutWrapper"]:has(.st-key-brand_lockup) {{
    flex: 0 0 auto !important; width: auto !important; min-width: 0;
}}
.st-key-topbar_actions .stElementContainer {{ align-self: center; }}
.st-key-topbar {{
    padding: 22px 0;
    margin-bottom: 22px;
    border-image: linear-gradient(to bottom,
        var(--surface) calc(100% - 1px), var(--line) calc(100% - 1px)) fill 0 / / 0 100vmax;
}}
/* Streamlit pulls every markdown block up by 15px to hide a trailing
   paragraph margin. There is no paragraph here, so the pull only knocks
   the bar's contents off centre. */
.st-key-topbar [data-testid="stMarkdownContainer"] {{ margin-bottom: 0 !important; }}
.st-key-topbar_row {{ flex-wrap: nowrap !important; align-items: center; min-height: 44px; }}
.st-key-topbar_left {{ flex-wrap: nowrap !important; align-items: center; gap: 28px !important; min-width: 0; }}
.st-key-topbar_actions {{ flex-wrap: nowrap !important; align-items: center; justify-content: flex-end; gap: 16px !important; }}

/* The one action in the bar, once a table is open. */
.st-key-change_dataset button {{
    min-height: 40px; padding: 6px 16px; font-weight: 600;
}}

/* Narrow windows keep the name and the action; the line under the name
   and the status make room. */
@media (max-width: 720px) {{ .st-key-topbar .brand-note, .st-key-topbar .brand-sep {{ display: none; }} }}
@media (max-width: 560px) {{ .st-key-topbar .status {{ display: none !important; }} }}
@media (max-width: 560px) {{ .st-key-topbar .dataset-chip {{ display: none; }} }}

.brand {{ display: flex; align-items: center; gap: 12px; min-height: 36px; }}
.brand-sep {{ width: 1px; height: 16px; background: var(--line); flex: 0 0 auto; }}
.brand-line {{ gap: 10px; padding-left: 2px; }}

/* The name as a way home: looks like the lockup, behaves like a button. */
.st-key-brand_home button {{
    padding: 4px 8px 4px 4px; min-height: 34px; gap: 10px;
    border-radius: var(--r-control); color: var(--ink);
}}
.st-key-brand_home button::before {{
    content: ""; width: 22px; height: 22px; border-radius: 6px; flex: 0 0 auto;
    background:
        linear-gradient(var(--accent), var(--accent)) left 5px bottom 5px / 3px 6px no-repeat,
        linear-gradient(var(--accent), var(--accent)) left 10px bottom 5px / 3px 11px no-repeat,
        linear-gradient(var(--accent), var(--accent)) left 15px bottom 5px / 3px 8px no-repeat,
        var(--accent-soft);
}}
.st-key-brand_home button p {{
    font-weight: 600 !important; font-size: 1.02rem !important;
    letter-spacing: -0.01em; color: var(--ink);
}}
.st-key-brand_home button:hover {{ background: var(--accent-soft); }}
.st-key-brand_home button:hover p {{ color: var(--accent); }}

.status {{
    display: inline-flex; align-items: center; gap: 7px; white-space: nowrap;
    font-size: .88rem; color: var(--muted); cursor: help;
}}
.status::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%; background: #A9A59B; }}
.status-on::before {{ background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }}
.brand-mark {{
    width: 22px; height: 22px; border-radius: 6px; flex: 0 0 auto;
    background:
        linear-gradient(var(--accent), var(--accent)) left 5px bottom 5px / 3px 6px no-repeat,
        linear-gradient(var(--accent), var(--accent)) left 10px bottom 5px / 3px 11px no-repeat,
        linear-gradient(var(--accent), var(--accent)) left 15px bottom 5px / 3px 8px no-repeat,
        var(--accent-soft);
}}
.brand-name {{ font-weight: 600; font-size: 1.02rem; letter-spacing: -0.01em; color: var(--ink); }}
.brand-note {{ color: var(--muted); font-size: .9rem; }}
@media (max-width: 720px) {{ .brand-note, .brand-sep {{ display: none; }} }}

.dataset-chip {{
    display: flex; justify-content: flex-end; align-items: baseline; gap: 10px;
    font-size: .9rem; color: var(--muted); white-space: nowrap;
    overflow: hidden; max-width: 340px;
}}
.dataset-chip .file {{ overflow: hidden; text-overflow: ellipsis; min-width: 0; }}
.dataset-chip .file {{ color: var(--ink); font-weight: 600; }}
.dataset-chip .num {{ font-family: var(--mono); font-size: .82rem; }}

/* ---------- landing ---------- */
.hero {{ padding: 4px 0 8px 0; }}
.hero-eyebrow {{
    color: var(--accent); font-weight: 600; font-size: .92rem;
    margin-bottom: 14px;
}}
.hero-title {{
    font-size: clamp(2.1rem, 3.8vw, 3.2rem); font-weight: 600;
    letter-spacing: -0.035em; line-height: 1.06; color: var(--ink);
    max-width: 17ch; margin: 0 0 18px 0; text-wrap: balance;
}}
.hero-copy {{
    font-size: 1.06rem; line-height: 1.6; color: var(--muted);
    max-width: 54ch; margin: 0 0 28px 0; text-wrap: pretty;
}}

/* How an answer is made: a numbered route, not three feature cards. */
.hero ol.route {{
    list-style: none; margin: 0 !important; padding: 0 !important;
    max-width: 60ch; counter-reset: step;
}}
.route li {{
    counter-increment: step; margin: 0 !important;
    display: grid; grid-template-columns: 2.2rem 1fr; gap: 2px 10px;
    padding: 12px 0; border-top: 1px solid var(--line);
}}
.route li:last-child {{ border-bottom: 1px solid var(--line); }}
.route li::before {{
    content: counter(step, decimal-leading-zero);
    font-family: var(--mono); font-size: .8rem; color: var(--accent);
    grid-row: span 2; padding-top: 2px;
}}
.route b {{ font-weight: 600; color: var(--ink); }}
.route span {{ color: var(--muted); font-size: .93rem; line-height: 1.5; }}

.panel-title {{ font-weight: 600; font-size: 1.02rem; margin: 2px 0 2px 0; color: var(--ink); }}
.panel-note {{ color: var(--muted); font-size: .9rem; margin: 0 0 12px 0; }}
.panel-note.sample-note {{ margin: 10px 0 4px 0; }}

.inside-title {{ font-weight: 600; font-size: 1.1rem; margin: 44px 0 6px 0; color: var(--ink); }}
.inside {{
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    column-gap: 36px;
}}
@media (max-width: 1000px) {{ .inside {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 640px)  {{ .inside {{ grid-template-columns: 1fr; }} }}
.inside > div {{ padding: 14px 0 16px 0; border-top: 1px solid var(--line); }}
.inside .item-title {{ font-size: .98rem; font-weight: 600; margin: 0 0 4px 0; color: var(--ink); }}
.inside p {{ font-size: .9rem; line-height: 1.5; color: var(--muted); margin: 0; max-width: 42ch; }}

/* Sample datasets: quiet rows, the title is the button. */
[class*="st-key-sample_card_"] {{
    border-top: 1px solid var(--line-soft); padding-top: 6px; gap: 0 !important;
}}


/* ---------- section headings inside tabs ---------- */
.sec.sec-first {{ margin-top: 4px; }}

/* The rule between sections comes from st.divider, so none here. */
.sec {{
    font-size: 1.18rem; font-weight: 600; letter-spacing: -0.01em;
    color: var(--ink); margin: 18px 0 6px 0; text-wrap: balance;
}}
.lede {{
    color: var(--muted); font-size: .95rem; margin: 0 0 12px 0;
    line-height: 1.6; max-width: 72ch; text-wrap: pretty;
}}

/* ---------- headline finding ----------
   The one sentence a reader should leave with. */
.headline {{
    font-size: clamp(1.18rem, 1.7vw, 1.46rem); font-weight: 600;
    letter-spacing: -0.015em; line-height: 1.4; color: var(--ink);
    background: var(--surface);
    border: 1px solid var(--line-soft); border-left: 3px solid var(--accent);
    border-radius: var(--r-panel);
    padding: 20px 24px; margin: 8px 0 14px 0;
    text-wrap: pretty;
}}
.headline .sub {{
    display: block; font-size: .9rem; font-weight: 400; letter-spacing: 0;
    color: var(--muted); margin-top: 10px;
}}

/* ---------- session usage ----------
   Label left, figure right, one per line, so nothing is cut off however
   narrow the column. */
.usage-list {{
    display: grid; grid-template-columns: 1fr auto; gap: 0; margin: 0 0 8px 0;
}}
.usage-list dt, .usage-list dd {{
    margin: 0; padding: 7px 0; border-bottom: 1px solid var(--line-soft);
    font-size: .9rem;
}}
.usage-list dt {{ color: var(--muted); }}
.usage-list dd {{
    font-family: var(--mono); font-variant-numeric: tabular-nums;
    color: var(--ink); text-align: right; white-space: nowrap; padding-left: 12px;
}}

/* ---------- badges ----------
   Severity is written in words; colour only reinforces it. */
.badge {{
    font-size: .74rem; font-weight: 500;
    color: var(--muted); background: var(--paper);
    border: 1px solid var(--line); border-radius: var(--r-inner);
    padding: 1px 7px; margin-left: 8px; white-space: nowrap;
    vertical-align: middle;
}}
.badge-act {{ color: var(--act); border-color: rgba(166, 75, 53, 0.45); background: rgba(166, 75, 53, 0.06); }}
.badge-watch {{ color: var(--watch); border-color: rgba(138, 100, 20, 0.45); background: rgba(138, 100, 20, 0.06); }}

/* ---------- native widgets: refinements the theme has no setting for ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 22px; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ padding: 10px 2px; font-size: .96rem; font-weight: 500; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

[data-testid="stMetric"] {{
    background: var(--surface);
    border: 1px solid var(--line-soft);
    border-radius: var(--r-panel);
    padding: 14px 16px 12px 16px;
}}
[data-testid="stMetricValue"] {{ font-family: var(--mono); letter-spacing: -0.02em; }}

/* Bordered containers are the app's panels: white on the paper page. */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
    background: var(--surface);
}}

[data-testid="stPlotlyChart"] {{
    background: var(--surface);
    border: 1px solid var(--line-soft);
    border-radius: var(--r-panel);
    padding: 6px;
    overflow: hidden;
}}
[data-testid="stChatMessage"] {{
    background: var(--surface);
    border: 1px solid var(--line-soft);
    border-radius: var(--r-panel);
}}
[data-testid="stFileUploaderDropzone"] {{ background: var(--surface); }}

.stButton > button, .stDownloadButton > button, [data-testid="stPopover"] button {{
    transition: background-color .15s ease, border-color .15s ease, transform .1s ease;
}}
.stButton > button:active, .stDownloadButton > button:active {{ transform: translateY(1px); }}

:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}

/* Streamlit makes the scrolling main region focusable, for keyboard
   scrolling, but draws no focus ring on it. */
[data-testid="stElementToolbar"] button:focus-visible,
[data-testid="stDataFrame"] canvas:focus-visible,
[data-testid="stDataFrame"] [tabindex]:focus-visible {{
    outline: 2px solid var(--accent) !important; outline-offset: 1px;
}}

section[tabindex]:focus-visible, [data-testid="stMain"]:focus-visible {{
    outline: 2px solid var(--accent) !important; outline-offset: -2px;
}}

/* Captions and hints: Streamlit dims them with opacity, which lands at
   4.35:1 on the paper background, under the WCAG AA 4.5:1 minimum. A
   solid muted ink gives 5.7:1. */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stWidgetLabel"] small, [data-testid="stTooltipHoverTarget"] {{
    color: var(--muted) !important; opacity: 1 !important;
}}
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ transition: none !important; animation: none !important; }}
}}
</style>
"""


# ──────────────────────────────────────────────────────────────
#  CHARTS
# ──────────────────────────────────────────────────────────────

# Categorical palette. Muted hues at a similar weight, so no series shouts,
# and each reads on white. The accent leads so a single series is teal.
CHART_COLORWAY = [
    ACCENT, "#C08A2E", BRICK, SLATE,
    "#85607F", "#7A8450", "#8C8A84", "#B59F7A"
]

# Ordered magnitude, paper to deep teal.
CHART_SEQUENTIAL = [
    [0.0, "#F1EFE9"],
    [0.5, "#6FA79D"],
    [1.0, "#0B3F39"]
]

# Correlation runs from -1 to +1, so it needs a neutral midpoint.
CHART_DIVERGING = [
    [0.0, BRICK],
    [0.5, "#F1EFE9"],
    [1.0, ACCENT]
]

# Colours the figure builders use by name, so a chart cannot drift from
# the palette.
CHART_ROLES = {
    "band_fill": "rgba(15, 110, 99, 0.08)",
    "band_line": "rgba(15, 110, 99, 0.35)",
    "flag": BRICK,
    "projection": SLATE,
    "projection_fill": "rgba(76, 106, 140, 0.14)",
    "reference": "#8C8A84",
    "increase": ACCENT,
    "decrease": BRICK,
    "observed": ACCENT,
    "total": "#A9A59B",
    "connector": "rgba(27, 29, 28, 0.25)",
}

AXIS = dict(
    gridcolor=LINE_SOFT,
    zerolinecolor=LINE,
    linecolor=LINE,
    tickfont=dict(family="Geist Mono, monospace", size=11, color=MUTED),
    title=dict(font=dict(color=MUTED, size=12)),
)

ANALYST_TEMPLATE = go.layout.Template(
    layout=dict(
        # Solid white, not transparent: a transparent figure shows
        # Streamlit's own surface colour through it.
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=CHART_COLORWAY,
        font=dict(family="Geist, sans-serif", color=MUTED, size=13),
        title=dict(
            font=dict(family="Geist, sans-serif", color=INK, size=16),
            x=0.01,
            xanchor="left"
        ),
        xaxis=AXIS,
        yaxis=AXIS,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        colorscale=dict(
            sequential=CHART_SEQUENTIAL,
            diverging=CHART_DIVERGING
        ),
        margin=dict(l=60, r=28, t=64, b=68),
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=LINE,
            font=dict(family="Geist, sans-serif", color=INK)
        )
    )
)

pio.templates["analyst"] = ANALYST_TEMPLATE
px.defaults.template = "analyst"


def styled(figure):
    """Apply the app theme to any Plotly figure before rendering."""
    # A figure built by an unusual code path may not accept a template.
    # Styling is cosmetic, so a failure here must not lose the chart.
    with contextlib.suppress(Exception):
        # Backgrounds are also set on the figure itself: Streamlit's chart
        # frontend fills template-level backgrounds with its own surface
        # colour, but leaves explicit layout values alone.
        figure.update_layout(
            template="analyst",
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
        )

    return figure


def table(data, column_config=None, **options):
    """
    Render a computed table with readable numbers.

    Numeric columns get thousands separators and at most three decimals,
    so 65009.78 reads as 65,009.78. Any column given its own configuration
    keeps it. Use this for computed results only: in a raw preview of the
    uploaded rows, a year or an ID would be misprinted as 2,024.
    """

    config = {}

    for name in getattr(data, "columns", []):
        with contextlib.suppress(Exception):
            if pd.api.types.is_numeric_dtype(data[name]) and not (
                pd.api.types.is_bool_dtype(data[name])
            ):
                # A share held as a fraction reads as a percentage.
                values = data[name].dropna()
                is_share = (
                    "share" in str(name).lower()
                    and not values.empty
                    and values.between(-1, 1).all()
                )
                config[str(name)] = st.column_config.NumberColumn(
                    format="percent" if is_share else "localized"
                )

    config.update(column_config or {})
    options.setdefault("width", "stretch")
    options.setdefault("hide_index", True)

    return st.dataframe(data, column_config=config or None, **options)


def chart(figure, **options):
    """
    Render a Plotly figure in the app's own template.

    Streamlit applies its own chart theme by default, which replaces the
    template's palette and backgrounds. Every chart goes through here so
    none can be drawn in the wrong colours.
    """

    return st.plotly_chart(styled(figure), theme=None, **options)


# ──────────────────────────────────────────────────────────────
#  LAYOUT HELPERS
# ──────────────────────────────────────────────────────────────

def safe(text):
    """
    Escape text for a block rendered with unsafe_allow_html.

    Column names, labels and anything derived from them come from the
    uploaded file, so they are untrusted. Every interpolation into raw HTML
    goes through here.
    """

    return html.escape(str(text), quote=True)


BRAND_NAME = "AI data analyst"
BRAND_LINE = "Answers you can check"


def brand(home_action=None):
    """
    The product name and its line, at the top of every page.

    Args:
        home_action:
            None on the start page, where the name is plain text marked as
            the current page. Elsewhere, a callback: the name becomes a
            button back to the start page, so it works from the keyboard
            and announces itself as a control.
    """

    note = f'<span class="brand-sep" aria-hidden="true"></span><span class="brand-note">{BRAND_LINE}</span>'

    if home_action is None:
        st.markdown(
            '<div class="brand" aria-current="page">'
            '<span class="brand-mark" aria-hidden="true"></span>'
            f'<span class="brand-name">{BRAND_NAME}</span>{note}</div>',
            unsafe_allow_html=True
        )

        return

    with st.container(
        horizontal=True, vertical_alignment="center", gap=None, key="brand_lockup"
    ):
        st.button(
            BRAND_NAME,
            key="brand_home",
            type="tertiary",
            on_click=home_action,
            help="Close this table and go back to the start page",
        )
        st.markdown(f'<div class="brand brand-line">{note}</div>', unsafe_allow_html=True)


def bar_status(model_ready):
    """
    The right side of the bar on the start page: what is switched on.

    Args:
        model_ready:
            Whether a model key is configured, which decides whether
            questions the rules cannot read have anywhere to go.
    """

    if model_ready:
        state, text, detail = "on", "AI questions on", (
            "An AI key is set. Questions in your own words are read by "
            "AI, and the numbers are still worked out here."
        )

    else:
        state, text, detail = "off", "AI questions off", (
            "No AI key is set. Simple questions about totals, periods, "
            "groups and shares still work. Questions in your own words "
            "need a key."
        )

    st.markdown(
        f'<span class="status status-{state}" title="{safe(detail)}">{text}</span>',
        unsafe_allow_html=True
    )


def dataset_chip(file_name, rows, columns):
    """The loaded dataset, stated once, next to the way to change it."""

    st.markdown(
        f'<div class="dataset-chip" title="{safe(file_name)}">'
        f'<span class="file">{safe(file_name)}</span>'
        f'<span class="num">{rows:,} rows · {columns:,} cols</span></div>',
        unsafe_allow_html=True
    )




ROUTE = [
    (
        "Read locally",
        ("Most questions are parsed by rules and computed in pandas. "
        "No model, no cost, and the working is shown.")
    ),
    (
        "Planned by a model",
        ("A question the rules cannot read is turned into a query plan. "
        "The plan is checked against your columns, then computed here.")
    ),
    (
        "Written by the analyst",
        ("Open questions go to an agent that calls tools. Its answer is "
        "labelled as written by AI, with every tool call listed.")
    ),
]


def hero():
    """The landing statement and how answers are made."""

    steps = "".join(
        f"<li><b>{safe(title)}</b><span>{safe(detail)}</span></li>"
        for title, detail in ROUTE
    )

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">For CSV and Excel files</div>
            <h1 class="hero-title">Ask your data a question. See how the answer was worked out.</h1>
            <p class="hero-copy">Profile, clean, chart and model a table, then ask
               about it in plain English. Figures are calculated from your rows,
               and every answer shows the plan, the arithmetic and the SQL.</p>
            <ol class="route">{steps}</ol>
        </div>
        """,
        unsafe_allow_html=True
    )


CAPABILITIES = [
    (
        "Dashboard",
        ("Headline figures with their arithmetic, what moved and why it "
        "might have, and a brief to download.")
    ),
    (
        "Dataset overview",
        ("Row and column counts, dtypes, and a quality report on missing "
        "values, duplicates and constant columns.")
    ),
    (
        "Time series",
        ("A robust trend, periods outside the expected band, and a "
        "projection tested against a naive baseline.")
    ),
    (
        "Data cleaning",
        ("Drop duplicates and columns, choose how gaps are filled, then "
        "apply or download the result.")
    ),
    (
        "Machine learning",
        ("Three models per task, compared on held-out data, with the "
        "features ranked.")
    ),
    (
        "AI analysis",
        ("Questions in plain English, answered by rules, a planner or an "
        "agent, and labelled by which.")
    ),
]


def capabilities():
    """What opens once a table is loaded, as a plain list."""

    items = "".join(
        f'<div><div class="item-title">{safe(title)}</div><p>{safe(detail)}</p></div>'
        for title, detail in CAPABILITIES
    )

    st.markdown(
        f'<div class="inside-title">What opens once a table is loaded</div>'
        f'<div class="inside">{items}</div>',
        unsafe_allow_html=True
    )


def section(text):
    st.markdown(
        f'<div class="sec">{safe(text)}</div>',
        unsafe_allow_html=True
    )


def lede(text):
    st.markdown(
        f'<p class="lede">{safe(text)}</p>',
        unsafe_allow_html=True
    )
