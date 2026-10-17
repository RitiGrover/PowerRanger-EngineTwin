"""Shared visual language for EngineTwin: palette, type, logo and the global Streamlit skin.

Presentation only. Nothing here touches a model, a signal calculation or a data table."""

# One restrained palette: near-black neutrals, a pale steel-blue accent, a soft sage for "healthy"
# and a dusty terracotta for faults. No yellow and nothing saturated enough to glow.
PALETTE = {
    "bg": "#0a0b0d",
    "panel": "#111316",
    "panel_2": "#15181b",
    "line": "#23262b",
    "line_strong": "#2f3338",
    "text": "#e6e8ea",
    "muted": "#8b9197",
    "dim": "#5d636a",
    "accent": "#a9c4d4",
    "ok": "#b9cf9a",
    "warn": "#d49a6a",
    "fault": "#d9795f",
}

SEQUENTIAL = [[0.0, "#101316"], [0.25, "#1f3440"], [0.5, "#3f6a82"], [0.75, "#86aec2"], [1.0, "#e4eef3"]]
FAULT_SCALE = [[0.0, "#2a1612"], [0.5, "#b85a43"], [1.0, "#f0c4b6"]]


def register_plotly_theme() -> None:
    """One dark chart template for every figure: transparent panels, quiet grid, the palette above."""
    import plotly.graph_objects as go
    import plotly.io as pio

    axis = dict(gridcolor="#1c1f23", zerolinecolor="#262a2f", linecolor="#2a2e33", tickcolor="#2a2e33",
                tickfont=dict(color="#8b9197", size=11), title=dict(font=dict(color="#8b9197", size=12)))
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="#111316", plot_bgcolor="#111316",
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif", color="#b4bac0", size=12),
        colorway=["#a9c4d4", "#b9cf9a", "#d9795f", "#8fa3b3", "#c7a98d", "#7d858c"],
        title=dict(font=dict(size=14, color="#e6e8ea"), x=0.02, xanchor="left"),
        xaxis=axis, yaxis=axis, legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#b4bac0")),
        hoverlabel=dict(bgcolor="#15181b", bordercolor="#2f3338", font=dict(color="#e6e8ea")),
        margin=dict(l=48, r=24, t=56, b=44),
    )
    pio.templates["enginetwin"] = template
    pio.templates.default = "enginetwin"


LOGO_MARK_SVG = (
    '<svg viewBox="0 0 32 32" width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="1" y="1" width="30" height="30" rx="9" fill="#111316" stroke="#a9c4d4" stroke-opacity=".55"/>'
    '<g fill="#a9c4d4" fill-opacity=".9">'
    '<rect x="7" y="8.5" width="2.6" height="9" rx="1.3"/>'
    '<rect x="12.4" y="12.5" width="2.6" height="11" rx="1.3"/>'
    '<rect x="17.8" y="12.5" width="2.6" height="11" rx="1.3"/>'
    '<rect x="23.2" y="8.5" width="2.6" height="9" rx="1.3"/>'
    "</g>"
    '<path d="M4.5 18.5 C8 9.5 11 9.5 14 18 S20.5 26.5 27.5 14.5" fill="none" stroke="#b9cf9a" stroke-width="1.7" stroke-linecap="round"/>'
    "</svg>"
)


def logo_mark(size: int = 32) -> str:
    return LOGO_MARK_SVG.format(size=size)


def brand_block() -> str:
    """Sidebar / top-left brand: mark, wordmark, team line."""
    return (
        '<div class="et-brand">'
        '<span class="et-brand__text"><b>Power Ranger</b></span>'
        "</div>"
    )


FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wdth,wght@12..96,75..100,400..700&display=swap');"

THEME_CSS = (
    "<style>"
    + FONT_IMPORT
    + """
:root {
  --et-bg:#0a0b0d; --et-panel:#111316; --et-panel-2:#15181b; --et-line:#23262b; --et-line-2:#2f3338;
  --et-text:#e6e8ea; --et-muted:#8b9197; --et-dim:#5d636a;
  --et-accent:#a9c4d4; --et-ok:#b9cf9a; --et-warn:#d49a6a; --et-fault:#d9795f;
  --et-cyan:#a9c4d4; --et-green:#b9cf9a; --et-amber:#d49a6a; --et-red:#d9795f;
  --et-sans:'Bricolage Grotesque','Segoe UI',Arial,sans-serif;
  --et-mono:'Bricolage Grotesque','Segoe UI',Arial,sans-serif;
}

/* ---- canvas: black with a faint graph-paper grid ---------------------------------------- */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#0a0b0d !important; color:var(--et-text); }
.stApp, .stApp input, .stApp button, .stApp select, .stApp textarea, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp [class^="et-"], .stApp [class^="et-"] * { font-family:var(--et-sans) !important; }
.stApp::before { content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:
    linear-gradient(rgba(169,196,212,.052) 1px, transparent 1px),
    linear-gradient(90deg, rgba(169,196,212,.052) 1px, transparent 1px),
    linear-gradient(rgba(169,196,212,.022) 1px, transparent 1px),
    linear-gradient(90deg, rgba(169,196,212,.022) 1px, transparent 1px);
  background-size:120px 120px,120px 120px,24px 24px,24px 24px;
  -webkit-mask-image:radial-gradient(ellipse 90% 80% at 62% 30%, #000 25%, transparent 92%);
          mask-image:radial-gradient(ellipse 90% 80% at 62% 30%, #000 25%, transparent 92%); }
[data-testid="stAppViewContainer"] > .main, [data-testid="stMain"], .block-container { position:relative; z-index:1; background:transparent !important; }
.block-container { max-width:1560px; padding:1.1rem 2rem 2rem; }
[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap:.7rem; }
h2, h3 { margin:0 0 .2rem !important; padding:0 !important; }
.et-kicker { margin:.3rem 0 .45rem; }

/* ---- chrome: no Deploy button, quiet header --------------------------------------------- */
[data-testid="stHeader"] { background:transparent !important; backdrop-filter:none; }
.stAppDeployButton, [data-testid="stAppDeployButton"], [data-testid="stToolbar"] .stDeployButton, .stDeployButton { display:none !important; }
[data-testid="stToolbar"] { right:.75rem; }
[data-testid="stStatusWidget"] { opacity:.6; }

/* ---- type ------------------------------------------------------------------------------- */
h1,h2,h3,h4 { font-weight:600 !important; letter-spacing:-.02em !important; color:var(--et-text) !important; }
h2, h3 { font-size:1.35rem !important; }
p, li, label, .stMarkdown { line-height:1.55; }
[data-testid="stCaptionContainer"], .stCaption, small { color:var(--et-muted) !important; }
.et-kicker, .et-section, .panel-label, .k-label, .et-label, .et-mini span, .et-sensor small {
  font-family:var(--et-mono) !important; font-size:.68rem !important; letter-spacing:.08em !important; text-transform:uppercase !important; color:var(--et-muted) !important; }
.et-section:before { content:"[ "; color:var(--et-dim); } .et-section:after { content:" ]"; color:var(--et-dim); }
.et-value, .k-value, .sensor-value, .et-sensor strong, [data-testid="stMetricValue"] { font-family:var(--et-mono) !important; font-weight:500 !important; letter-spacing:-.02em; }
[data-testid="stMetricValue"] { font-size:1.7rem !important; }
[data-testid="stMetricLabel"] p { font-family:var(--et-mono) !important; font-size:.68rem !important; letter-spacing:.06em; text-transform:uppercase; color:var(--et-muted) !important; }

/* ---- sidebar ---------------------------------------------------------------------------- */
[data-testid="stSidebar"] { background:#0d0f11 !important; border-right:1px solid var(--et-line) !important; }
[data-testid="stSidebar"] > div:first-child { padding-top:0 !important; }
[data-testid="stSidebarContent"] { padding-top:.35rem !important; }
.et-brand { display:flex; align-items:center; gap:.7rem; padding:.4rem .25rem 1rem; border-bottom:1px solid var(--et-line); margin-bottom:.65rem; }
.et-brand__mark { display:grid; place-items:center; line-height:0; }
.et-brand__text { display:flex; flex-direction:column; line-height:1.05; }
.et-brand__text b { font-size:1.34rem; font-weight:650; letter-spacing:-.035em; color:var(--et-text); }
.et-brand__text small { margin-top:.28rem; font-family:var(--et-mono) !important; font-size:.6rem; letter-spacing:.16em; text-transform:uppercase; color:var(--et-dim); }
.et-nav-group { margin:.8rem .35rem .35rem; color:#566271; font-size:.58rem; font-weight:600; letter-spacing:.12em; text-transform:uppercase; }
[data-testid="stSidebar"] div[class*="st-key-nav_"] { margin:-.08rem 0; }
[data-testid="stSidebar"] div[class*="st-key-nav_"] button { position:relative; justify-content:flex-start; min-height:2.25rem; padding:.42rem .7rem !important; border-radius:6px !important; border:1px solid transparent !important; background:transparent !important; color:var(--et-muted) !important; box-shadow:none !important; }
[data-testid="stSidebar"] div[class*="st-key-nav_"] button p { font-size:.88rem; font-weight:450; letter-spacing:.005em; white-space:pre; }
[data-testid="stSidebar"] div[class*="st-key-nav_"] button:hover { background:#14171a !important; color:var(--et-text) !important; }
[data-testid="stSidebar"] div[class*="st-key-nav_"] button[kind="primary"] { background:#1a1d25 !important; border-color:#252b38 !important; color:#f0f2f4 !important; }
[data-testid="stSidebar"] div[class*="st-key-nav_"] button[kind="primary"]:before { content:""; position:absolute; left:-1px; top:24%; bottom:24%; width:3px; border-radius:0 3px 3px 0; background:#9cbcd0; }
[data-testid="stSidebar"] div[class*="st-key-nav_"] button[kind="primary"] p { font-weight:620; }
[data-testid="stSidebar"] hr { border-color:var(--et-line); }
[data-testid="stSidebar"] [data-testid="stExpander"] { border:1px solid var(--et-line); border-radius:12px; background:var(--et-panel); }

/* ---- surfaces ---------------------------------------------------------------------------- */
.et-card, .kpi, .panel, .sensor, .et-sensor, .et-engine-card, .et-log, .et-health, .et-toolbar, .et-mini, .event, .bus,
[data-testid="stPlotlyChart"], [data-testid="stExpander"], [data-testid="stDataFrame"], [data-testid="stMetric"] {
  background:var(--et-panel) !important; border:1px solid var(--et-line) !important; border-radius:14px !important; box-shadow:none !important; }
[data-testid="stMetric"] { padding:.9rem 1rem; }
.et-card:after, .kpi:after, .sensor:before, .et-sensor:before { box-shadow:none !important; }
.et-card:after, .kpi:after { height:2px !important; opacity:.7; }
.et-sensor { border-left:1px solid var(--et-line) !important; position:relative; }
.et-sensor:before { content:""; position:absolute; left:.8rem; top:.85rem; width:6px; height:6px; border-radius:50%; background:var(--sensor,var(--et-accent)); }
.et-sensor small { padding-left:1rem; }
.et-sensor em { font-family:var(--et-mono) !important; font-size:.62rem !important; letter-spacing:.06em; text-transform:uppercase !important; }
.et-hero { border-bottom:1px solid var(--et-line) !important; }
.et-hero h1 { font-weight:600 !important; letter-spacing:-.03em !important; }
.et-demo { background:#14171a !important; border:1px solid var(--et-line-2) !important; color:var(--et-muted) !important; border-radius:12px !important; }
.et-demo b { color:var(--et-warn) !important; font-family:var(--et-mono) !important; font-size:.68rem; }
.et-live { border-radius:999px !important; border:1px solid var(--et-line-2) !important; color:var(--et-muted) !important; font-family:var(--et-mono) !important; }
.et-dot, .live-dot { background:var(--et-ok) !important; box-shadow:none !important; animation:none !important; }
.et-ring, .health-ring { filter:none !important; }
.et-ring { background:conic-gradient(var(--et-accent) calc(var(--health)*1%), #1d2125 0) !important; }
.et-ring:after { background:var(--et-panel) !important; }
[data-testid="stAlert"] { border-radius:12px !important; background:#14171a !important; border:1px solid var(--et-line-2) !important; color:var(--et-text); }
[data-testid="stAlert"] [data-testid="stAlertContentError"], [data-testid="stAlert"] svg { color:inherit; }

[data-testid="stAlertContainer"] { background:#111316 !important; border:1px solid var(--et-line) !important; border-left:3px solid var(--et-accent) !important; border-radius:12px !important; color:var(--et-text) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { border-left-color:var(--et-ok) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) { border-left-color:var(--et-fault) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) { border-left-color:var(--et-warn) !important; }
[data-testid="stAlertContainer"] p, [data-testid="stAlertContainer"] li { color:var(--et-text) !important; }
[data-testid="stAlertContainer"] svg { color:var(--et-muted) !important; fill:var(--et-muted) !important; }

/* ---- controls ---------------------------------------------------------------------------- */
.stApp [data-testid="stButton"] button, .stApp [data-testid="stDownloadButton"] button {
  border-radius:999px !important; border:1px solid var(--et-line-2) !important; background:#14171a !important; color:var(--et-text) !important;
  font-weight:500; letter-spacing:0; transition:border-color .15s, background .15s; }
.stApp [data-testid="stButton"] button:hover { border-color:var(--et-accent) !important; color:var(--et-accent) !important; background:#171b1f !important; }
.stApp [data-testid="stButton"] button[kind="primary"] { background:var(--et-accent) !important; color:#0c1114 !important; border-color:var(--et-accent) !important; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="base-input"] { background:#111316 !important; border-color:var(--et-line-2) !important; border-radius:10px !important; }
[data-baseweb="popover"] [role="listbox"] { background:#14171a !important; border:1px solid var(--et-line-2); }
[data-testid="stSlider"] [role="slider"] { background:var(--et-accent) !important; box-shadow:none !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div:first-child { background:var(--et-accent) !important; }
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { font-family:var(--et-mono) !important; color:var(--et-muted) !important; }
[data-baseweb="tab-list"] { border-bottom:1px solid var(--et-line) !important; gap:.4rem; }
[data-baseweb="tab"] { color:var(--et-muted) !important; background:transparent !important; }
[aria-selected="true"][data-baseweb="tab"] { color:var(--et-text) !important; background:transparent !important; }
[data-baseweb="tab-highlight"] { background-color:var(--et-accent) !important; }
[data-testid="stToggle"] label p { color:var(--et-text) !important; }
[data-testid="stProgress"] > div > div > div > div { background:var(--et-accent) !important; }
[data-testid="stProgressBar"], .stProgress > div > div { background:#1b1e22 !important; }
[data-testid="stFileUploaderDropzone"] { background:var(--et-panel) !important; border:1px dashed var(--et-line-2) !important; border-radius:14px !important; }
::selection { background:rgba(169,196,212,.28); }
::-webkit-scrollbar { width:10px; height:10px; } ::-webkit-scrollbar-thumb { background:#24282c; border-radius:8px; } ::-webkit-scrollbar-track { background:transparent; }
.stApp a { color:var(--et-accent); }

/* ---- the replay section: scoped styles from the integration are re-skinned ----------------- */
.stApp .st-key-updated_live { --line:#23262b; --ice:#e6e8ea; --muted:#8b9197; --cyan:#a9c4d4; --mint:#b9cf9a; --amber:#d49a6a; --red:#d9795f; }
.stApp .st-key-updated_live .kpi, .stApp .st-key-updated_live .panel, .stApp .st-key-updated_live .sensor,
.stApp .st-key-updated_live .event, .stApp .st-key-updated_live .bus, .stApp .st-key-updated_live .control-rail {
  background:var(--et-panel) !important; border:1px solid var(--et-line) !important; border-radius:14px !important; }
.stApp .st-key-updated_live .kpi:after { box-shadow:none !important; height:2px; opacity:.75; }
.stApp .st-key-updated_live .sensor:before { box-shadow:none !important; width:3px; }
.stApp .st-key-updated_live .k-label, .stApp .st-key-updated_live .sensor-name, .stApp .st-key-updated_live .panel-label, .stApp .st-key-updated_live .clock span {
  font-family:var(--et-mono) !important; font-size:.68rem !important; letter-spacing:.07em !important; text-transform:uppercase !important; color:var(--et-muted) !important; }
.stApp .st-key-updated_live .k-value, .stApp .st-key-updated_live .sensor-value, .stApp .st-key-updated_live .clock strong { font-family:var(--et-mono) !important; font-weight:500 !important; }
.stApp .st-key-updated_live .guard-alert { background:#241613 !important; border:1px solid #6a3b30 !important; border-radius:14px !important; box-shadow:none !important; animation:none !important; }
.stApp .st-key-updated_live .guard-alert strong { color:#f0d9d2 !important; font-weight:600; letter-spacing:0; }
.stApp .st-key-updated_live .guard-alert span { color:#c9a69c !important; }
.stApp .st-key-updated_live .guard-alert .guard-mark { color:#f0d9d2 !important; border-color:#6a3b30 !important; border-radius:8px; }
.stApp .st-key-updated_live .health-ring { background:conic-gradient(var(--ring) calc(var(--score)*1%), #1d2125 0) !important; }
.stApp .st-key-updated_live .health-ring:before { background:var(--et-panel) !important; border-color:var(--et-line) !important; }
.stApp .st-key-updated_live [data-testid="stButton"] button { border-radius:999px !important; background:#14171a !important; border:1px solid var(--et-line-2) !important; }
.stApp .st-key-updated_live .engine-core:before { border-color:rgba(169,196,212,.10) !important; box-shadow:none !important; }
.stApp .st-key-updated_live .live-pill { border-radius:999px; background:#14171a !important; border-color:var(--et-line-2) !important; color:var(--et-muted) !important; }
.stApp .st-key-updated_live .title { letter-spacing:-.03em; font-weight:600; }

/* ---- synthetic scan strip (the old animated powertrain) ------------------------------------ */
.et-drive { border-radius:16px !important; border:1px solid var(--et-line) !important; background:radial-gradient(circle at 58% 36%, #171b1f 0, #0e1013 36%, #0a0b0d 74%) !important; }
.et-drive:after { background:#2c3238 !important; box-shadow:none !important; }
.et-marquee { color:rgba(169,196,212,.05) !important; font-weight:900 !important; }
.et-drive-meta, .et-drive-status, .et-drive-bottom strong, .et-scroll-cue { color:var(--et-accent) !important; font-family:var(--et-mono) !important; }
.et-drive-status { border-color:var(--et-line-2) !important; border-radius:999px; }

/* Sidebar navigation sits after the shared button rules so it keeps its own quieter treatment. */
.stApp [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding:.35rem .9rem 1rem !important; }
.stApp [data-testid="stSidebar"] .et-brand { padding:.25rem .15rem .8rem; margin-bottom:.35rem; }
.stApp [data-testid="stSidebar"] .et-nav-group { margin:.52rem .25rem .2rem; color:#566271; font-size:.58rem; font-weight:600; letter-spacing:.12em; }
.stApp [data-testid="stSidebar"] div[class*="st-key-nav_"] { margin:-.18rem 0 !important; }
.stApp [data-testid="stSidebar"] div[class*="st-key-nav_"] button,
.stApp [data-testid="stSidebar"] div[class*="st-key-nav_"] button[kind="secondary"] {
  justify-content:flex-start !important; min-height:1.95rem !important; padding:.28rem .62rem !important;
  border-radius:6px !important; border:1px solid transparent !important;
  background:transparent !important; color:var(--et-muted) !important; box-shadow:none !important;
}
.stApp [data-testid="stSidebar"] div[class*="st-key-nav_"] button > div,
.stApp [data-testid="stSidebar"] div[class*="st-key-nav_"] button > div > span,
.stApp [data-testid="stSidebar"] div[class*="st-key-nav_"] button [data-testid="stMarkdownContainer"] { width:100%; }
.stApp [data-testid="stSidebar"] div[class*="st-key-nav_"] button p { width:100%; margin:0; text-align:left; font-size:.82rem; line-height:1.15; font-variant-numeric:tabular-nums; }
.stApp [data-testid="stSidebar"] div[class*="st-key-nav_"] button:hover { background:#14171a !important; color:var(--et-text) !important; border-color:transparent !important; }
.stApp [data-testid="stSidebar"] div[class*="st-key-nav_"] button[kind="primary"] {
  background:#1a1d25 !important; border-color:#252b38 !important; color:#f0f2f4 !important;
}
.stApp [data-testid="stSidebar"] div[class*="st-key-nav_"] button[kind="primary"]:before {
  content:""; position:absolute; left:-1px; top:22%; bottom:22%; width:3px; border-radius:0 3px 3px 0; background:#9cbcd0;
}

@media (max-width:900px) { .block-container { padding:1.2rem 1rem 3rem; } }
@media (prefers-reduced-motion:reduce) { * { animation:none !important; transition:none !important; } }
</style>"""
)
