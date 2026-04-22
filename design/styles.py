# design/styles.py
"""
Unified design system for PM Onboarding KB.

Usage:
    import streamlit as st
    from design.styles import apply_design_system
    st.set_page_config(page_title="...", layout="wide", initial_sidebar_state="expanded")
    apply_design_system()
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* ==========================================================================
   1. Fonts + tokens
   ========================================================================== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  /* Dark-mode scale: 100 lightest (primary text), 900 darkest (canvas).
     CSS rules below use flipped references (e.g. a light-mode rule that
     read `color: var(--ink-900)` now reads `color: var(--ink-100)`), so the
     role-to-token mapping inverts while the semantic scale is preserved. */
  --ink-900: #0a0a0a;
  --ink-700: #2a2a2a;
  --ink-500: #6b7280;
  --ink-300: #a3a3a3;
  --ink-100: #e5e5e5;
  --canvas: #0a0a0a;
  --surface: #1a1a1a;
  --accent-600: #6366f1;
  --accent-50: #1e1b4b;
  --success-600: #10b981;
  --success-50: #064e3b;
  --warn-600: #f59e0b;
  --warn-50: #451a03;
  --danger-600: #ef4444;
  --danger-50: #450a0a;
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace;
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --radius-xl: 12px;
  --shadow-xs: 0 1px 2px rgba(0, 0, 0, 0.4);
  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.5), 0 1px 2px rgba(0, 0, 0, 0.3);
  --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.6), 0 1px 3px rgba(0, 0, 0, 0.3);
  --shadow-focus: 0 0 0 3px rgba(99, 102, 241, 0.35);
}

/* ==========================================================================
   2. Global resets + Streamlit chrome
   ========================================================================== */
html, body, [class*="css"] {
  font-family: var(--font-sans);
  color: var(--ink-100);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.stApp { background-color: var(--canvas); }

/* Kill Streamlit's default top padding so hero content sits closer to nav */
.block-container {
  padding-top: 2rem !important;
  padding-bottom: 4rem !important;
  max-width: 1200px;
}

/* Hide Streamlit's default header/footer branding */
header[data-testid="stHeader"] { background: transparent; height: 2.5rem; }
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ==========================================================================
   3. Typography
   ========================================================================== */
h1, h2, h3, h4 {
  font-family: var(--font-sans);
  color: var(--ink-100);
  letter-spacing: -0.01em;
  font-weight: 600;
}
h1 { font-size: 30px; line-height: 38px; margin: 0 0 8px 0; }
h2 { font-size: 22px; line-height: 30px; margin: 32px 0 12px 0; }
h3 { font-size: 18px; line-height: 28px; margin: 24px 0 8px 0; font-weight: 500; }
h4 { font-size: 15px; line-height: 24px; margin: 16px 0 4px 0; font-weight: 500; color: var(--ink-300); }
p, li { font-size: 15px; line-height: 24px; color: var(--ink-100); }
small, .caption { font-size: 13px; color: var(--ink-500); }
code, pre, .mono { font-family: var(--font-mono); font-size: 13px; }

/* ==========================================================================
   4. Buttons
   ========================================================================== */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  font-family: var(--font-sans);
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0;
  padding: 8px 14px;
  border-radius: var(--radius-md);
  border: 1px solid var(--ink-700);
  background: var(--surface);
  color: var(--ink-100);
  box-shadow: var(--shadow-xs);
  transition: all 120ms ease;
  min-height: 36px;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
  border-color: var(--ink-500);
  box-shadow: var(--shadow-sm);
  transform: none;
}
.stButton > button:focus, .stDownloadButton > button:focus, .stFormSubmitButton > button:focus {
  outline: none;
  box-shadow: var(--shadow-focus);
  border-color: var(--accent-600);
}

/* Primary: use kind="primary" in Streamlit */
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: var(--ink-100);
  color: var(--surface);
  border-color: var(--ink-100);
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
  background: #f5f5f5;
  border-color: #f5f5f5;
}

/* ==========================================================================
   5. Inputs, selects, textarea, file uploader
   ========================================================================== */
.stTextInput input, .stTextArea textarea, .stNumberInput input, .stDateInput input {
  font-family: var(--font-sans);
  font-size: 14px;
  color: var(--ink-100);
  background: var(--surface);
  border: 1px solid var(--ink-700);
  border-radius: var(--radius-md);
  padding: 8px 12px;
  box-shadow: var(--shadow-xs);
  transition: border-color 120ms, box-shadow 120ms;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
  outline: none;
  border-color: var(--accent-600);
  box-shadow: var(--shadow-focus);
}

/* Selectbox */
div[data-baseweb="select"] > div {
  background: var(--surface);
  border: 1px solid var(--ink-700);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-xs);
  min-height: 36px;
  font-size: 14px;
}
div[data-baseweb="select"] > div:hover { border-color: var(--ink-500); }

/* File uploader */
[data-testid="stFileUploader"] section {
  background: var(--surface);
  border: 1px dashed var(--ink-700);
  border-radius: var(--radius-lg);
  padding: 24px;
  transition: border-color 120ms, background 120ms;
}
[data-testid="stFileUploader"] section:hover {
  border-color: var(--accent-600);
  background: var(--accent-50);
}

/* Labels above inputs */
label p, .stTextInput label, .stTextArea label, .stSelectbox label {
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--ink-300) !important;
  margin-bottom: 4px !important;
}

/* ==========================================================================
   6. Sidebar
   ========================================================================== */
section[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--ink-700);
  padding-top: 8px;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ink-500);
  font-weight: 500;
  margin: 16px 0 8px 0;
}

/* ==========================================================================
   7. Tabs — pill style
   ========================================================================== */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: var(--ink-900);
  padding: 4px;
  border-radius: var(--radius-md);
  display: inline-flex;
  margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--font-sans);
  font-size: 13px;
  font-weight: 500;
  color: var(--ink-500);
  background: transparent;
  border: none;
  border-radius: var(--radius-sm);
  padding: 6px 14px;
  min-height: 32px;
  transition: all 120ms ease;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--ink-100); }
.stTabs [aria-selected="true"] {
  background: var(--surface) !important;
  color: var(--ink-100) !important;
  box-shadow: var(--shadow-xs);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ==========================================================================
   8. Expanders — used as source citation panels
   ========================================================================== */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
  font-family: var(--font-sans);
  font-size: 13px;
  font-weight: 500;
  color: var(--ink-300);
  background: var(--surface);
  border: 1px solid var(--ink-700);
  border-radius: var(--radius-md);
  padding: 10px 14px;
  transition: all 120ms;
}
.streamlit-expanderHeader:hover, [data-testid="stExpander"] summary:hover {
  border-color: var(--ink-500);
  background: var(--ink-900);
}
[data-testid="stExpander"] {
  border: none;
  box-shadow: none;
}

/* ==========================================================================
   9. Chat messages — reframe as evidence, not bubbles
   ========================================================================== */
[data-testid="stChatMessage"] {
  background: transparent;
  border: none;
  padding: 16px 0;
  border-bottom: 1px solid var(--ink-900);
}
[data-testid="stChatMessage"]:last-child { border-bottom: none; }
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  background: var(--ink-900);
  border-radius: var(--radius-md);
}

/* Chat input — the Perplexity-style hero */
[data-testid="stChatInput"] {
  background: var(--surface);
  border: 1px solid var(--ink-700);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  transition: border-color 120ms, box-shadow 120ms;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent-600);
  box-shadow: var(--shadow-focus);
}
[data-testid="stChatInput"] textarea {
  font-family: var(--font-sans);
  font-size: 15px;
  color: var(--ink-100);
}

/* ==========================================================================
   10. Inline citations [1] [2] — academic paper feel
   ========================================================================== */
.citation-marker {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 18px;
  height: 18px;
  padding: 0 5px;
  margin: 0 2px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  color: var(--accent-600);
  background: var(--accent-50);
  border: 1px solid var(--accent-50);
  border-radius: var(--radius-sm);
  cursor: pointer;
  vertical-align: baseline;
  position: relative;
  top: -1px;
  transition: all 100ms;
  text-decoration: none;
}
.citation-marker:hover {
  background: var(--accent-600);
  color: var(--surface);
  border-color: var(--accent-600);
}

/* ==========================================================================
   11. Badges — confidence, status, counts
   ========================================================================== */
.kb-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  font-family: var(--font-sans);
  font-size: 11px;
  font-weight: 500;
  line-height: 16px;
  border-radius: var(--radius-sm);
  border: 1px solid transparent;
}
.kb-badge--success { background: var(--success-50); color: var(--success-600); border-color: var(--success-600); }
.kb-badge--warn    { background: var(--warn-50);    color: var(--warn-600);    border-color: var(--warn-600); }
.kb-badge--danger  { background: var(--danger-50);  color: var(--danger-600);  border-color: var(--danger-600); }
.kb-badge--neutral { background: var(--ink-900);    color: var(--ink-300);     border-color: var(--ink-700); }
.kb-badge--accent  { background: var(--accent-50);  color: var(--accent-600);  border-color: var(--accent-600); }
.kb-badge .dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: currentColor;
}

/* Confidence score — mono, with progress bar vibe */
.kb-confidence {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-300);
}
.kb-confidence-track {
  width: 48px; height: 4px;
  background: var(--ink-900);
  border-radius: 2px;
  overflow: hidden;
}
.kb-confidence-fill {
  height: 100%;
  background: var(--success-600);
  border-radius: 2px;
}

/* ==========================================================================
   12. Cards + panels (utility classes for custom HTML via st.markdown)
   ========================================================================== */
.kb-card {
  background: var(--surface);
  border: 1px solid var(--ink-700);
  border-radius: var(--radius-lg);
  padding: 20px;
  box-shadow: var(--shadow-xs);
}
.kb-card--hover:hover { box-shadow: var(--shadow-sm); border-color: var(--ink-500); }

.kb-metric {
  background: var(--surface);
  border: 1px solid var(--ink-700);
  border-radius: var(--radius-lg);
  padding: 16px 20px;
}
.kb-metric__label {
  font-size: 11px;
  font-weight: 500;
  color: var(--ink-500);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 6px;
}
.kb-metric__value {
  font-family: var(--font-sans);
  font-size: 28px;
  font-weight: 600;
  color: var(--ink-100);
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
}
.kb-metric__delta {
  font-family: var(--font-mono);
  font-size: 11px;
  margin-top: 4px;
  color: var(--success-600);
}
.kb-metric__delta--down { color: var(--danger-600); }

/* Source citation card (for RAG evidence) */
.kb-source {
  background: var(--surface);
  border: 1px solid var(--ink-700);
  border-left: 3px solid var(--accent-600);
  border-radius: var(--radius-md);
  padding: 12px 14px;
  margin-bottom: 8px;
}
.kb-source__num {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  color: var(--accent-600);
  margin-right: 8px;
}
.kb-source__title {
  font-size: 13px;
  font-weight: 500;
  color: var(--ink-100);
}
.kb-source__meta {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-500);
  margin-top: 4px;
}
.kb-source__snippet {
  font-size: 13px;
  color: var(--ink-300);
  margin-top: 8px;
  line-height: 1.5;
  border-top: 1px dashed var(--ink-700);
  padding-top: 8px;
}

/* ==========================================================================
   13. Dataframes — spreadsheet grade
   ========================================================================== */
[data-testid="stDataFrame"] {
  border: 1px solid var(--ink-700);
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: var(--shadow-xs);
}
[data-testid="stDataFrame"] [role="columnheader"] {
  background: var(--ink-900) !important;
  color: var(--ink-300) !important;
  font-family: var(--font-sans) !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
[data-testid="stDataFrame"] [role="gridcell"] {
  font-family: var(--font-sans) !important;
  font-size: 13px !important;
  color: var(--ink-100) !important;
  font-variant-numeric: tabular-nums;
}

/* ==========================================================================
   14. Alerts
   ========================================================================== */
.stAlert {
  border-radius: var(--radius-md);
  border: 1px solid;
  padding: 12px 16px;
  font-size: 13px;
}
.stAlert[data-baseweb="notification"][kind="info"]    { background: var(--accent-50);  color: var(--accent-600);  border-color: var(--accent-600); }
.stAlert[data-baseweb="notification"][kind="success"] { background: var(--success-50); color: var(--success-600); border-color: var(--success-600); }
.stAlert[data-baseweb="notification"][kind="warning"] { background: var(--warn-50);    color: var(--warn-600);    border-color: var(--warn-600); }
.stAlert[data-baseweb="notification"][kind="error"]   { background: var(--danger-50);  color: var(--danger-600);  border-color: var(--danger-600); }

/* ==========================================================================
   15. Dividers, radios, checkboxes
   ========================================================================== */
hr { border: none; border-top: 1px solid var(--ink-700); margin: 24px 0; }
.stRadio label, .stCheckbox label { font-size: 13px; color: var(--ink-100); }

/* ==========================================================================
   16. Mobile responsive — consumer UI priority
   ========================================================================== */
@media (max-width: 768px) {
  .block-container { padding: 1rem !important; max-width: 100%; }
  h1 { font-size: 24px; line-height: 32px; }
  h2 { font-size: 19px; line-height: 26px; }
  .stTabs [data-baseweb="tab-list"] {
    flex-wrap: wrap;
    width: 100%;
    justify-content: flex-start;
  }
  .stTabs [data-baseweb="tab"] { font-size: 12px; padding: 6px 10px; }
  [data-testid="stChatInput"] { border-radius: var(--radius-md); }
  .kb-metric__value { font-size: 22px; }
  .kb-card { padding: 14px; }
  /* Collapse sidebar affordance is native — just make sure content breathes */
  section[data-testid="stSidebar"] { width: 85% !important; }
}

/* Contributor UI can keep desktop feel on tablet (>=768) */
@media (max-width: 480px) {
  .stTabs [data-baseweb="tab-list"] { overflow-x: auto; flex-wrap: nowrap; }
  .kb-metric { padding: 12px 14px; }
}
</style>
"""


def apply_design_system() -> None:
    """Inject the unified design system CSS. Call once per app, right after st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


# Helper renderers for the richer custom components
# (Optional — use where st-native widgets aren't expressive enough.)

def render_citation_marker(num: int, source_title: str = "") -> str:
    """Inline [n] citation marker with hover title. Use inside st.markdown(..., unsafe_allow_html=True)."""
    title = f' title="{source_title}"' if source_title else ""
    return f'<a class="citation-marker" href="#source-{num}"{title}>{num}</a>'


def render_confidence(score: float) -> str:
    """Small confidence indicator, 0.0–1.0."""
    pct = int(round(score * 100))
    return (
        f'<span class="kb-confidence">'
        f'<span class="kb-confidence-track"><span class="kb-confidence-fill" style="width: {pct}%"></span></span>'
        f'{pct}% confidence</span>'
    )


def render_badge(text: str, variant: str = "neutral", dot: bool = False) -> str:
    """Variant: success | warn | danger | neutral | accent."""
    dot_html = '<span class="dot"></span>' if dot else ""
    return f'<span class="kb-badge kb-badge--{variant}">{dot_html}{text}</span>'


def render_source(num: int, title: str, meta: str, snippet: str) -> str:
    """Evidence card for RAG sources."""
    return (
        f'<div class="kb-source" id="source-{num}">'
        f'<span class="kb-source__num">[{num}]</span>'
        f'<span class="kb-source__title">{title}</span>'
        f'<div class="kb-source__meta">{meta}</div>'
        f'<div class="kb-source__snippet">{snippet}</div>'
        f'</div>'
    )


def render_metric(label: str, value: str, delta: str | None = None, delta_down: bool = False) -> str:
    """Metric card for the pod health dashboard."""
    delta_html = ""
    if delta:
        cls = "kb-metric__delta--down" if delta_down else ""
        delta_html = f'<div class="kb-metric__delta {cls}">{delta}</div>'
    return (
        f'<div class="kb-metric">'
        f'<div class="kb-metric__label">{label}</div>'
        f'<div class="kb-metric__value">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )
