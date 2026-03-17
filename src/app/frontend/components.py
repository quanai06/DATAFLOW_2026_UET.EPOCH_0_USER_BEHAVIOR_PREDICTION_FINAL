from __future__ import annotations

from typing import Any

from src.app.frontend.utils import badge_tone


def inject_styles(st: Any) -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(221, 234, 240, 0.9), transparent 30%),
                radial-gradient(circle at top right, rgba(244, 224, 206, 0.75), transparent 28%),
                linear-gradient(180deg, #f8f6f1 0%, #f1eee6 100%);
        }
        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
            color: #111111;
        }
        p, li, span, label, small, strong, div {
            color: inherit;
        }
        .stMarkdown,
        .stMarkdown p,
        .stMarkdown li,
        .stMarkdown span,
        .stMarkdown div,
        .stCaption,
        .stText,
        .stAlert,
        .stInfo,
        .stWarning,
        .stSuccess,
        .stException {
            color: #111111 !important;
        }
        .stSlider label,
        .stNumberInput label,
        .stSelectbox label,
        .stTextInput label,
        .stTextArea label,
        .stMultiSelect label,
        .stCheckbox label,
        .stRadio label,
        .stDateInput label,
        [data-testid="stWidgetLabel"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: #111111 !important;
            font-weight: 600;
        }
        .stNumberInput input,
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] *,
        .stMultiSelect div[data-baseweb="select"] * {
            color: #111111 !important;
        }
        .stSlider [data-baseweb="slider"] * {
            color: #111111 !important;
        }
        .stNumberInput div[data-baseweb="input"] {
            background: #2b2f3a !important;
            border: 1px solid #2b2f3a !important;
            border-radius: 12px !important;
        }
        .stNumberInput div[data-baseweb="input"] input {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background: #2b2f3a !important;
            caret-color: #ffffff !important;
        }
        .stNumberInput button,
        .stNumberInput button svg,
        .stNumberInput [data-testid="stNumberInputStepUp"],
        .stNumberInput [data-testid="stNumberInputStepDown"] {
            color: #ffffff !important;
            fill: #ffffff !important;
            background: #2b2f3a !important;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            background: #2b2f3a !important;
            border: 1px solid #2b2f3a !important;
            border-radius: 12px !important;
            color: #ffffff !important;
        }
        .stSelectbox div[data-baseweb="select"] span,
        .stSelectbox div[data-baseweb="select"] input,
        .stSelectbox div[data-baseweb="select"] svg,
        .stSelectbox div[data-baseweb="select"] [role="combobox"],
        .stSelectbox div[data-baseweb="select"] div,
        .stSelectbox div[data-baseweb="select"] p {
            color: #ffffff !important;
            fill: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        .stTextInput > div > div {
            background: #2b2f3a !important;
            border-radius: 12px !important;
        }
        .stTextInput input {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background: #2b2f3a !important;
            caret-color: #ffffff !important;
        }
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: rgba(255, 255, 255, 0.72) !important;
            -webkit-text-fill-color: rgba(255, 255, 255, 0.72) !important;
        }
        .dashboard-shell {
            padding: 1.5rem 1.4rem 1.2rem 1.4rem;
            border-radius: 24px;
            background: rgba(255, 252, 246, 0.88);
            border: 1px solid rgba(13, 59, 102, 0.08);
            box-shadow: 0 22px 60px rgba(34, 48, 69, 0.08);
        }
        .hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.3rem;
            line-height: 1.05;
            font-weight: 700;
            color: #111111;
            margin-bottom: 0.3rem;
        }
        .hero-subtitle {
            color: #1b1b1b;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        .section-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: #111111;
            margin: 0.4rem 0 0.8rem 0;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(23, 50, 77, 0.09);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            min-height: 120px;
        }
        .metric-label {
            color: #2f2f2f;
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            color: #111111;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }
        .metric-caption {
            color: #2d2d2d;
            font-size: 0.92rem;
            margin-top: 0.25rem;
        }
        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.38rem 0.78rem;
            color: #ffffff !important;
            background-clip: padding-box;
            font-weight: 600;
            font-size: 0.84rem;
            letter-spacing: 0.02em;
            text-shadow: 0 1px 0 rgba(0, 0, 0, 0.12);
        }
        .badge,
        .badge *,
        .badge span,
        .badge div {
            color: #ffffff !important;
            fill: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        .recommendation-panel {
            background: linear-gradient(135deg, #0d3b66 0%, #184e77 55%, #c46b48 100%);
            color: #ffffff !important;
            padding: 1.25rem;
            border-radius: 22px;
            box-shadow: 0 20px 45px rgba(13, 59, 102, 0.22);
        }
        .recommendation-panel h3 {
            font-family: 'Space Grotesk', sans-serif;
            margin: 0 0 0.45rem 0;
            font-size: 1.25rem;
            color: #ffffff !important;
        }
        .recommendation-panel p {
            margin: 0.15rem 0;
            color: rgba(255, 255, 255, 0.92) !important;
        }
        .recommendation-panel strong,
        .recommendation-panel span,
        .recommendation-panel div {
            color: #ffffff !important;
        }
        .stButton > button,
        .stDownloadButton > button,
        button[kind="secondary"],
        button[kind="primary"] {
            background: #233142 !important;
            color: #ffffff !important;
            border: 1px solid #233142 !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        button[kind="secondary"]:hover,
        button[kind="primary"]:hover {
            background: #1b2633 !important;
            color: #ffffff !important;
            border-color: #1b2633 !important;
        }
        .stButton > button *,
        .stDownloadButton > button *,
        button[kind="secondary"] *,
        button[kind="primary"] * {
            color: #ffffff !important;
        }
        .warning-panel {
            background: #fff0eb;
            border: 1px solid #d58b73;
            border-radius: 18px;
            padding: 1rem;
            color: #6f2417;
        }
        .mini-note {
            color: #222222;
            font-size: 0.9rem;
        }
        .chain-panel {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            color: #111111;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(st: Any, title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def metric_card(st: Any, label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(st: Any, text: str) -> None:
    st.markdown(
        f'<span class="badge" style="background:{badge_tone(text)};">{text}</span>',
        unsafe_allow_html=True,
    )


def recommendation_panel(st: Any, decision: dict[str, Any], risk: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="recommendation-panel">
            <h3>{decision['recommended_action']}</h3>
            <p>Produce today: <strong>{decision['today_production_pct']:.1f}%</strong></p>
            <p>Warehouse pressure: <strong>{decision['warehouse_waiting_pressure_pct']:.1f}%</strong></p>
            <p>Priority: <strong>{decision['priority_level']}</strong> | Risk: <strong>{risk['risk_level']}</strong></p>
            <p>{decision['explanation']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def warning_panel(st: Any, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="warning-panel">
            <strong>{title}</strong>
            <div style="margin-top:0.4rem;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
