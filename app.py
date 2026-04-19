from __future__ import annotations

import streamlit as st

from src.agent import RealEstateAgent
from src.config import get_settings
from src.llm import LLMClient
from src.pricing import PricingEngine
from src.rag import RAGEngine

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tokyo Real Estate Concierge",
    page_icon="🏯",
    layout="wide",
)

# ── Premium Dark Theme CSS ───────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Fonts ────────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600;700;800&display=swap');

/* ── Design Tokens ────────────────────────────────────────────────────────── */
:root {
    --bg:              #07070d;
    --bg-glass:        rgba(255, 255, 255, 0.025);
    --bg-glass-hover:  rgba(255, 255, 255, 0.055);
    --border:          rgba(255, 255, 255, 0.06);
    --border-hover:    rgba(255, 255, 255, 0.12);
    --gold:            #c9a84c;
    --gold-light:      #e8d48b;
    --gold-glow:       rgba(201, 168, 76, 0.12);
    --blue:            #4a6cf7;
    --blue-glow:       rgba(74, 108, 247, 0.12);
    --emerald:         #34d399;
    --amber:           #fbbf24;
    --rose:            #f87171;
    --text:            #e4dfd4;
    --text-secondary:  #8e899a;
    --text-muted:      #504b60;
    --radius-lg:       18px;
    --radius:          14px;
    --radius-sm:       10px;
    --pill:            999px;
    --shadow:          0 8px 32px rgba(0, 0, 0, 0.4);
    --shadow-sm:       0 4px 16px rgba(0, 0, 0, 0.28);
    --ease:            cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Base ─────────────────────────────────────────────────────────────────── */
html, body, .stApp {
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background-image:
        radial-gradient(ellipse at 8% 0%, rgba(74, 108, 247, 0.07) 0%, transparent 55%),
        radial-gradient(ellipse at 92% 100%, rgba(201, 168, 76, 0.05) 0%, transparent 55%) !important;
    background-size: 200% 200% !important;
    animation: bg-drift 22s ease-in-out infinite alternate !important;
}

/* ── Global Text ──────────────────────────────────────────────────────────── */
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] *,
.stApp [data-testid="stText"],
.stApp [data-testid="stText"] *,
.stApp label, .stApp p, .stApp li, .stApp span {
    color: var(--text) !important;
}
.stApp a { color: var(--gold) !important; text-decoration: none !important; }
.stApp a:hover { color: var(--gold-light) !important; }

/* ── Hide Streamlit Chrome ────────────────────────────────────────────────── */
#MainMenu, footer, header, [data-testid="stDecoration"] { display: none !important; }

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(201, 168, 76, 0.18); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(201, 168, 76, 0.38); }

/* ── Main Container ───────────────────────────────────────────────────────── */
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 980px !important;
    padding: 1.5rem 1rem 3rem !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(8, 8, 16, 0.94) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
    color: var(--text) !important;
}

.sb-brand { text-align: center; padding: 1.8rem 1rem 0.8rem; }
.sb-icon  { font-size: 2.8rem; margin-bottom: 0.3rem; filter: drop-shadow(0 0 12px rgba(201,168,76,0.35)); }
.sb-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.22rem; font-weight: 700;
    background: linear-gradient(135deg, var(--gold), var(--gold-light));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sb-sub { font-size: 0.66rem; color: var(--text-muted) !important; text-transform: uppercase; letter-spacing: 0.17em; margin-top: 0.15rem; }

.sb-status {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    margin: 0.9rem 1.2rem; padding: 7px 16px;
    border-radius: var(--pill); background: var(--bg-glass); border: 1px solid var(--border);
    font-size: 0.74rem; font-weight: 500; color: var(--text-secondary) !important;
}
.sb-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; animation: pulse 2s ease-in-out infinite; }

.sb-hr { height: 1px; background: linear-gradient(to right, transparent, var(--border), transparent); margin: 0.7rem 1.5rem; }

.sb-section { padding: 0.3rem 1.3rem; }
.sb-heading {
    font-size: 0.63rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.14em; color: var(--text-muted) !important; margin-bottom: 0.5rem;
}
.sb-item  { padding: 0.28rem 0; font-size: 0.82rem; color: var(--text-secondary) !important; }
.sb-tags  { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 0.15rem; }
.sb-tag   {
    padding: 3px 10px; border-radius: var(--pill);
    font-size: 0.66rem; font-weight: 600;
    background: var(--gold-glow); border: 1px solid rgba(201,168,76,0.18); color: var(--gold) !important;
}

/* ── Hero Card ────────────────────────────────────────────────────────────── */
.hero {
    position: relative; overflow: hidden;
    background: var(--bg-glass); backdrop-filter: blur(14px);
    border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: 2.2rem 2.5rem 2rem; margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
}
.hero::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--gold), var(--blue), var(--gold-light), var(--gold));
    background-size: 300% auto; animation: gradient-flow 5s linear infinite;
}
.hero-wm {
    position: absolute; right: 2.2rem; top: 50%; transform: translateY(-50%);
    font-size: 6.5rem; opacity: 0.035; pointer-events: none; user-select: none;
}
.hero-kicker {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.22em; color: var(--gold) !important; margin-bottom: 0.6rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif; font-weight: 800;
    font-size: 2.35rem; line-height: 1.15; margin: 0 0 0.55rem;
    background: linear-gradient(135deg, var(--text) 30%, var(--gold-light) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-desc { color: var(--text-secondary) !important; font-size: 0.92rem; line-height: 1.65; max-width: 590px; margin: 0; }

/* ── Section Helpers ──────────────────────────────────────────────────────── */
.section-label {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.16em; color: var(--text-muted) !important;
    margin-bottom: 0.55rem; padding-left: 2px;
}
.section-divider {
    height: 1px; margin: 1.2rem 0 1.4rem;
    background: linear-gradient(to right, transparent, rgba(201,168,76,0.22), transparent);
}

/* ── Quick-Prompt Buttons ─────────────────────────────────────────────────── */
div[data-testid="stButton"] > button {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 0.78rem 1rem !important;
    text-align: left !important;
    backdrop-filter: blur(8px);
    transition: all 0.3s var(--ease) !important;
    box-shadow: var(--shadow-sm) !important;
}
div[data-testid="stButton"] > button:hover {
    background: var(--bg-glass-hover) !important;
    border-color: rgba(201,168,76,0.28) !important;
    color: var(--text) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 36px rgba(201,168,76,0.07) !important;
}
div[data-testid="stButton"] > button:active,
div[data-testid="stButton"] > button:focus {
    color: var(--text) !important;
    border-color: rgba(201,168,76,0.4) !important;
    box-shadow: 0 0 0 3px var(--gold-glow) !important;
}

/* ── Chat Messages ────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid rgba(201,168,76,0.28) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.55rem !important;
    box-shadow: var(--shadow-sm) !important;
    backdrop-filter: blur(10px);
    animation: fade-up 0.35s ease-out;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
    color: var(--text) !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
    color: var(--gold-light) !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
    background: rgba(255,255,255,0.06) !important;
    color: var(--gold-light) !important;
    padding: 1px 6px; border-radius: 4px;
}

/* ── Chat Input ───────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-glass) !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s var(--ease);
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(201,168,76,0.32) !important;
    box-shadow: 0 0 0 3px var(--gold-glow) !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] input::placeholder {
    color: var(--text-muted) !important; opacity: 1 !important;
}

/* ── Source Pills ─────────────────────────────────────────────────────────── */
.source-pill {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; margin-right: 5px; margin-top: 0.6rem;
    border-radius: var(--pill); font-size: 0.7rem; font-weight: 600;
    background: var(--gold-glow); border: 1px solid rgba(201,168,76,0.18);
    color: var(--gold) !important;
}

/* ── Confidence Badge ─────────────────────────────────────────────────────── */
.conf { display: inline-flex; align-items: center; gap: 6px; padding: 3px 11px; border-radius: var(--pill); font-size: 0.67rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; margin-top: 0.5rem; }
.conf-high   { background: rgba(52,211,153,0.08); color: var(--emerald) !important; border: 1px solid rgba(52,211,153,0.16); }
.conf-mid    { background: rgba(251,191,36,0.08); color: var(--amber) !important;  border: 1px solid rgba(251,191,36,0.16); }
.conf-low    { background: rgba(248,113,113,0.08); color: var(--rose) !important;   border: 1px solid rgba(248,113,113,0.16); }
.conf-dot    { width: 6px; height: 6px; border-radius: 50%; background: currentColor; display: inline-block; }

/* ── Alerts & Spinner ─────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    backdrop-filter: blur(8px);
}
[data-testid="stSpinner"] * { color: var(--gold) !important; }

/* ── Keyframes ────────────────────────────────────────────────────────────── */
@keyframes bg-drift      { 0% { background-position: 0% 0%; } 100% { background-position: 100% 100%; } }
@keyframes gradient-flow { to  { background-position: 300% center; } }
@keyframes fade-up       { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulse         { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
""",
    unsafe_allow_html=True,
)


# ── Load Backend ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing Tokyo Real Estate Intelligence …")
def load_assistant() -> tuple[RealEstateAgent, LLMClient]:
    """Load and cache all backend components."""
    settings = get_settings()

    pricing = PricingEngine(settings)
    pricing.load()

    llm = LLMClient(settings)
    rag = RAGEngine(settings, llm)
    rag.ensure_index()

    agent = RealEstateAgent(settings=settings, llm=llm, rag=rag, pricing=pricing)
    return agent, llm


agent, llm = load_assistant()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div class="sb-brand">
        <div class="sb-icon">🏯</div>
        <div class="sb-title">Tokyo RE Intel</div>
        <div class="sb-sub">Real Estate Concierge</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    status_color = "#34d399" if llm.configured else "#f87171"
    status_label = "LLM Connected" if llm.configured else "Fallback Mode"
    st.markdown(
        f"""
    <div class="sb-status">
        <span class="sb-dot" style="background:{status_color}"></span>
        {status_label}
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)

    st.markdown(
        """
    <div class="sb-section">
        <div class="sb-heading">Capabilities</div>
        <div class="sb-item">🎯&ensp;ML Price Prediction</div>
        <div class="sb-item">📊&ensp;Market Analytics</div>
        <div class="sb-item">🔍&ensp;Hybrid RAG Retrieval</div>
        <div class="sb-item">🤖&ensp;Agentic Reasoning</div>
        <div class="sb-item">🛡️&ensp;Output Guardrails</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)

    st.markdown(
        """
    <div class="sb-section">
        <div class="sb-heading">Technology</div>
        <div class="sb-tags">
            <span class="sb-tag">LangGraph</span>
            <span class="sb-tag">Groq</span>
            <span class="sb-tag">Random Forest</span>
            <span class="sb-tag">Hybrid RAG</span>
            <span class="sb-tag">Streamlit</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)

    st.markdown(
        """
    <div style="text-align:center; padding:0.6rem 0;">
        <div style="font-size:0.62rem; color:#504b60; letter-spacing:0.12em; text-transform:uppercase;">Built with</div>
        <div style="font-size:0.74rem; color:#8e899a; margin-top:0.2rem;">Streamlit · LangGraph · Groq</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ── Hero Header ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <div class="hero-wm">🏯</div>
    <div class="hero-kicker">Tokyo Market Intelligence</div>
    <h1>Real Estate Concierge</h1>
    <p class="hero-desc">
        Agentic AI assistant for property valuation, market analytics, and investment
        insights — powered by retrieval-augmented generation and ML pricing models.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

if not llm.configured:
    st.warning(
        "⚠️  GROQ_API_KEY is not configured. Responses will use deterministic fallback mode."
    )

# ── Quick Prompts ────────────────────────────────────────────────────────────
QUICK_PROMPTS = [
    {"icon": "📊", "label": "Top Regions by Price", "query": "Which region has the highest average trade price?"},
    {"icon": "💰", "label": "Investment Strategy", "query": "Where should I invest right now to maximize profit potential?"},
    {"icon": "📈", "label": "Market Overview", "query": "Show the overall average and median trade price."},
    {"icon": "📅", "label": "Yearly Price Trends", "query": "What is the yearly price trend in the dataset?"},
    {"icon": "🏠", "label": "Estimate a Property", "query": "Estimate price for land area 220 and floor area 130, built in 2014, 12 min walk to station."},
]

st.markdown('<div class="section-label">✦ Suggested Queries</div>', unsafe_allow_html=True)

row1 = st.columns(3)
for idx, p in enumerate(QUICK_PROMPTS[:3]):
    with row1[idx]:
        if st.button(f"{p['icon']}  {p['label']}", key=f"qp_{idx}", use_container_width=True):
            st.session_state["pending_prompt"] = p["query"]

row2 = st.columns(3)
for idx, p in enumerate(QUICK_PROMPTS[3:]):
    with row2[idx]:
        if st.button(f"{p['icon']}  {p['label']}", key=f"qp_{idx + 3}", use_container_width=True):
            st.session_state["pending_prompt"] = p["query"]

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _render_sources(sources: list[str]) -> None:
    """Render source citations as gold pills."""
    names = sorted({s.split("#", 1)[0] for s in sources if s})
    if names:
        pills = " ".join(f'<span class="source-pill">📄 {n}</span>' for n in names)
        st.markdown(f"<div>{pills}</div>", unsafe_allow_html=True)


def _render_confidence(confidence: float | None) -> None:
    """Render a colour-coded confidence badge."""
    if confidence is None:
        return
    if confidence >= 0.8:
        label, cls = "High", "conf-high"
    elif confidence >= 0.5:
        label, cls = "Moderate", "conf-mid"
    else:
        label, cls = "Low", "conf-low"
    st.markdown(
        f'<div class="conf {cls}"><span class="conf-dot"></span>{label} confidence · {confidence:.0%}</div>',
        unsafe_allow_html=True,
    )


# ── Chat State ───────────────────────────────────────────────────────────────
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome to **Tokyo Real Estate Intelligence**. I can help you with:\n\n"
                "• **Price estimation** — ML-powered valuations for any property\n"
                "• **Market analytics** — Top regions, trends, and investment signals\n"
                "• **Knowledge base** — Insights from Tokyo's real estate landscape\n\n"
                "What would you like to explore?"
            ),
            "sources": [],
            "confidence": None,
        }
    ]

# ── Display Chat History ─────────────────────────────────────────────────────
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        _render_sources(msg.get("sources") or [])
        _render_confidence(msg.get("confidence"))

# ── Handle Input ─────────────────────────────────────────────────────────────
chat_input = st.chat_input("Ask about valuation, trends, investment regions, or request a price estimate …")
user_prompt = chat_input or st.session_state.pop("pending_prompt", None)

if user_prompt:
    st.session_state.chat_messages.append(
        {"role": "user", "content": user_prompt, "sources": [], "confidence": None}
    )
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question …"):
            result = agent.ask(user_prompt)

        st.markdown(result.answer)
        _render_sources(result.citations)
        _render_confidence(result.confidence)

        friendly_warnings = [
            w for w in result.warnings
            if "I need property details" in w or "Low confidence" in w
        ]
        if friendly_warnings:
            st.info(friendly_warnings[0])

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": result.citations,
            "confidence": result.confidence,
        }
    )
