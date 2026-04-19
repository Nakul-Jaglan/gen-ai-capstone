from __future__ import annotations

import streamlit as st

from src.agent import RealEstateAgent
from src.config import get_settings
from src.llm import LLMClient
from src.pricing import PricingEngine
from src.rag import RAGEngine

st.set_page_config(
    page_title="Tokyo Real Estate Concierge",
    page_icon=":material/temple_buddhist:",
    layout="wide",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@500;600;700&family=Zen+Kaku+Gothic+New:wght@400;500;700&display=swap');

:root {
  --paper: #f6f0e2;
  --paper-strong: #fffaf0;
  --ink: #1d2730;
  --muted: #5a6673;
  --line: #d9c9a5;
  --vermilion: #b12933;
  --indigo: #1f3a5c;
  --moss: #4f7069;
  --gold: #a07a33;
}

html, body, .stApp {
  font-family: 'Zen Kaku Gothic New', sans-serif;
  background:
    radial-gradient(circle at 12% 8%, #fdf7ea 0%, var(--paper) 36%),
    radial-gradient(circle at 88% 85%, #efe5cd 0%, var(--paper) 38%),
    linear-gradient(to right, rgba(175, 140, 72, 0.06) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(175, 140, 72, 0.06) 1px, transparent 1px);
  background-size: auto, auto, 22px 22px, 22px 22px;
  color: var(--ink) !important;
}

.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] *,
.stApp [data-testid="stText"],
.stApp [data-testid="stText"] *,
.stApp label,
.stApp p,
.stApp li {
  color: var(--ink) !important;
}

.stApp a {
  color: var(--indigo) !important;
}

#MainMenu, footer, header {
  visibility: hidden;
}

.app-shell {
  max-width: 1040px;
  margin: 0 auto;
  padding-bottom: 2rem;
}

.mast {
  background: linear-gradient(145deg, rgba(255, 250, 240, 0.96), rgba(255, 255, 255, 0.94));
  border: 1px solid var(--line);
  border-top: 6px solid var(--vermilion);
  border-radius: 18px;
  padding: 1.35rem 1.45rem;
  margin-bottom: 0.95rem;
  box-shadow: 0 10px 28px rgba(46, 31, 12, 0.08);
}

.mast-kicker {
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--moss) !important;
  margin-bottom: 0.3rem;
}

.mast h1 {
  margin: 0;
  font-size: 2rem;
  line-height: 1.2;
  font-weight: 700;
  font-family: 'Noto Serif JP', serif;
  color: var(--indigo) !important;
}

.mast p {
  margin: 0.45rem 0 0;
  color: var(--muted);
  line-height: 1.55;
  font-size: 1rem;
}

.divider {
  height: 1px;
  background: linear-gradient(to right, rgba(177, 41, 51, 0), rgba(177, 41, 51, 0.7), rgba(177, 41, 51, 0));
  margin: 0.6rem 0 0.95rem;
}

.prompt-title {
  margin: 0.2rem 0 0.55rem;
  color: var(--moss) !important;
  font-size: 0.83rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.09em;
}

div[data-testid="stButton"] > button {
  border-radius: 999px !important;
  border: 1px solid #7f1b25 !important;
  background: linear-gradient(135deg, #ba2e37, #8e1f2a) !important;
  color: #fff8ef !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em !important;
  padding: 0.54rem 0.95rem !important;
  box-shadow: 0 3px 10px rgba(108, 20, 31, 0.26) !important;
}

div[data-testid="stButton"] > button:hover {
  border-color: #6b111b !important;
  background: linear-gradient(135deg, #9f1f29, #79131f) !important;
  color: #fff8ef !important;
}

div[data-testid="stButton"] > button:focus,
div[data-testid="stButton"] > button:active {
  color: #fff8ef !important;
  box-shadow: 0 0 0 0.2rem rgba(177, 41, 51, 0.25) !important;
}

[data-testid="stChatMessage"] {
  background: var(--paper-strong) !important;
  border: 1px solid var(--line) !important;
  border-left: 4px solid var(--indigo) !important;
  border-radius: 14px !important;
  padding: 0.45rem 0.65rem !important;
  box-shadow: 0 3px 10px rgba(31, 58, 92, 0.08) !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
  color: var(--ink) !important;
}

[data-testid="stChatInput"] {
  background: var(--paper-strong) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
  background: #fffdf8 !important;
  color: var(--ink) !important;
}

[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] input::placeholder {
  color: var(--muted) !important;
  opacity: 1 !important;
}

.source-note {
  margin-top: 0.55rem;
  display: inline-block;
  color: #3f4e5d !important;
  background: #f1e6cb;
  border: 1px solid #d5bf8a;
  border-radius: 999px;
  padding: 0.18rem 0.62rem;
  font-size: 0.82rem;
}

[data-testid="stAlert"] {
  border-radius: 12px !important;
}

[data-testid="stSpinner"] * {
  color: var(--indigo) !important;
}

</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing assistant...")
def load_assistant() -> tuple[RealEstateAgent, LLMClient]:
    """
    Load the required components for the Real Estate Concierge assistant.
    Initializes pricing engine, RAG engine, LLM client, and the main agent orchestrator.
    """
    settings = get_settings()

    pricing = PricingEngine(settings)
    pricing.load()

    llm = LLMClient(settings)
    rag = RAGEngine(settings, llm)
    rag.ensure_index()

    agent = RealEstateAgent(settings=settings, llm=llm, rag=rag, pricing=pricing)
    return agent, llm


agent, llm = load_assistant()

st.markdown('<div class="app-shell">', unsafe_allow_html=True)

st.markdown(
    """
<div class="mast">
  <div class="mast-kicker">Tokyo Market Desk</div>
  <h1>Real Estate Intelligence Concierge</h1>
  <p>
    Japanese-inspired calm interface for agentic valuation, market trends, and data-backed insights.
    Ask naturally and get grounded answers from tools plus retrieval.
  </p>
</div>
<div class="divider"></div>
""",
    unsafe_allow_html=True,
)

if not llm.configured:
  st.warning("GROQ_API_KEY is not configured in environment. Some answers may fall back to deterministic mode.")

quick_prompts = [
    "Which region has the highest average trade price?",
    "Where should I invest right now to maximize profit potential?",
    "Show the overall average and median trade price.",
    "What is the yearly price trend in the dataset?",
    "Estimate price for land area 220 and floor area 130, built in 2014, 12 min walk to station.",
]

st.markdown('<div class="prompt-title">Try a Prompt</div>', unsafe_allow_html=True)

cols = st.columns(2)
for idx, text in enumerate(quick_prompts):
    with cols[idx % 2]:
        with st.container(border=False):
            if st.button(text, key=f"quick_{idx}", use_container_width=True):
                st.session_state["pending_prompt"] = text

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
        "content": "Welcome. I can help with valuation, price leaders, trends, and investment-opportunity signals from transaction data.",
            "sources": [],
        }
    ]

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        sources = msg.get("sources") or []
        if sources:
            source_names = sorted({src.split("#", 1)[0] for src in sources if src})
            st.markdown(
                f"<div class='source-note'>Sources: {', '.join(source_names)}</div>",
                unsafe_allow_html=True,
            )

chat_input = st.chat_input("Ask about valuation, trends, top investment regions, or request an estimate...")
user_prompt = chat_input or st.session_state.pop("pending_prompt", None)

if user_prompt:
    st.session_state.chat_messages.append({"role": "user", "content": user_prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question..."):
            result = agent.ask(user_prompt)

        st.markdown(result.answer)

        source_names = sorted({src.split("#", 1)[0] for src in result.citations if src})
        if source_names:
            st.markdown(
                f"<div class='source-note'>Sources: {', '.join(source_names)}</div>",
                unsafe_allow_html=True,
            )

        friendly_warnings = [
            w
            for w in result.warnings
            if "I need property details" in w or "Low confidence" in w
        ]
        if friendly_warnings:
            st.info(friendly_warnings[0])

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": result.citations,
        }
    )

st.markdown("</div>", unsafe_allow_html=True)
