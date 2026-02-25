"""
app.py — Legal Research RAG Chatbot
Run with: streamlit run app.py
(Requires the API server to be running: uvicorn src.api.server:app --port 8000)

Model is set in .env:
    MODEL=gemma-3-27b-it          # default — free, 14 K requests/day
    MODEL=gemini-2.5-flash-lite   # premium — 20 requests/day
"""

import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = "http://localhost:8000"
MODEL        = os.getenv("MODEL", "gemma-3-27b-it")

st.set_page_config(
    page_title="LegalAI — BNS & BNSS",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu, footer, header { display: none !important; }
    .stApp { background-color: #0f0f0f; color: #e3e3e3; }
    section[data-testid="stSidebar"] { background: #0f0f0f; }
    [data-testid="stChatMessage"] { background: transparent !important; }
    .stButton > button { border-radius: 8px !important; }
    h1 a, h2 a, h3 a, h4 a { display: none !important; }

    /* ── Hide ALL Streamlit running/status/toolbar indicators ─────────────── */
    [data-testid="stStatusWidget"]  { display: none !important; }
    [data-testid="stToolbar"]       { display: none !important; }
    [data-testid="stDecoration"]    { display: none !important; }
    [data-testid="stHeader"]        { display: none !important; }
    [data-testid="stAppRunningIcon"]{ display: none !important; }
    .stAppRunningIcon               { display: none !important; }
    [aria-label="Running..."]       { display: none !important; }

    /* ── stBottom container — opaque so content doesn't bleed through ───── */
    [data-testid="stBottom"] {
        background: #0f0f0f !important;
        padding-top: 4px !important;
    }

    /* ── "💬 Your Question:" label — centered above the search bar ──────── */
    [data-testid="stBottom"]::before {
        content:        "💬  Your Question:";
        display:        block;
        max-width:      736px;
        margin:         0 auto;
        background:     #0f0f0f;
        color:          #666;
        font-size:      0.78rem;
        letter-spacing: 0.04em;
        padding:        6px 16px 2px 16px;
    }

    /* ── Hide stale content instantly (stops hero ghosting during load) ─────── */
    div[data-stale="true"] { opacity: 0 !important; pointer-events: none !important; }

    /* ── Hide chat message avatars ───────────────────────────────────────── */
    [data-testid*="stChatMessageAvatar"] { display: none !important; }

    /* ── User message bubble — rendered via custom HTML, not st.chat_message ── */
    .user-bubble-wrap {
        display: flex; justify-content: flex-end;
        margin: 4px 0 12px 0;
    }
    .user-bubble {
        background:    #2a2a2a;
        border-radius: 16px 16px 4px 16px;
        padding:       10px 18px;
        max-width:     80%;
        color:         #e3e3e3;
        font-size:     1rem;
        line-height:   1.6;
    }

    /* ── All buttons — ghost style, text wraps inside ────────────────────── */
    .stButton > button {
        background:   transparent !important;
        border:       1px solid #3a3a3a !important;
        color:        #777 !important;
        font-size:    0.8rem !important;
        padding:      8px 16px !important;
        border-radius:20px !important;
        white-space:  normal !important;
        word-wrap:    break-word !important;
        height:       auto !important;
        line-height:  1.4 !important;
        text-align:   center !important;
        transition:   border-color 0.2s, color 0.2s !important;
    }
    .stButton > button:hover {
        border-color: #666 !important;
        color:        #bbb !important;
    }

    /* ── Chat input — clean, simple ──────────────────────────────────────── */
    [data-testid="stChatInput"] textarea {
        background: #1c1c1c !important;
        color:      #e3e3e3 !important;
        font-size:  1rem !important;
    }

    /* ── Loading screen ───────────────────────────────────────────────────── */
    .loading-wrap {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; min-height: 70vh; gap: 14px; text-align: center;
    }
    .loading-icon { font-size: 80px; animation: pulse 1.8s ease-in-out infinite; }
    @keyframes pulse {
        0%, 100% { transform: scale(1);   opacity: 1; }
        50%       { transform: scale(1.1); opacity: 0.7; }
    }
    .loading-title { font-size: 2rem; font-weight: 700; margin: 0; color: #e3e3e3; }
    .loading-sub   { margin: 0; color: #888; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Backend health check — runs ONCE per browser session ──────────────────────
def _ensure_backend():
    if "backend_ok" not in st.session_state:
        try:
            ok = requests.get(f"{BACKEND_URL}/health", timeout=2).status_code == 200
        except Exception:
            ok = False
        st.session_state.backend_ok = ok
    return st.session_state.backend_ok


# ── First-load: animated loading screen while health-check runs ───────────────
if "backend_ok" not in st.session_state:
    st.markdown("""
    <div class="loading-wrap">
        <div class="loading-icon">⚖️</div>
        <p class="loading-title">LegalAI</p>
        <p class="loading-sub">Connecting....</p>
    </div>
    """, unsafe_allow_html=True)
    _ensure_backend()
    st.rerun()
    st.stop()


# ── User bubble (right-aligned, Gemini-style) ──────────────────────────────────
def _user_bubble(text: str):
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f'<div class="user-bubble-wrap"><div class="user-bubble">{safe}</div></div>',
        unsafe_allow_html=True,
    )


# ── Query helper ───────────────────────────────────────────────────────────────
def _run(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    _user_bubble(question)
    with st.chat_message("assistant"):
        with st.spinner("Looking up the law..."):
            try:
                r = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"question": question, "model_name": MODEL},
                    timeout=60,
                )
                res = r.json()
            except Exception:
                st.error("Could not reach the API server. Is it running?\n"
                         "`uvicorn src.api.server:app --port 8000`")
                return
        st.markdown(res["answer"])
    st.session_state.messages.append({
        "role": "assistant", "content": res["answer"],
        "citations": res.get("citations", []), "model": res["model_used"]})


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## ⚖️ LegalAI")
st.markdown("*Ask questions about India's new criminal laws — BNS & BNSS (2024)*")

# ── Backend down banner ────────────────────────────────────────────────────────
if not _ensure_backend():
    st.error("API server is not running. Start it with:\n"
             "```\nuvicorn src.api.server:app --port 8000\n```")
    st.stop()

st.divider()

# ── Hero (empty state) ────────────────────────────────────────────────────────
if not st.session_state.messages and "pending_question" not in st.session_state:
    st.markdown("### Hello! Where should we start?")
    st.markdown("Try one of these questions, or type your own below:")
    st.write("")

    suggestions = [
        "What is the punishment for murder?",
        "What is the procedure for filing an FIR?",
        "Is kidnapping a bailable offence?",
        "What is the difference between robbery and dacoity?",
        "What are the offences related to assault?",
        "What replaced IPC Section 302?",
        "What is the procedure when a person is arrested?",
        "What is the punishment for rape under BNS?",
    ]

    col1, col2 = st.columns(2)
    for i, text in enumerate(suggestions):
        col = col1 if i % 2 == 0 else col2
        col.button(
            text, key=f"s{i}", use_container_width=True,
            on_click=lambda t=text: st.session_state.update({"pending_question": t}),
        )

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        _user_bubble(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# ── Clear button ──────────────────────────────────────────────────────────────
if st.session_state.messages:
    st.write("")
    _, mid, _ = st.columns([3, 1.5, 3])
    with mid:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ── Chat input — registered FIRST so it stays visible during _run() ───────────
if prompt := st.chat_input("Ask about any law, offence or legal procedure...."):
    st.session_state.pending_question = prompt
    st.rerun()

# ── Handle pending question ────────────────────────────────────────────────────
if "pending_question" in st.session_state:
    _run(st.session_state.pop("pending_question"))
    st.rerun()
