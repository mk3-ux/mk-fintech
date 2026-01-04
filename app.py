# ============================================================
# KATTA WEALTH INSIGHTS — CANONICAL SINGLE-FILE APP
# GUARANTEED ORDER • NO NAMEERRORS • STREAMLIT SAFE
# ============================================================

# ----------------------------
# IMPORTS (TOP, ONLY ONCE)
# ----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import datetime as dt
import json
import os

# ----------------------------
# STREAMLIT CONFIG (FIRST CALL)
# ----------------------------
st.set_page_config(
    page_title="Katta Wealth Insights",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SESSION INITIALIZATION (AUTHORITATIVE)
# ============================================================

def init_session():
    defaults = {
        "app_mode": "about",
        "current_user": None,
        "is_paid": False,
        "portfolio_df": None,
        "chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ============================================================
# DATABASE (SQLITE — SAFE)
# ============================================================

DB_PATH = "kwi.db"

def db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def db_init():
    conn = db_connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            is_paid INTEGER,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

db_init()

def mark_user_paid(email):
    conn = db_connect()
    conn.execute(
        "UPDATE users SET is_paid=1 WHERE email=?",
        (email,)
    )
    conn.commit()
    conn.close()

# ============================================================
# LEGAL (ALWAYS AVAILABLE)
# ============================================================

def render_about_us_legal():
    st.markdown("## ℹ️ About & Legal")
    st.markdown("""
    **Educational platform only.**  
    No financial, investment, tax, or legal advice.  
    All analytics are hypothetical and illustrative.
    """)

def render_legal_always_on():
    with st.expander("About & Legal (Always Available)", expanded=False):
        render_about_us_legal()
    st.markdown(
        "<div style='font-size:0.8rem;color:#9ca3af;text-align:center;'>"
        "Educational use only · No advice · No guarantees"
        "</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# TOP NAVIGATION
# ============================================================

def render_top_nav():
    cols = st.columns(6)
    buttons = [
        ("About", "about"),
        ("Features", "features"),
        ("How", "how"),
        ("Benefits", "benefits"),
        ("Legal", "legal"),
        ("Demo", "demo"),
    ]
    for i, (label, mode) in enumerate(buttons):
        if cols[i].button(label, use_container_width=True):
            st.session_state.app_mode = mode
            st.rerun()

# ============================================================
# MARKETING PAGES
# ============================================================

def render_about():
    st.header("💎 Katta Wealth Insights")
    st.write("Education-first portfolio analytics.")

def render_features():
    st.header("✨ Features")
    st.write("- Portfolio analytics\n- Monte Carlo goals\n- AI explanations")

def render_how():
    st.header("🛠️ How It Works")
    st.write("Upload → Analyze → Learn")

def render_benefits():
    st.header("🎯 Benefits")
    st.write("Clarity. Education. No hype.")

# ============================================================
# AUTH + PAYWALL
# ============================================================

def enforce_auth_and_paywall():
    if not st.session_state.current_user:
        st.info("Demo login auto-enabled.")
        st.session_state.current_user = "demo@user"
        st.session_state.is_paid = True
        return True
    if not st.session_state.is_paid:
        st.warning("Upgrade required.")
        return False
    return True

# ============================================================
# PORTFOLIO ENGINE
# ============================================================

def render_portfolio_overview():
    st.header("📊 Portfolio Overview")

    df = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "VOO"],
        "MarketValue": [10000, 8000, 12000],
    })
    df["Weight"] = df["MarketValue"] / df["MarketValue"].sum()

    st.session_state.portfolio_df = df
    st.dataframe(df, use_container_width=True)

# ============================================================
# MONTE CARLO
# ============================================================

def render_goal_probability():
    st.header("🎯 Goal Probability")

    df = st.session_state.get("portfolio_df")
    if df is None:
        st.info("Load portfolio first.")
        return

    start = df["MarketValue"].sum()
    goal = st.number_input("Goal ($)", 0.0, 1_000_000.0, 500_000.0)

    sims = []
    for _ in range(3000):
        value = start
        for _ in range(20):
            value *= (1 + np.random.normal(0.07, 0.15))
        sims.append(value)

    prob = round(np.mean(np.array(sims) >= goal) * 100, 2)
    st.metric("Probability", f"{prob}%")

# ============================================================
# AI CHATBOT (SAFE FALLBACK)
# ============================================================

def render_chatbot():
    st.header("🤖 AI Chatbot")

    q = st.text_input("Ask a concept question")
    if st.button("Ask"):
        st.session_state.chat_history.append(
            {"q": q, "a": "This is an educational explanation."}
        )

    for msg in st.session_state.chat_history[-5:]:
        st.write("You:", msg["q"])
        st.write("AI:", msg["a"])

# ============================================================
# SIDEBAR + DEMO ROUTER
# ============================================================

def render_sidebar():
    return st.sidebar.radio(
        "Demo",
        [
            "Portfolio",
            "Goals",
            "Chatbot",
        ],
    )

def render_demo_router(page):
    if page == "Portfolio":
        render_portfolio_overview()
    elif page == "Goals":
        render_goal_probability()
    elif page == "Chatbot":
        render_chatbot()

# ============================================================
# FINAL ROUTER (SINGLE SOURCE OF TRUTH)
# ============================================================

def route_app():
    init_session()
    render_top_nav()

    mode = st.session_state.app_mode

    if mode == "about":
        render_about()
    elif mode == "features":
        render_features()
    elif mode == "how":
        render_how()
    elif mode == "benefits":
        render_benefits()
    elif mode == "legal":
        render_about_us_legal()
    elif mode == "demo":
        if enforce_auth_and_paywall():
            page = render_sidebar()
            render_demo_router(page)
    else:
        render_about()

    render_legal_always_on()

# ============================================================
# ENTRYPOINT (ONLY ONE)
# ============================================================

route_app()
