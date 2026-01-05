from __future__ import annotations

import os
import json
import sqlite3
import hashlib
import datetime as dt
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional deps
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from groq import Groq
except Exception:
    Groq = None


# ============================================================
# APP CONFIG
# ============================================================

APP_NAME = "Katta Wealth Insights"
DB_PATH = "kwi.db"
MODEL_NAME = "llama-3.1-8b-instant"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title=APP_NAME, layout="wide")


# ============================================================
# SESSION STATE
# ============================================================

def init_session():
    defaults = {
        "user": None,
        "tier": "Free",
        "portfolio": None,
        "chat": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_session()


# ============================================================
# DATABASE
# ============================================================

def db_connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def db_init():
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                pw_hash TEXT,
                tier TEXT,
                created_at TEXT
            )
        """)

db_init()

def hash_pw(pw: str) -> str:
    return hashlib.sha256(("kwi_" + pw).encode()).hexdigest()

def db_get_user(email: str):
    with db_connect() as conn:
        row = conn.execute(
            "SELECT email, pw_hash, tier FROM users WHERE email=?",
            (email,),
        ).fetchone()
    return row

def db_create_user(email, pw_hash):
    try:
        with db_connect() as conn:
            conn.execute(
                "INSERT INTO users VALUES (?,?,?,?)",
                (email, pw_hash, "Free", dt.datetime.utcnow().isoformat()),
            )
        return True
    except Exception:
        return False


# ============================================================
# AUTH UI
# ============================================================

def auth_ui():
    st.header("🔐 Sign In")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            user = db_get_user(email)
            if not user or user[1] != hash_pw(pw):
                st.error("Invalid credentials")
                return
            st.session_state.user = email
            st.session_state.tier = user[2]
            st.rerun()

    with tab2:
        email = st.text_input("New Email")
        pw = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            if db_create_user(email, hash_pw(pw)):
                st.success("Account created. Please log in.")
            else:
                st.error("Account already exists.")


# ============================================================
# PORTFOLIO HELPERS
# ============================================================

def portfolio_template():
    return "Ticker,Shares\nAAPL,10\nMSFT,5\n"

def validate_portfolio(df: pd.DataFrame):
    if not {"Ticker", "Shares"}.issubset(df.columns):
        return False, "CSV must contain Ticker and Shares"
    return True, ""

def compute_portfolio(df: pd.DataFrame):
    rows = []
    total = 0

    for _, r in df.iterrows():
        price = 0
        if yf:
            try:
                price = yf.Ticker(r["Ticker"]).history(period="1d")["Close"].iloc[-1]
            except Exception:
                pass

        mv = price * r["Shares"]
        total += mv

        rows.append({
            "Ticker": r["Ticker"],
            "Shares": r["Shares"],
            "Price": round(price, 2),
            "Market Value": round(mv, 2),
        })

    out = pd.DataFrame(rows)
    out.loc["TOTAL"] = ["TOTAL", "", "", round(total, 2)]
    return out


# ============================================================
# PORTFOLIO PAGE
# ============================================================

def render_portfolio():
    st.header("📊 Portfolio")

    st.download_button(
        "Download CSV Template",
        portfolio_template(),
        "portfolio_template.csv",
    )

    file = st.file_uploader("Upload Portfolio CSV", type="csv")
    if not file:
        return

    raw = pd.read_csv(file)
    ok, msg = validate_portfolio(raw)
    if not ok:
        st.error(msg)
        return

    p = compute_portfolio(raw)
    st.session_state.portfolio = p
    st.dataframe(p, use_container_width=True)

    fig, ax = plt.subplots()
    p.drop(index="TOTAL").set_index("Ticker")["Market Value"].plot.pie(ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)


# ============================================================
# AI CHAT
# ============================================================

def ai(prompt: str) -> str:
    if Groq is None or not GROQ_API_KEY:
        return "AI not configured."
    client = Groq(api_key=GROQ_API_KEY)
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content


def render_chatbot():
    st.header("💬 AI Assistant")
    q = st.text_input("Ask a question")
    if st.button("Send") and q:
        st.markdown(ai(q))


# ============================================================
# SIDEBAR & ROUTER
# ============================================================

def sidebar():
    with st.sidebar:
        page = st.radio("Navigate", ["Portfolio", "AI Chatbot", "Logout"])
        return page

def run_app():
    if not st.session_state.user:
        auth_ui()
        return

    page = sidebar()

    if page == "Portfolio":
        render_portfolio()
    elif page == "AI Chatbot":
        render_chatbot()
    elif page == "Logout":
        st.session_state.user = None
        st.rerun()


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    run_app()
