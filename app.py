
# ============================================================
# KATTA WEALTH INSIGHTS — SINGLE FILE STREAMLIT APP
# ============================================================
# Educational use only. No financial, tax, or legal advice.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import datetime as dt
import io
import os
import matplotlib.pyplot as plt
from typing import Dict

# ---------------- Streamlit Config ----------------
st.set_page_config(
    page_title="Katta Wealth Insights",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- Session Init ----------------
def init_session():
    defaults = {
        "app_mode": "about",
        "current_user": None,
        "is_paid": False,
        "portfolio_raw": None,
        "portfolio": None,
        "portfolio_meta": {},
        "chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ---------------- Database ----------------
DB_PATH = "kwi_app.db"

def db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def db_init():
    conn = db_connect()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            pw_hash TEXT NOT NULL,
            is_paid INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );"""
    )
    conn.commit()
    conn.close()

db_init()

def now_iso():
    return dt.datetime.utcnow().isoformat()

def hash_pw(pw: str) -> str:
    return hashlib.sha256(("kwi_salt_" + pw).encode()).hexdigest()

def db_get_user(email: str):
    conn = db_connect()
    row = conn.execute(
        "SELECT email, pw_hash, is_paid FROM users WHERE email=?",
        (email,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"email": row[0], "pw": row[1], "is_paid": bool(row[2])}

def db_create_user(email: str, pw_hash: str) -> bool:
    try:
        conn = db_connect()
        conn.execute(
            "INSERT INTO users VALUES (?,?,?,?)",
            (email, pw_hash, 0, now_iso()),
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def db_mark_user_paid(email: str):
    conn = db_connect()
    conn.execute("UPDATE users SET is_paid=1 WHERE email=?", (email,))
    conn.commit()
    conn.close()

# ---------------- Top Nav ----------------
def render_top_nav():
    st.markdown(
        """
        <style>
        .topnav {display:flex;justify-content:space-between;
        padding:0.7rem 1.5rem;background:#0f172a;border-bottom:1px solid #1f2937}
        .brand {font-weight:800;font-size:1.3rem;color:#e5e7eb}
        </style>
        <div class='topnav'>
            <div class='brand'>💎 Katta Wealth Insights</div>
        </div>
        """, unsafe_allow_html=True
    )

# ---------------- Marketing Pages ----------------
def render_about():
    st.header("About Katta Wealth Insights")
    st.write("Education-first portfolio analytics platform.")

def render_features():
    st.header("Features")
    st.write("Portfolio analytics, simulations, AI explanations.")

def render_benefits():
    st.header("Benefits")
    st.write("Clarity, education, safety.")

# ---------------- Auth ----------------
def render_auth():
    st.header("Sign In")
    email = st.text_input("Email")
    pw = st.text_input("Password", type="password")
    if st.button("Log In"):
        user = db_get_user(email)
        if user and user["pw"] == hash_pw(pw):
            st.session_state.current_user = email
            st.session_state.is_paid = user["is_paid"]
            st.success("Logged in")
            st.rerun()
        else:
            st.error("Invalid credentials")

# ---------------- Portfolio ----------------
REQUIRED_COLUMNS = ["Ticker", "Shares", "Cost_Basis"]

def compute_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        mv = r["Shares"] * r["Cost_Basis"]
        rows.append({
            "Ticker": r["Ticker"],
            "Shares": r["Shares"],
            "Market Value": mv
        })
    out = pd.DataFrame(rows)
    out.loc["TOTAL", "Market Value"] = out["Market Value"].sum()
    return out

def render_allocation_chart():
    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        return
    df = portfolio.drop(index="TOTAL")
    fig, ax = plt.subplots()
    ax.pie(df["Market Value"], labels=df["Ticker"], autopct="%1.1f%%")
    st.pyplot(fig)

def render_portfolio_overview():
    st.header("Portfolio Overview")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        raw = pd.read_csv(uploaded)
        portfolio = compute_portfolio(raw)
        st.session_state.portfolio = portfolio
        st.dataframe(portfolio)
        render_allocation_chart()

# ---------------- Groq AI ----------------
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

def get_groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key or not GROQ_AVAILABLE:
        return None
    return Groq(api_key=key)

def generate_ai_response(question: str) -> str:
    client = get_groq_client()
    if not client:
        return "AI unavailable. Configure GROQ_API_KEY."
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "Educational assistant only."},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return completion.choices[0].message.content.strip()

def render_ai_chatbot():
    st.header("AI Chatbot")
    q = st.text_input("Ask a question")
    if st.button("Ask"):
        a = generate_ai_response(q)
        st.write(a)

# ---------------- Router ----------------
def route_app():
    render_top_nav()
    mode = st.sidebar.selectbox("Mode", ["about", "features", "benefits", "demo", "ai"])
    if mode == "about":
        render_about()
    elif mode == "features":
        render_features()
    elif mode == "benefits":
        render_benefits()
    elif mode == "demo":
        if not st.session_state.current_user:
            render_auth()
        else:
            render_portfolio_overview()
    elif mode == "ai":
        render_ai_chatbot()

route_app()
