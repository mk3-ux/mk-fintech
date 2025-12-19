import os
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import hashlib
from groq import Groq
from fpdf import FPDF
from streamlit_cookies_manager import EncryptedCookieManager

# ============================================================
# DEV MODE (TURN OFF FOR REAL PAYMENTS)
# ============================================================
DEV_MODE = True

# ============================================================
# COOKIES (PERSIST LOGIN)
# ============================================================
cookies = EncryptedCookieManager(
    prefix="kwi_",
    password="very-secret-key"
)

if not cookies.ready():
    st.stop()

# ============================================================
# CONFIG
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

st.set_page_config("Katta Wealth Insights", layout="wide")

# ============================================================
# IN-MEMORY USER DB (DEMO)
# ============================================================
if "users" not in st.session_state:
    st.session_state.users = {}

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "ai_uses" not in st.session_state:
    st.session_state.ai_uses = 0

# ============================================================
# PASSWORD UTILS
# ============================================================
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def logged_in():
    return st.session_state.current_user is not None

def tier():
    if DEV_MODE:
        return "Pro"
    if not logged_in():
        return "Free"
    return st.session_state.users[st.session_state.current_user]["tier"]

# ============================================================
# AUTO LOGIN FROM COOKIE
# ============================================================
saved_user = cookies.get("user")
if not st.session_state.current_user and saved_user in st.session_state.users:
    st.session_state.current_user = saved_user

# ============================================================
# AUTH UI
# ============================================================
def auth_ui():
    st.title("🔐 Katta Wealth Insights")

    tabs = st.tabs(["Log In", "Sign Up", "Forgot Password"])

    # ---------- LOGIN ----------
    with tabs[0]:
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")

        if st.button("Log In"):
            user = st.session_state.users.get(email)
            if user and user["pw"] == hash_pw(pw):
                st.session_state.current_user = email
                cookies["user"] = email
                cookies.save()
                st.rerun()
            else:
                st.error("Invalid email or password")

    # ---------- SIGN UP ----------
    with tabs[1]:
        email = st.text_input("New Email")
        pw = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if email in st.session_state.users:
                st.error("Account already exists")
            else:
                st.session_state.users[email] = {
                    "pw": hash_pw(pw),
                    "tier": "Free"
                }
                st.success("Account created — please log in")

    # ---------- FORGOT PASSWORD ----------
    with tabs[2]:
        email = st.text_input("Account Email")
        new_pw = st.text_input("New Password", type="password")

        if st.button("Reset Password"):
            if email in st.session_state.users:
                st.session_state.users[email]["pw"] = hash_pw(new_pw)
                st.success("Password reset successfully")
            else:
                st.error("Email not found")

# ============================================================
# PAYMENT / UPGRADE
# ============================================================
def upgrade_ui():
    st.header("💎 Upgrade to Pro")

    st.markdown("""
    **Pro includes**
    - Unlimited AI insights
    - Client risk profiles
    - Advanced portfolio metrics
    - Scenario comparisons
    """)

    if st.button("Pay $29/month (Demo)"):
        st.session_state.users[
            st.session_state.current_user
        ]["tier"] = "Pro"

        st.session_state.show_upgrade = False
        st.success("Pro activated")
        st.rerun()

# ============================================================
# CORE DATA
# ============================================================
SECTORS = [
    "Technology", "Financials", "Healthcare",
    "Consumer", "Energy", "Real Estate", "Fixed Income"
]

# ============================================================
# ANALYTICS
# ============================================================
def sector_impact(move, primary):
    rows = []
    for s in SECTORS:
        impact = move if s == primary else move * 0.35
        rows.append({"Sector": s, "Score": round(impact, 2)})
    df = pd.DataFrame(rows)
    df["Score"] = (df["Score"] / df["Score"].abs().max() * 5).round(2)
    return df

def portfolio_metrics(port):
    weights = port["Allocation"] / port["Allocation"].sum()
    hhi = np.sum(weights ** 2)
    return round(1 - hhi, 2), round(hhi, 2)

# ============================================================
# AI
# ============================================================
def ai(prompt):
    if client is None:
        return "AI unavailable"

    if tier() == "Free":
        if st.session_state.ai_uses >= 2:
            return "🔒 Upgrade to Pro for unlimited AI"
        st.session_state.ai_uses += 1

    return client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Wealth analyst. No investment advice."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
    ).choices[0].message.content

# ============================================================
# START APP
# ============================================================
if not logged_in():
    auth_ui()
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"**Logged in as:** {st.session_state.current_user}")
    st.markdown(f"**Plan:** {tier()}")

    if tier() == "Free":
        if st.button("Upgrade to Pro"):
            st.session_state.show_upgrade = True

    if st.button("Log out"):
        st.session_state.current_user = None
        cookies["user"] = ""
        cookies.save()
        st.rerun()

# ============================================================
# UPGRADE SCREEN (ONLY WHEN CLICKED)
# ============================================================
if st.session_state.get("show_upgrade"):
    upgrade_ui()
    st.stop()

# ============================================================
# MAIN UI
# ============================================================
st.title("📈 Katta Wealth Insights")

tabs = st.tabs([
    "Market Scenario",
    "Portfolio Analyzer",
    "Scenario Comparison (Pro)",
    "Client Profile (Pro)",
    "AI Advisor"
])

# ---------- MARKET ----------
with tabs[0]:
    move = st.slider("Market move (%)", -20, 20, 0)
    sector = st.selectbox("Primary sector", SECTORS)
    s_df = sector_impact(move, sector)
    st.session_state["scenario"] = s_df
    st.dataframe(s_df)

# ---------- PORTFOLIO ----------
with tabs[1]:
    f = st.file_uploader("Upload portfolio CSV", type="csv")
    if f:
        port = pd.read_csv(f)
        st.session_state["portfolio"] = port
        st.dataframe(port)

        score = (port["Allocation"] / port["Allocation"].sum() *
                 s_df["Score"].values[:len(port)]).sum()
        st.metric("Portfolio Sensitivity", round(score, 2))

        if tier() == "Pro":
            div, hhi = portfolio_metrics(port)
            st.metric("Diversification Score", div)
            st.metric("Concentration Risk", hhi)
        else:
            st.info("🔒 Advanced metrics available in Pro")

# ---------- COMPARISON ----------
with tabs[2]:
    if tier() != "Pro":
        st.warning("Pro feature")
    else:
        a = st.slider("Scenario A", -20, 20, 0)
        b = st.slider("Scenario B", -20, 20, 5)
        st.bar_chart(pd.DataFrame({"A": [a], "B": [b]}))

# ---------- CLIENT PROFILE ----------
with tabs[3]:
    if tier() != "Pro":
        st.warning("Pro feature")
    else:
        risk = st.selectbox("Risk tolerance", ["Conservative", "Moderate", "Aggressive"])
        horizon = st.selectbox("Time horizon", ["<5 yrs", "5–10 yrs", "10+ yrs"])
        st.session_state["client"] = {"risk": risk, "horizon": horizon}
        st.success("Client profile saved")

# ---------- AI ----------
with tabs[4]:
    q = st.text_area("Ask AI to explain this to a client")
    if st.button("Generate"):
        context = f"""
        Scenario: {st.session_state.get("scenario")}
        Portfolio: {st.session_state.get("portfolio")}
        Client: {st.session_state.get("client")}
        """
        st.markdown(ai(context + q))

        if tier() == "Free":
            st.caption(f"Free AI uses: {st.session_state.ai_uses}/2")

st.caption("Decision-support only. Not investment advice.")
