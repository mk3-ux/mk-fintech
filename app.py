import os
import streamlit as st
import pandas as pd
import altair as alt
from groq import Groq
from fpdf import FPDF
import hashlib

try:
    import yfinance as yf
except ImportError:
    yf = None

# ============================================================
# CONFIG
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

st.set_page_config(
    page_title="Katta Wealth Insights",
    layout="wide",
    page_icon="📈",
)

# ============================================================
# SIMPLE USER DATABASE (DEMO)
# ============================================================
# In production → PostgreSQL / Firebase / Supabase
if "users" not in st.session_state:
    st.session_state.users = {}  # email → {password_hash, tier}

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "ai_uses" not in st.session_state:
    st.session_state.ai_uses = 0

# ============================================================
# UTILS
# ============================================================
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def is_logged_in():
    return st.session_state.current_user is not None

def current_tier():
    if not is_logged_in():
        return "Free"
    return st.session_state.users[st.session_state.current_user]["tier"]

# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
        .block-container { max-width: 1200px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# AUTH UI
# ============================================================
def auth_screen():
    st.title("🔐 Katta Wealth Insights")
    st.subheader("Create an account or log in")

    tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        pw = st.text_input("Password", type="password", key="login_pw")

        if st.button("Log In"):
            user = st.session_state.users.get(email)
            if not user or user["password"] != hash_password(pw):
                st.error("Invalid email or password")
            else:
                st.session_state.current_user = email
                st.success("Logged in successfully")
                st.rerun()

    with tab_signup:
        email = st.text_input("Email", key="signup_email")
        pw = st.text_input("Password", type="password", key="signup_pw")

        if st.button("Create Account"):
            if email in st.session_state.users:
                st.error("Account already exists")
            else:
                st.session_state.users[email] = {
                    "password": hash_password(pw),
                    "tier": "Free",
                }
                st.success("Account created. Please log in.")

# ============================================================
# PAYMENT (DEMO STRIPE FLOW)
# ============================================================
def upgrade_to_pro():
    st.subheader("💎 Upgrade to Pro")

    st.markdown(
        """
        **Pro includes:**
        - Unlimited AI insights  
        - Client risk profiles  
        - Advanced reports  
        - Scenario comparisons  
        """
    )

    st.info("💳 Demo payment — Stripe-ready architecture")

    if st.button("Pay $29/month (Demo)"):
        # 🔐 STRIPE WEBHOOK WOULD GO HERE
        st.session_state.users[st.session_state.current_user]["tier"] = "Pro"
        st.success("🎉 Upgrade successful! Pro features unlocked.")
        st.rerun()

# ============================================================
# DATA MODELS
# ============================================================
sectors = [
    "Technology",
    "Financials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Healthcare",
    "Energy",
    "Real Estate",
    "Fixed Income",
]

# ============================================================
# HELPERS
# ============================================================
def get_live_stock(ticker):
    if yf is None:
        return None
    try:
        hist = yf.Ticker(ticker).history(period="2d")
        last, prev = hist["Close"].iloc[-1], hist["Close"].iloc[-2]
        return {
            "price": round(last, 2),
            "change_pct": round((last - prev) / prev * 100, 2),
        }
    except Exception:
        return None


def compute_sector_sensitivity(move, primary_sector):
    rows = []
    for s in sectors:
        sensitivity = 1.0 if s == primary_sector else 0.35
        rows.append(
            {
                "Sector": s,
                "Impact Score": round(sensitivity * move, 2),
            }
        )

    df = pd.DataFrame(rows)
    max_abs = df["Impact Score"].abs().max()
    df["Impact Score"] = (df["Impact Score"] / max_abs * 5).round(2)

    df["Interpretation"] = df["Impact Score"].apply(
        lambda x: "Higher downside risk" if x < -2 else
                  "Moderate sensitivity" if abs(x) < 2 else
                  "Higher upside exposure"
    )
    return df


def compute_portfolio_exposure(portfolio, sector_df):
    portfolio = portfolio.copy()
    portfolio["Weight"] = portfolio["Allocation"] / portfolio["Allocation"].sum()
    merged = portfolio.merge(sector_df, on="Sector", how="left")
    merged["Weighted Impact"] = merged["Weight"] * merged["Impact Score"]
    return merged["Weighted Impact"].sum(), merged


def call_ai(prompt):
    if client is None:
        return "AI not configured."

    if current_tier() == "Free":
        if st.session_state.ai_uses >= 2:
            return "🔒 Free tier AI limit reached. Upgrade to Pro."
        st.session_state.ai_uses += 1

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior wealth management analyst. "
                    "Explain scenarios in client-friendly language. "
                    "No investment advice."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return completion.choices[0].message.content.strip()

# ============================================================
# MAIN APP
# ============================================================
if not is_logged_in():
    auth_screen()
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"**Logged in as:** {st.session_state.current_user}")
    st.markdown(f"**Plan:** {current_tier()}")

    if current_tier() == "Free":
        if st.button("Upgrade to Pro"):
            st.session_state.show_upgrade = True
    else:
        st.success("Pro features active")

    if st.button("Log out"):
        st.session_state.current_user = None
        st.session_state.ai_uses = 0
        st.rerun()

# ============================================================
# UPGRADE MODAL
# ============================================================
if st.session_state.get("show_upgrade"):
    upgrade_to_pro()
    st.stop()

# ============================================================
# HEADER
# ============================================================
st.title("📈 Katta Wealth Insights")
st.caption("Client-focused portfolio analysis • Decision support only")

# ============================================================
# TABS
# ============================================================
tab_market, tab_portfolio, tab_ai = st.tabs(
    ["Market Scenario", "Portfolio Analyzer", "AI Assistant"]
)

# ---------------- MARKET SCENARIO ----------------
with tab_market:
    ticker = st.text_input("Representative stock", "AAPL")
    sector = st.selectbox("Primary sector", sectors)
    move = st.slider("Assumed price move (%)", -20, 20, 0)

    live = get_live_stock(ticker)
    if live:
        st.metric(ticker, live["price"], f"{live['change_pct']}%")

    sector_df = compute_sector_sensitivity(move, sector)
    st.session_state["sector_df"] = sector_df
    st.dataframe(sector_df, use_container_width=True)

# ---------------- PORTFOLIO ----------------
with tab_portfolio:
    uploaded = st.file_uploader("Upload portfolio CSV", type="csv")

    if uploaded:
        portfolio = pd.read_csv(uploaded)
        score, breakdown = compute_portfolio_exposure(
            portfolio, st.session_state["sector_df"]
        )
        st.metric("Portfolio Sensitivity Score", f"{score:.2f}")
        st.dataframe(breakdown, use_container_width=True)

# ---------------- AI ----------------
with tab_ai:
    prompt = st.text_area("Ask the AI to explain this scenario to a client")

    if st.button("Generate Insight"):
        context = f"Sector impacts:\n{st.session_state.get('sector_df')}"
        st.markdown(call_ai(context + "\n\n" + prompt))

        if current_tier() == "Free":
            st.caption(f"Free AI uses: {st.session_state.ai_uses}/2")

st.caption("Educational & decision-support only. Not investment advice.")
