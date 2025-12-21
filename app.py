from __future__ import annotations
import os, json, uuid, hashlib, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
from fpdf import FPDF

# =====================
# CONFIG
# =====================
APP_NAME = "Katta Wealth Insights"
PRO_PRICE = 24
FREE_AI_USES = 2
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(APP_NAME, layout="wide")

# =====================
# SESSION STATE
# =====================
st.session_state.setdefault("user", None)
st.session_state.setdefault("tier", "Free")
st.session_state.setdefault("ai_uses", 0)
st.session_state.setdefault("portfolio", None)

# =====================
# AUTH (SIMPLIFIED)
# =====================
def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

if not st.session_state.user:
    st.title("🔐 Login")
    email = st.text_input("Email")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        st.session_state.user = email
        st.session_state.tier = "Pro" if email.endswith("@demo.com") else "Free"
        st.rerun()
    st.stop()

# =====================
# SIDEBAR
# =====================
st.sidebar.markdown(f"**User:** {st.session_state.user}")
st.sidebar.markdown(f"**Plan:** {st.session_state.tier}")

if st.session_state.tier != "Pro":
    st.sidebar.caption(f"AI uses: {st.session_state.ai_uses}/{FREE_AI_USES}")
    if st.sidebar.button("Upgrade to Pro (demo)"):
        st.session_state.tier = "Pro"
        st.rerun()

# =====================
# ANALYTICS
# =====================
def load_prices(tickers):
    return yf.download(tickers, period="1y")["Adj Close"].dropna()

def portfolio_returns(prices, w):
    return (prices.pct_change().dropna() * w).sum(axis=1)

def cumulative(r): return (1+r).cumprod()-1
def vol(r): return r.std()*np.sqrt(252)
def sharpe(r): return (r.mean()*252-0.02)/(r.std()*np.sqrt(252))
def max_dd(c): return (c - c.cummax()).min()

# =====================
# MAIN UI
# =====================
st.title("📊 AI Portfolio Analyzer")
tabs = st.tabs(["Portfolio", "AI Advisor", "Report"])

# =====================
# PORTFOLIO TAB
# =====================
with tabs[0]:
    st.subheader("Upload Portfolio")
    sample = pd.DataFrame({"Ticker":["AAPL","MSFT","VOO"],"Weight":[0.4,0.3,0.3]})
    st.download_button("Sample CSV", sample.to_csv(index=False), "portfolio.csv")

    f = st.file_uploader("Upload CSV", type="csv")
    if f:
        port = pd.read_csv(f)
        port["Weight"] /= port["Weight"].sum()
        st.session_state.portfolio = port
        st.dataframe(port)

        prices = load_prices(port["Ticker"].tolist())
        r = portfolio_returns(prices, port["Weight"].values)
        c = cumulative(r)

        st.metric("Total Return", f"{c.iloc[-1]*100:.2f}%")
        st.plotly_chart(px.line(c, title="Cumulative Return"), use_container_width=True)

        if st.session_state.tier == "Pro":
            c1,c2,c3 = st.columns(3)
            c1.metric("Volatility", f"{vol(r):.2%}")
            c2.metric("Sharpe", f"{sharpe(r):.2f}")
            c3.metric("Max Drawdown", f"{max_dd(c):.2%}")
        else:
            st.info("🔒 Risk metrics are Pro features")

# =====================
# AI TAB
# =====================
with tabs[1]:
    q = st.text_area("Ask about your portfolio")
    if st.button("Explain"):
        if st.session_state.tier != "Pro" and st.session_state.ai_uses >= FREE_AI_USES:
            st.warning("Upgrade to Pro for unlimited AI")
        else:
            st.session_state.ai_uses += 1
            st.markdown(
                f"""
**AI Insight**
- Portfolio is diversified across {len(st.session_state.portfolio)} holdings
- Risk depends on equity exposure
- Performance reflects market trends
*(Decision-support only)*
"""
            )

# =====================
# PDF REPORT
# =====================
with tabs[2]:
    if st.session_state.tier != "Pro":
        st.info("🔒 Reports are Pro features")
    else:
        if st.button("Generate PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0,10,"Katta Wealth Insights Report",ln=True)
            pdf.multi_cell(0,8, json.dumps(st.session_state.portfolio.to_dict(), indent=2))
            st.download_button(
                "Download PDF",
                pdf.output(dest="S").encode("latin-1"),
                "report.pdf",
                "application/pdf"
            )
