import os
import streamlit as st
import pandas as pd
import altair as alt
from groq import Groq
from fpdf import FPDF
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None

# =========================
# CONFIG
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

st.set_page_config(
    page_title="Katta Wealth Insights",
    layout="wide",
    page_icon="📈",
)

# =========================
# UI STYLING
# =========================
COLORS = {
    "bg": "#F8FAFC",
    "text": "#0F172A",
    "card": "#FFFFFF",
    "accent": "#2563EB",
    "subtle": "#E5E7EB",
}

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}
        .block-container {{ max-width: 1200px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# DATA MODELS
# =========================
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

# =========================
# HELPERS
# =========================
def get_realtime_stock_data(ticker):
    if yf is None:
        return None
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2d")
        last = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2]
        return {
            "price": round(last, 2),
            "change_pct": round((last - prev) / prev * 100, 2),
        }
    except Exception:
        return None


def compute_sector_sensitivity(stock_move, primary_sector):
    rows = []
    for s in sectors:
        sensitivity = 1.0 if s == primary_sector else 0.35
        score = sensitivity * stock_move
        rows.append({"Sector": s, "Impact Score": score})

    df = pd.DataFrame(rows)
    max_abs = df["Impact Score"].abs().max()
    df["Impact Score"] = (df["Impact Score"] / max_abs * 5).round(2)

    def label(x):
        if x <= -3:
            return "Higher downside sensitivity"
        if x < -1:
            return "Moderate downside sensitivity"
        if x < 1:
            return "Largely neutral"
        if x < 3:
            return "Moderate upside sensitivity"
        return "Higher upside sensitivity"

    df["Interpretation"] = df["Impact Score"].apply(label)
    return df


def compute_portfolio_exposure(portfolio, sector_df):
    portfolio["Weight"] = portfolio["Allocation"] / portfolio["Allocation"].sum()
    merged = portfolio.merge(sector_df, on="Sector", how="left")
    merged["Weighted Impact"] = merged["Weight"] * merged["Impact Score"]
    return merged["Weighted Impact"].sum(), merged


def call_ai_wealth(history, user_text, style):
    if client is None:
        return "AI not configured."

    system = (
        "You are a senior wealth management research analyst. "
        "Write advisor-ready insights focused on diversification, risk, and long-term planning. "
        "Do NOT give buy/sell advice or price targets."
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return completion.choices[0].message.content.strip()


def create_pdf(title, scenario, sector_df, ai_summary=""):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Scenario: {scenario}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Sector Sensitivity Overview", ln=True)
    pdf.set_font("Helvetica", "", 10)

    for _, r in sector_df.iterrows():
        pdf.cell(
            0,
            6,
            f"{r['Sector']}: {r['Impact Score']} ({r['Interpretation']})",
            ln=True,
        )

    if ai_summary:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Advisor Commentary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, ai_summary)

    return pdf.output(dest="S").encode("latin-1")

# =========================
# HEADER
# =========================
st.markdown(
    """
    <div style="background:#EFF6FF;padding:16px;border-radius:12px;">
        <h2>Katta Wealth Insights</h2>
        <p>Client-focused portfolio analysis • Risk-aware scenarios • Advisor-ready insights</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# TABS
# =========================
tab_market, tab_portfolio, tab_whatif, tab_ai, tab_report = st.tabs(
    [
        "Market Scenario Explorer",
        "Portfolio Analyzer",
        "What-If Builder",
        "AI Wealth Assistant",
        "Generate Client Report",
    ]
)

# =========================
# MARKET SCENARIO
# =========================
with tab_market:
    st.subheader("Market Scenario Explorer")

    ticker = st.text_input("Representative stock or market leader", "AAPL")
    primary_sector = st.selectbox("Primary economic exposure", sectors)
    move = st.slider("Assumed price move (%)", -20, 20, 0)

    live = get_realtime_stock_data(ticker)
    if live:
        st.metric(
            f"{ticker} current price",
            live["price"],
            delta=f"{live['change_pct']}%",
        )

    sector_df = compute_sector_sensitivity(move, primary_sector)
    st.session_state["sector_df"] = sector_df
    st.session_state["scenario"] = f"{ticker} {move:+.1f}%"

    st.dataframe(sector_df, use_container_width=True)

    chart = (
        alt.Chart(sector_df)
        .mark_bar()
        .encode(
            x="Sector",
            y="Impact Score",
            tooltip=["Sector", "Impact Score", "Interpretation"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

# =========================
# PORTFOLIO ANALYZER
# =========================
with tab_portfolio:
    st.subheader("Portfolio Exposure Analysis")
    uploaded = st.file_uploader("Upload portfolio CSV (Sector, Allocation)", type="csv")

    if uploaded:
        portfolio = pd.read_csv(uploaded)
        st.dataframe(portfolio)

        score, breakdown = compute_portfolio_exposure(
            portfolio, st.session_state["sector_df"]
        )

        st.metric("Overall Portfolio Sensitivity Score", f"{score:.2f}")
        st.dataframe(breakdown, use_container_width=True)

# =========================
# WHAT-IF
# =========================
with tab_whatif:
    st.subheader("What-If Allocation Builder")
    if uploaded:
        multipliers = {}
        for s in portfolio["Sector"]:
            multipliers[s] = st.slider(f"{s} adjustment", 0.5, 1.5, 1.0)

        adjusted = portfolio.copy()
        adjusted["Allocation"] = adjusted.apply(
            lambda r: r["Allocation"] * multipliers[r["Sector"]], axis=1
        )

        new_score, _ = compute_portfolio_exposure(
            adjusted, st.session_state["sector_df"]
        )

        st.metric(
            "Adjusted Portfolio Sensitivity",
            f"{new_score:.2f}",
            delta=f"{new_score - score:+.2f}",
        )

# =========================
# AI ASSISTANT
# =========================
with tab_ai:
    st.subheader("AI Wealth Research Assistant")
    prompt = st.text_area(
        "Ask for advisor-ready insights (e.g. client explanation, risk summary)"
    )

    if st.button("Generate"):
        context = f"Scenario: {st.session_state['scenario']}\n{st.session_state['sector_df']}"
        output = call_ai_wealth([], context + "\n" + prompt, "Professional")
        st.markdown(output)
        st.session_state["ai_output"] = output

# =========================
# REPORT
# =========================
with tab_report:
    st.subheader("Generate Client-Ready PDF")
    title = st.text_input("Report title", "Client Portfolio Scenario Review")

    if st.button("Create PDF"):
        pdf = create_pdf(
            title,
            st.session_state["scenario"],
            st.session_state["sector_df"],
            st.session_state.get("ai_output", ""),
        )
        st.download_button(
            "Download PDF",
            pdf,
            file_name="client_portfolio_report.pdf",
            mime="application/pdf",
        )

st.caption("Decision-support only. Not investment advice.")
