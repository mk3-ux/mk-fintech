from __future__ import annotations

import os
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# Optional dependencies
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

APP_NAME = "Educational Wealth Insights"
MODEL_NAME = "llama-3.1-8b-instant"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title=APP_NAME, layout="wide")


# ============================================================
# LEGAL / EDUCATIONAL NOTICE
# ============================================================

LEGAL_NOTICE = """
⚠️ **Educational Use Only**

This application is for **learning and informational purposes only**.

• It does **NOT** provide financial, investment, tax, or legal advice  
• It does **NOT** recommend buying, selling, or holding any asset  
• All metrics are **historical and descriptive**, not predictive  
• Past performance does **not** guarantee future results  

Always consult a qualified financial professional for real decisions.
"""


# ============================================================
# CORE EDUCATIONAL WEALTH ANALYSIS ENGINE
# ============================================================

def educational_wealth_analysis(
    portfolio: List[Dict[str, float]],
    groq_api_key: str | None = None,
) -> Dict:
    """
    EDUCATIONAL wealth analysis only.
    No advice. No recommendations.
    """

    tickers = [p["ticker"].upper() for p in portfolio]
    shares = {p["ticker"].upper(): p["shares"] for p in portfolio}

    if yf is None:
        raise RuntimeError("yfinance is required for live stock data")

    # --------------------------------------------------------
    # LOAD PRICE DATA (1Y)
    # --------------------------------------------------------
    prices = {}
    returns = {}

    for t in tickers:
        hist = yf.Ticker(t).history(period="1y")
        if hist.empty:
            continue
        prices[t] = hist["Close"]
        returns[t] = hist["Close"].pct_change()

    prices_df = pd.DataFrame(prices).dropna()
    returns_df = pd.DataFrame(returns).dropna()

    latest_prices = prices_df.iloc[-1]

    # --------------------------------------------------------
    # MARKET VALUES & WEIGHTS
    # --------------------------------------------------------
    market_values = {
        t: latest_prices[t] * shares[t] for t in tickers
    }

    total_value = sum(market_values.values())
    weights = {t: mv / total_value for t, mv in market_values.items()}

    # --------------------------------------------------------
    # PORTFOLIO RETURNS
    # --------------------------------------------------------
    portfolio_returns = sum(
        returns_df[t] * weights[t] for t in tickers
    )

    # --------------------------------------------------------
    # 12+ EDUCATIONAL FEATURES
    # --------------------------------------------------------
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)

    sharpe = annual_return / annual_vol if annual_vol != 0 else 0

    cumulative = (1 + portfolio_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    spy = yf.Ticker("SPY").history(period="1y")["Close"].pct_change().dropna()
    aligned = portfolio_returns.align(spy, join="inner")

    beta = (
        np.cov(aligned[0], aligned[1])[0][1] / np.var(aligned[1])
        if len(aligned[0]) > 10 else 0
    )

    top_weight = max(weights.values())
    diversification_score = -sum(w * np.log(w) for w in weights.values())

    dividend_income = 0.0
    for t in tickers:
        try:
            divs = yf.Ticker(t).dividends.tail(4).sum()
            dividend_income += divs * shares[t]
        except Exception:
            pass

    dividend_yield = dividend_income / total_value if total_value else 0

    if annual_vol < 0.15:
        risk_bucket = "Lower historical volatility"
    elif annual_vol < 0.25:
        risk_bucket = "Moderate historical volatility"
    else:
        risk_bucket = "Higher historical volatility"

    # --------------------------------------------------------
    # STOCK-LEVEL FEATURES
    # --------------------------------------------------------
    stock_features = {}
    for t in tickers:
        r = returns_df[t]
        stock_features[t] = {
            "price": round(latest_prices[t], 2),
            "market_value": round(market_values[t], 2),
            "weight_pct": round(weights[t] * 100, 2),
            "annual_return": round(r.mean() * 252, 4),
            "annual_volatility": round(r.std() * np.sqrt(252), 4),
        }

    # --------------------------------------------------------
    # AI EDUCATIONAL EXPLANATION (NO ADVICE)
    # --------------------------------------------------------
    ai_text = None
    if Groq and groq_api_key:
        client = Groq(api_key=groq_api_key)
        prompt = f"""
        Explain these portfolio concepts for EDUCATION ONLY:
        - volatility
        - diversification
        - drawdowns
        - beta
        - risk buckets

        Do NOT give investment advice or recommendations.

        Metrics:
        Total value: {round(total_value,2)}
        Annual volatility: {round(annual_vol,2)}
        Max drawdown: {round(max_drawdown,2)}
        """
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Educational finance explainer only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
        )
        ai_text = resp.choices[0].message.content

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "educational_notice": "Educational use only. No financial advice.",
        "portfolio_features": {
            "total_value": round(total_value, 2),
            "annual_return": round(annual_return, 4),
            "annual_volatility": round(annual_vol, 4),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_drawdown, 4),
            "beta_vs_sp500": round(beta, 2),
            "top_holding_pct": round(top_weight * 100, 2),
            "diversification_score": round(diversification_score, 3),
            "dividend_yield": round(dividend_yield, 4),
            "estimated_dividend_income": round(dividend_income, 2),
            "risk_bucket": risk_bucket,
            "num_holdings": len(tickers),
        },
        "stock_features": stock_features,
        "ai_education": ai_text,
    }


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("📘 Educational Wealth Management App")
st.markdown(LEGAL_NOTICE)

st.divider()

st.subheader("📥 Enter Portfolio (Educational)")
st.caption("Any valid stock or ETF ticker can be used.")

portfolio_input = st.text_area(
    "Enter one holding per line (TICKER,SHARES)",
    value="AAPL,10\nMSFT,5\nVOO,3",
)

if st.button("Analyze Portfolio"):
    portfolio = []
    for line in portfolio_input.splitlines():
        try:
            t, s = line.split(",")
            portfolio.append({"ticker": t.strip(), "shares": float(s)})
        except Exception:
            pass

    if not portfolio:
        st.error("Invalid portfolio input.")
    else:
        with st.spinner("Running educational analysis..."):
            result = educational_wealth_analysis(
                portfolio,
                groq_api_key=GROQ_API_KEY,
            )

        st.success("Analysis complete")

        st.subheader("📊 Portfolio Features (Educational)")
        st.json(result["portfolio_features"])

        st.subheader("📄 Stock-Level Features")
        st.dataframe(pd.DataFrame(result["stock_features"]).T)

        if result["ai_education"]:
            st.subheader("🤖 AI Educational Explanation")
            st.markdown(result["ai_education"])

st.divider()
st.caption("© Educational use only — no investment advice.")
