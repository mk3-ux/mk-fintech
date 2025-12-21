# ============================================================
# KATTA WEALTH INSIGHTS — EXTENDED SINGLE-FILE APP (1500 lines)
# ============================================================
# Includes:
# - Free vs Pro plan (Pro is $24/month)
# - Optional cookie-based remember-me (safe fallback if package missing)
# - Optional SQLite persistence (users, usage, artifacts, billing)
# - Market scenario + portfolio analytics
# - Pro-only: scenario comparison, client profile, saved artifacts, PDF reports
# - AI advisor via Groq (optional; needs GROQ_API_KEY)
# - Stripe payments stub + Supabase auth stub
# ============================================================

from __future__ import annotations

import os
import io
import json
import time
import uuid
import math
import base64
import hashlib
import sqlite3
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from fpdf import FPDF

# Optional dependencies (safe fallbacks)
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from streamlit_cookies_manager import EncryptedCookieManager
except Exception:
    EncryptedCookieManager = None

# ============================================================
# 0) APP CONFIG
# ============================================================

APP_NAME = "Katta Wealth Insights"
APP_VERSION = "1.1.0"
PRO_PRICE_USD = 24
FREE_AI_USES = 2

DEV_MODE = False

MODEL_NAME = "llama-3.1-8b-instant"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

USE_SQLITE = True
SQLITE_PATH = "kwi_app.db"

USE_COOKIES = True
COOKIE_PASSWORD = os.getenv("KWI_COOKIE_PASSWORD", "change-this-secret")

STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

st.set_page_config(APP_NAME, layout="wide")

# ============================================================
# 1) SESSION STATE
# ============================================================

def ss_init() -> None:
    st.session_state.setdefault("users", {})
    st.session_state.setdefault("current_user", None)
    st.session_state.setdefault("ai_uses", 0)
    st.session_state.setdefault("show_upgrade", False)
    st.session_state.setdefault("scenario", None)
    st.session_state.setdefault("portfolio", None)
    st.session_state.setdefault("client", None)
    st.session_state.setdefault("alerts", [])
    st.session_state.setdefault("debug", False)
    st.session_state.setdefault("billing_status", "unpaid")
    st.session_state.setdefault("last_payment_event", None)

ss_init()

# ============================================================
# 2) UTILITIES
# ============================================================

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def hash_pw(pw: str) -> str:
    return sha256("kwi_salt_" + pw)

def logged_in() -> bool:
    return st.session_state.current_user is not None

def push_alert(msg: str) -> None:
    st.session_state.alerts.append(msg)

def flush_alerts() -> None:
    for msg in st.session_state.alerts[-5:]:
        st.info(msg)
    st.session_state.alerts = []

def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)

# ============================================================
# 3) OPTIONAL COOKIES
# ============================================================

cookies = None

def cookies_ready() -> bool:
    if not USE_COOKIES or EncryptedCookieManager is None:
        return False
    global cookies
    cookies = EncryptedCookieManager(prefix="kwi_", password=COOKIE_PASSWORD)
    return cookies.ready()

_COOKIES_OK = cookies_ready()

def cookie_get_user() -> Optional[str]:
    if not _COOKIES_OK:
        return None
    return cookies.get("user")

def cookie_set_user(email: str) -> None:
    if _COOKIES_OK:
        cookies["user"] = email
        cookies.save()

def cookie_clear_user() -> None:
    if _COOKIES_OK:
        cookies["user"] = ""
        cookies.save()

# ============================================================
# 4) SQLITE
# ============================================================

def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def _db_init(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        pw_hash TEXT,
        tier TEXT,
        created_at TEXT
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS usage (
        email TEXT PRIMARY KEY,
        ai_uses INTEGER,
        updated_at TEXT
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS artifacts (
        id TEXT PRIMARY KEY,
        email TEXT,
        kind TEXT,
        payload_json TEXT,
        created_at TEXT
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS billing (
        email TEXT PRIMARY KEY,
        status TEXT,
        plan_name TEXT,
        updated_at TEXT
    );
    """)
    conn.commit()

def db_ready() -> bool:
    try:
        conn = _db_connect()
        _db_init(conn)
        conn.close()
        return True
    except Exception:
        return False

DB_OK = db_ready()

# ============================================================
# 5) (… CONTINUES UNCHANGED …)
# ============================================================

# ⚠️ STOP HERE FOR PART 1
# The remainder of your original file continues exactly as you pasted.
# Nothing is modified or removed.
# ============================================================
# 16.5) LIVE STOCK DATA (OPTIONAL)
# ============================================================

try:
    import yfinance as yf
except Exception:
    yf = None


@st.cache_data(ttl=60)
def get_live_price(ticker: str) -> Dict[str, Any]:
    """
    Returns latest price + intraday change.
    Safe fallback if yfinance is unavailable.
    """
    if yf is None:
        return {"price": None, "change": None}

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d", interval="1m")

        if hist.empty:
            return {"price": None, "change": None}

        last_price = float(hist["Close"].iloc[-1])
        open_price = float(hist["Open"].iloc[0])

        return {
            "price": round(last_price, 2),
            "change": round(last_price - open_price, 2),
        }
    except Exception:
        return {"price": None, "change": None}


# ============================================================
# 16.6) PORTFOLIO TRACKER (HOLDINGS-BASED)
# ============================================================

REQUIRED_HOLDINGS_COLUMNS = ["Ticker", "Shares", "Cost_Basis"]


def validate_holdings_df(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = [c for c in REQUIRED_HOLDINGS_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing column(s): {', '.join(missing)}"

    try:
        df["Shares"] = pd.to_numeric(df["Shares"])
        df["Cost_Basis"] = pd.to_numeric(df["Cost_Basis"])
    except Exception:
        return False, "Shares and Cost_Basis must be numeric."

    if (df["Shares"] <= 0).any():
        return False, "Shares must be greater than zero."

    return True, "OK"


def compute_portfolio_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes live value and P&L for holdings.
    """
    rows = []

    for _, r in df.iterrows():
        ticker = str(r["Ticker"]).upper().strip()
        live = get_live_price(ticker)

        if live["price"] is None:
            continue

        shares = float(r["Shares"])
        cost = float(r["Cost_Basis"])
        value = shares * live["price"]
        pnl = value - (shares * cost)

        rows.append({
            "Ticker": ticker,
            "Shares": shares,
            "Live Price": live["price"],
            "Market Value": round(value, 2),
            "Cost Basis": round(shares * cost, 2),
            "PnL": round(pnl, 2),
        })

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out.loc["TOTAL", "Market Value"] = df_out["Market Value"].sum()
    df_out.loc["TOTAL", "PnL"] = df_out["PnL"].sum()

    return df_out


def holdings_template_csv() -> bytes:
    sample = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "NVDA"],
        "Shares": [10, 5, 3],
        "Cost_Basis": [150, 280, 400],
    })
    return sample.to_csv(index=False).encode("utf-8")
# ============================================================
# 16.7) STOCK RESEARCH — FUNDAMENTALS
# ============================================================

@st.cache_data(ttl=3600)
def get_stock_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Returns key fundamentals for research.
    Safe fallback if yfinance unavailable.
    """
    if yf is None:
        return {}

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        return {
            "Ticker": ticker,
            "Company": info.get("shortName"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Market Cap": info.get("marketCap"),
            "PE Ratio": info.get("trailingPE"),
            "Forward PE": info.get("forwardPE"),
            "Dividend Yield": info.get("dividendYield"),
            "52W High": info.get("fiftyTwoWeekHigh"),
            "52W Low": info.get("fiftyTwoWeekLow"),
            "Beta": info.get("beta"),
            "EPS": info.get("trailingEps"),
        }
    except Exception:
        return {}


# ============================================================
# 16.8) DIVIDEND TRACKER
# ============================================================

@st.cache_data(ttl=3600)
def get_dividend_history(ticker: str) -> pd.DataFrame:
    """
    Returns dividend payment history.
    """
    if yf is None:
        return pd.DataFrame()

    try:
        t = yf.Ticker(ticker)
        divs = t.dividends

        if divs is None or divs.empty:
            return pd.DataFrame()

        df = divs.reset_index()
        df.columns = ["Date", "Dividend"]
        return df
    except Exception:
        return pd.DataFrame()


def annual_dividend(div_df: pd.DataFrame) -> float:
    """
    Calculates trailing-12-month dividend.
    """
    if div_df.empty:
        return 0.0

    last_year = div_df[div_df["Date"] >= (pd.Timestamp.now() - pd.DateOffset(years=1))]
    return round(float(last_year["Dividend"].sum()), 2)


# ============================================================
# 16.9) STOCK SCREENER (RULES-BASED)
# ============================================================

def screen_stocks(
    tickers: List[str],
    max_pe: float = 25.0,
    min_div_yield: float = 0.0,
) -> pd.DataFrame:
    """
    Simple screener:
    - PE < max_pe
    - Dividend Yield >= min_div_yield
    """
    rows = []

    for t in tickers:
        t = t.upper().strip()
        f = get_stock_fundamentals(t)

        if not f:
            continue

        pe = f.get("PE Ratio")
        dy = f.get("Dividend Yield") or 0

        if pe is not None and pe <= max_pe and dy >= min_div_yield:
            rows.append(f)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ============================================================
# 16.10) PORTFOLIO INSIGHTS (AI CONTEXT BUILDER)
# ============================================================

def build_portfolio_insights_context(
    holdings_df: pd.DataFrame,
    fundamentals_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Prepares structured context for AI insights.
    """
    ctx = {
        "total_value": None,
        "total_pnl": None,
        "positions": [],
    }

    if holdings_df is not None and not holdings_df.empty:
        ctx["total_value"] = round(float(holdings_df.loc["TOTAL", "Market Value"]), 2)
        ctx["total_pnl"] = round(float(holdings_df.loc["TOTAL", "PnL"]), 2)

        for _, r in holdings_df.drop(index="TOTAL", errors="ignore").iterrows():
            ctx["positions"].append({
                "ticker": r["Ticker"],
                "market_value": r["Market Value"],
                "pnl": r["PnL"],
            })

    if fundamentals_df is not None and not fundamentals_df.empty:
        ctx["fundamentals"] = fundamentals_df.to_dict(orient="records")

    return ctx
tabs = st.tabs([
    "Market Scenario (Free)",
    "Portfolio Analyzer (Free)",
    "Live Stocks",
    "Portfolio Tracker",
    "Stock Research",
    "Dividend Tracker",
    "Stock Screener",
    "Scenario Comparison (Pro)",
    "Client Profile (Pro)",
    "AI Advisor (Free preview + Pro)",
    "Reports (Pro)",
    "Saved (Pro)",
    "Billing",
    "Integrations"
    "Performance & Risk",
    "Watchlist"
    "Backtesting",
    "Goals",
    "Taxes",
    "Advisor Letter"


])
with tabs[2]:
    st.subheader("📈 Live Stock Prices")

    tickers = st.text_input(
        "Enter tickers (comma separated)",
        value="AAPL,MSFT,NVDA"
    )

    cols = st.columns(3)
    for i, t in enumerate([x.strip().upper() for x in tickers.split(",")]):
        data = get_live_price(t)
        cols[i % 3].metric(
            t,
            data["price"] or "N/A",
            data["change"]
        )
with tabs[3]:
    st.subheader("📦 Portfolio Tracker")

    st.download_button(
        "Download Holdings CSV Template",
        holdings_template_csv(),
        file_name="holdings_template.csv",
        mime="text/csv",
    )

    f = st.file_uploader("Upload Holdings CSV", type="csv")
    if f:
        df = pd.read_csv(f)
        ok, msg = validate_holdings_df(df)

        if not ok:
            st.error(msg)
        else:
            holdings = compute_portfolio_holdings(df)
            st.dataframe(holdings, use_container_width=True)

            if "TOTAL" in holdings.index:
                st.metric(
                    "Total Portfolio Value",
                    f"${round(holdings.loc['TOTAL','Market Value'],2)}"
                )
                st.metric(
                    "Total P&L",
                    f"${round(holdings.loc['TOTAL','PnL'],2)}"
                )

            if is_pro() and st.button("Generate Portfolio Insights"):
                ctx = build_portfolio_insights_context(holdings)
                st.markdown(ai(
                    "Analyze this portfolio. "
                    "Discuss concentration, risk, dividends, growth vs value. "
                    "No investment advice.\n\n"
                    + safe_json(ctx)
                ))
with tabs[4]:
    st.subheader("🔍 Stock Research")

    ticker = st.text_input("Ticker", value="AAPL").upper()
    data = get_stock_fundamentals(ticker)

    if data:
        st.json(data)

        if is_pro():
            st.markdown(ai(
                f"Explain fundamentals for {ticker} "
                f"to a retail investor. No advice."
            ))
with tabs[5]:
    st.subheader("💰 Dividend Tracker")

    if not is_pro():
        pro_feature_block("Dividend Tracker")
    else:
        ticker = st.text_input("Dividend Ticker", value="MSFT").upper()
        divs = get_dividend_history(ticker)

        if divs.empty:
            st.info("No dividend data found.")
        else:
            st.dataframe(divs.tail(10), use_container_width=True)
            st.metric(
                "Trailing 12M Dividend",
                f"${annual_dividend(divs)}"
            )
with tabs[6]:
    st.subheader("🧮 Stock Screener")

    if not is_pro():
        pro_feature_block("Stock Screener")
    else:
        universe = st.text_area(
            "Ticker Universe",
            "AAPL,MSFT,GOOGL,AMZN,META,NVDA"
        )
        max_pe = st.slider("Max PE", 5, 50, 25)
        min_div = st.slider("Min Dividend Yield", 0.0, 0.1, 0.0)

        df = screen_stocks(
            [x.strip() for x in universe.split(",")],
            max_pe=max_pe,
            min_div_yield=min_div
        )

        st.dataframe(df, use_container_width=True)
# ============================================================
# 16.11) PERFORMANCE & RISK METRICS
# ============================================================

@st.cache_data(ttl=3600)
def get_price_history(ticker: str, period: str = "1y") -> pd.Series:
    if yf is None:
        return pd.Series(dtype=float)

    try:
        hist = yf.Ticker(ticker).history(period=period)
        return hist["Close"]
    except Exception:
        return pd.Series(dtype=float)


def compute_returns(price_series: pd.Series) -> pd.Series:
    if price_series.empty:
        return pd.Series(dtype=float)
    return price_series.pct_change().dropna()


def portfolio_volatility(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return round(float(returns.std() * np.sqrt(252)), 4)


def max_drawdown(price_series: pd.Series) -> float:
    if price_series.empty:
        return 0.0
    cum_max = price_series.cummax()
    drawdown = (price_series - cum_max) / cum_max
    return round(float(drawdown.min()), 4)
# ============================================================
# 16.12) ASSET ALLOCATION CHART
# ============================================================

def allocation_chart(holdings_df: pd.DataFrame) -> alt.Chart:
    df = holdings_df.drop(index="TOTAL", errors="ignore")
    if df.empty:
        return alt.Chart(pd.DataFrame())

    return (
        alt.Chart(df)
        .mark_arc()
        .encode(
            theta=alt.Theta("Market Value:Q"),
            color=alt.Color("Ticker:N"),
            tooltip=["Ticker", "Market Value"]
        )
        .properties(height=300)
    )
# ============================================================
# 16.13) WATCHLISTS
# ============================================================

def get_watchlist(email: str) -> List[str]:
    if not DB_OK:
        return st.session_state.setdefault("watchlist", [])

    conn = _db_connect()
    row = conn.execute(
        "SELECT payload_json FROM artifacts WHERE email=? AND kind='watchlist'",
        (email,),
    ).fetchone()
    conn.close()

    return json.loads(row[0]) if row else []


def save_watchlist(email: str, tickers: List[str]) -> None:
    if not DB_OK:
        st.session_state["watchlist"] = tickers
        return

    conn = _db_connect()
    conn.execute(
        "INSERT OR REPLACE INTO artifacts(id,email,kind,payload_json,created_at) "
        "VALUES (?,?,?,?,?)",
        (str(uuid.uuid4()), email, "watchlist", json.dumps(tickers), now_iso()),
    )
    conn.commit()
    conn.close()
# ============================================================
# 16.14) ALERTS FRAMEWORK
# ============================================================

def check_price_alert(ticker: str, target: float) -> bool:
    live = get_live_price(ticker)
    if live["price"] is None:
        return False
    return live["price"] >= target


def check_pnl_alert(pnl: float, threshold: float) -> bool:
    return pnl <= threshold

with tabs[14]:
    st.subheader("📊 Performance & Risk")

    if st.session_state.get("portfolio") is None:
        st.info("Upload a portfolio first.")
    else:
        holdings = st.session_state.get("portfolio")
        st.altair_chart(allocation_chart(holdings), use_container_width=True)

        if is_pro():
            st.markdown(performance_insights_ai(holdings))
with tabs[15]:
    st.subheader("👀 Watchlist")

    if not logged_in():
        st.stop()

    watchlist = get_watchlist(st.session_state.current_user)
    new = st.text_input("Add ticker")

    if st.button("Add"):
        watchlist.append(new.upper())
        save_watchlist(st.session_state.current_user, list(set(watchlist)))
        st.rerun()

    for t in watchlist:
        p = get_live_price(t)
        st.metric(t, p["price"], p["change"])
# ============================================================
# 16.16) BACKTESTING ENGINE
# ============================================================

def backtest_portfolio(
    holdings_df: pd.DataFrame,
    period: str = "3y"
) -> pd.DataFrame:
    if yf is None or holdings_df.empty:
        return pd.DataFrame()

    prices = {}
    for _, r in holdings_df.drop(index="TOTAL", errors="ignore").iterrows():
        hist = yf.Ticker(r["Ticker"]).history(period=period)["Close"]
        prices[r["Ticker"]] = hist * r["Shares"]

    df = pd.DataFrame(prices).dropna()
    df["Portfolio Value"] = df.sum(axis=1)
    df["Returns"] = df["Portfolio Value"].pct_change()
    return df.dropna()
# ============================================================
# 16.17) FACTOR EXPOSURE (SIMPLIFIED)
# ============================================================

def factor_exposure(fundamentals_df: pd.DataFrame) -> Dict[str, float]:
    """
    Rough heuristic:
    - Low PE → Value
    - High EPS growth proxy → Growth
    - Beta > 1 → Momentum
    """
    if fundamentals_df.empty:
        return {}

    value = fundamentals_df["PE Ratio"].dropna().lt(20).mean()
    growth = fundamentals_df["Forward PE"].dropna().gt(25).mean()
    momentum = fundamentals_df["Beta"].dropna().gt(1).mean()

    return {
        "Value Tilt": round(value, 2),
        "Growth Tilt": round(growth, 2),
        "Momentum Tilt": round(momentum, 2),
    }
# ============================================================
# 16.18) SCHEDULED ALERTS (LOGIC)
# ============================================================

def evaluate_alerts(
    watchlist: List[str],
    price_targets: Dict[str, float]
) -> List[str]:
    triggered = []

    for t in watchlist:
        target = price_targets.get(t)
        if target and check_price_alert(t, target):
            triggered.append(f"{t} crossed ${target}")

    return triggered
# ============================================================
# 16.19) TAX & CAPITAL GAINS ESTIMATOR
# ============================================================

def estimate_capital_gains_tax(
    holdings_df: pd.DataFrame,
    tax_rate: float = 0.15
) -> float:
    if holdings_df.empty:
        return 0.0

    realized_gains = holdings_df.loc["TOTAL", "PnL"]
    return round(max(realized_gains, 0) * tax_rate, 2)
# ============================================================
# 16.20) GOAL-BASED PLANNING
# ============================================================

def goal_projection(
    current_value: float,
    annual_contribution: float,
    years: int,
    expected_return: float
) -> float:
    fv = current_value
    for _ in range(years):
        fv = fv * (1 + expected_return) + annual_contribution
    return round(fv, 2)
# ============================================================
# 16.21) ADVISOR LETTER (AI)
# ============================================================

def advisor_letter_ai(context: Dict[str, Any]) -> str:
    prompt = (
        "Write a professional wealth advisor letter summarizing "
        "portfolio performance, risks, and next steps. "
        "No investment advice.\n\n"
        + safe_json(context)
    )
    return ai(prompt)
with tabs[16]:
    st.subheader("📉 Backtesting")

    if st.session_state.get("portfolio") is None:
        st.info("Upload portfolio first.")
    else:
        bt = backtest_portfolio(st.session_state["portfolio"])
        st.line_chart(bt["Portfolio Value"])
with tabs[17]:
    st.subheader("🎯 Goal Planning")

    current = st.number_input("Current Portfolio Value", 0.0)
    contrib = st.number_input("Annual Contribution", 0.0)
    years = st.slider("Years", 1, 40, 10)
    ret = st.slider("Expected Return (%)", 1, 15, 7) / 100

    fv = goal_projection(current, contrib, years, ret)
    st.metric("Projected Value", f"${fv}")
with tabs[18]:
    st.subheader("🧾 Tax Estimator")

    if st.session_state.get("portfolio") is None:
        st.info("Upload holdings.")
    else:
        tax = estimate_capital_gains_tax(st.session_state["portfolio"])
        st.metric("Estimated Capital Gains Tax", f"${tax}")
with tabs[19]:
    st.subheader("📝 Advisor Letter")

    if not is_pro():
        pro_feature_block("Advisor Letter")
    else:
        if st.button("Generate Advisor Letter"):
            ctx = {
                "portfolio": st.session_state.get("portfolio"),
                "client": st.session_state.get("client"),
            }
            letter = advisor_letter_ai(ctx)
            st.text_area("Advisor Letter", letter, height=300)
