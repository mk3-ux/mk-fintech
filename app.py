# ============================================================
# KATTA WEALTH INSIGHTS — FULL APP (PART 1 / 6)
# Core setup, config, session, utilities
# ============================================================

from __future__ import annotations

import os
import io
import json
import uuid
import math
import time
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
    import yfinance as yf
except Exception:
    yf = None

try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from streamlit_cookies_manager import EncryptedCookieManager
except Exception:
    EncryptedCookieManager = None

# ============================================================
# 0) APP CONFIG  (THEME COMES FROM .streamlit/config.toml)
# ============================================================

APP_NAME = "Katta Wealth Insights"
APP_VERSION = "1.2.0"

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

st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 1) SESSION STATE
# ============================================================

def ss_init() -> None:
    st.session_state.setdefault("current_user", None)
    st.session_state.setdefault("tier", "Free")
    st.session_state.setdefault("ai_uses", 0)

    st.session_state.setdefault("scenario", None)
    st.session_state.setdefault("portfolio", None)
    st.session_state.setdefault("client", None)

    st.session_state.setdefault("alerts", [])
    st.session_state.setdefault("debug", False)

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
# 3) OPTIONAL COOKIES (SAFE FALLBACK)
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
    try:
        return cookies.get("user")
    except Exception:
        return None

def cookie_set_user(email: str) -> None:
    if not _COOKIES_OK:
        return
    try:
        cookies["user"] = email
        cookies.save()
    except Exception:
        pass

def cookie_clear_user() -> None:
    if not _COOKIES_OK:
        return
    try:
        cookies["user"] = ""
        cookies.save()
    except Exception:
        pass


# ============================================================
# 4) SQLITE (USERS, USAGE, BILLING, ARTIFACTS)
# ============================================================

def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def _db_init(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        pw_hash TEXT NOT NULL,
        tier TEXT NOT NULL DEFAULT 'Free',
        created_at TEXT NOT NULL
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS usage (
        email TEXT PRIMARY KEY,
        ai_uses INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS billing (
        email TEXT PRIMARY KEY,
        status TEXT NOT NULL DEFAULT 'unpaid',
        plan_name TEXT NOT NULL DEFAULT 'Free',
        updated_at TEXT NOT NULL
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS artifacts (
        id TEXT PRIMARY KEY,
        email TEXT NOT NULL,
        kind TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """)
    conn.commit()

def db_ready() -> bool:
    if not USE_SQLITE:
        return False
    try:
        conn = _db_connect()
        _db_init(conn)
        conn.close()
        return True
    except Exception:
        return False

DB_OK = db_ready()


# ============================================================
# 5) DB HELPERS
# ============================================================

def db_get_user(email: str) -> Optional[Dict[str, Any]]:
    if not DB_OK:
        return None
    conn = _db_connect()
    row = conn.execute(
        "SELECT email, pw_hash, tier FROM users WHERE email=?",
        (email,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"email": row[0], "pw": row[1], "tier": row[2]}

def db_create_user(email: str, pw_hash: str) -> bool:
    if not DB_OK:
        return False
    try:
        conn = _db_connect()
        conn.execute(
            "INSERT INTO users(email, pw_hash, tier, created_at) VALUES (?,?,?,?)",
            (email, pw_hash, "Free", now_iso()),
        )
        conn.execute(
            "INSERT INTO usage(email, ai_uses, updated_at) VALUES (?,?,?)",
            (email, 0, now_iso()),
        )
        conn.execute(
            "INSERT INTO billing(email, status, plan_name, updated_at) VALUES (?,?,?,?)",
            (email, "unpaid", "Free", now_iso()),
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def db_set_tier(email: str, tier: str) -> None:
    if not DB_OK:
        return
    conn = _db_connect()
    conn.execute("UPDATE users SET tier=? WHERE email=?", (tier, email))
    conn.commit()
    conn.close()

def db_get_usage(email: str) -> int:
    if not DB_OK:
        return st.session_state.ai_uses
    conn = _db_connect()
    row = conn.execute(
        "SELECT ai_uses FROM usage WHERE email=?",
        (email,),
    ).fetchone()
    conn.close()
    return int(row[0]) if row else 0

def db_set_usage(email: str, uses: int) -> None:
    if not DB_OK:
        st.session_state.ai_uses = uses
        return
    conn = _db_connect()
    conn.execute(
        "INSERT OR REPLACE INTO usage(email, ai_uses, updated_at) VALUES (?,?,?)",
        (email, uses, now_iso()),
    )
    conn.commit()
    conn.close()


# ============================================================
# 6) TIER / PRO LOGIC
# ============================================================

def effective_tier() -> str:
    if DEV_MODE:
        return "Pro"
    if not logged_in():
        return "Free"
    email = st.session_state.current_user
    if DB_OK:
        u = db_get_user(email)
        if u:
            return u.get("tier", "Free")
    return "Free"

def is_pro() -> bool:
    return effective_tier() == "Pro"


# ============================================================
# 7) AUTO LOGIN (COOKIE)
# ============================================================

def upgrade_current_user_to_pro():
    email = st.session_state.current_user
    if not email:
        return
    if DB_OK:
        conn = _db_connect()
        conn.execute("UPDATE users SET tier='Pro' WHERE email=?", (email,))
        conn.commit()
        conn.close()
    st.session_state.tier = "Pro"

def auto_login() -> None:
    if logged_in():
        return
    saved = cookie_get_user()
    if not saved:
        return
    if DB_OK:
        u = db_get_user(saved)
        if u:
            st.session_state.current_user = saved
            st.session_state.tier = u["tier"]
            st.session_state.ai_uses = db_get_usage(saved)

auto_login()


# ============================================================
# 8) AUTH UI (LOGIN / SIGNUP)
# ============================================================

def auth_ui() -> None:
    st.title(APP_NAME)
    st.caption(f"Version {APP_VERSION}")

    tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

    with tab_login:
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        remember = st.checkbox("Remember me", value=True)

        if st.button("Log In"):
            if not email or not pw:
                st.error("Missing email or password.")
                return
            pw_h = hash_pw(pw)
            if DB_OK:
                user = db_get_user(email)
                if user and user["pw"] == pw_h:
                    st.session_state.current_user = email
                    st.session_state.tier = user["tier"]
                    st.session_state.ai_uses = db_get_usage(email)
                    if remember:
                        cookie_set_user(email)
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

    with tab_signup:
        email = st.text_input("New Email")
        pw = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if not email or not pw:
                st.error("Missing email or password.")
                return
            ok = db_create_user(email, hash_pw(pw))
            if ok:
                st.success("Account created. Please log in.")
            else:
                st.error("Account already exists.")
# ============================================================
# 9) MARKET SCENARIO
# ============================================================

SECTORS = [
    "Technology",
    "Financials",
    "Healthcare",
    "Consumer",
    "Energy",
    "Real Estate",
    "Fixed Income",
]

def sector_impact(move: float, primary: str) -> pd.DataFrame:
    rows = []
    for s in SECTORS:
        impact = move if s == primary else move * 0.35
        rows.append({"Sector": s, "Score": round(impact, 2)})

    df = pd.DataFrame(rows)
    max_abs = df["Score"].abs().max()
    if max_abs > 0:
        df["Score"] = (df["Score"] / max_abs * 5).round(2)
    return df


# ============================================================
# 10) PORTFOLIO ANALYZER (SECTOR-BASED)
# ============================================================

REQUIRED_PORTFOLIO_COLUMNS = ["Sector", "Allocation"]

def validate_portfolio_df(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = [c for c in REQUIRED_PORTFOLIO_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"

    try:
        df["Allocation"] = pd.to_numeric(df["Allocation"])
    except Exception:
        return False, "Allocation must be numeric."

    if (df["Allocation"] < 0).any():
        return False, "Allocation must be non-negative."

    return True, "OK"


def portfolio_template_csv() -> bytes:
    sample = pd.DataFrame({
        "Sector": ["Technology", "Financials", "Fixed Income"],
        "Allocation": [40, 30, 30],
    })
    return sample.to_csv(index=False).encode("utf-8")


def diversification_and_hhi(port: pd.DataFrame) -> Tuple[float, float]:
    weights = port["Allocation"] / port["Allocation"].sum()
    hhi = float((weights ** 2).sum())
    diversification = 1 - hhi
    return round(diversification, 2), round(hhi, 2)


# ============================================================
# 11) LIVE STOCK DATA
# ============================================================

@st.cache_data(ttl=60)
def get_live_price(ticker: str) -> Dict[str, Any]:
    if yf is None:
        return {"price": None, "change": None}

    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="1m")
        if hist.empty:
            return {"price": None, "change": None}

        last = float(hist["Close"].iloc[-1])
        open_ = float(hist["Open"].iloc[0])

        return {
            "price": round(last, 2),
            "change": round(last - open_, 2),
        }
    except Exception:
        return {"price": None, "change": None}


# ============================================================
# 12) HOLDINGS-BASED PORTFOLIO TRACKER (PRO)
# ============================================================

REQUIRED_HOLDINGS_COLUMNS = ["Ticker", "Shares", "Cost_Basis"]

def validate_holdings_df(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = [c for c in REQUIRED_HOLDINGS_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"

    try:
        df["Shares"] = pd.to_numeric(df["Shares"])
        df["Cost_Basis"] = pd.to_numeric(df["Cost_Basis"])
    except Exception:
        return False, "Shares and Cost_Basis must be numeric."

    return True, "OK"


def compute_portfolio_holdings(df: pd.DataFrame) -> pd.DataFrame:
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

    out = pd.DataFrame(rows)
    out.loc["TOTAL", "Market Value"] = out["Market Value"].sum()
    out.loc["TOTAL", "PnL"] = out["PnL"].sum()
    return out


def holdings_template_csv() -> bytes:
    sample = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "NVDA"],
        "Shares": [10, 5, 3],
        "Cost_Basis": [150, 280, 400],
    })
    return sample.to_csv(index=False).encode("utf-8")


# ============================================================
# 13) STOCK RESEARCH — FUNDAMENTALS
# ============================================================

@st.cache_data(ttl=3600)
def get_stock_fundamentals(ticker: str) -> Dict[str, Any]:
    if yf is None:
        return {}

    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "Ticker": ticker,
            "Company": info.get("shortName"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Market Cap": info.get("marketCap"),
            "PE Ratio": info.get("trailingPE"),
            "Forward PE": info.get("forwardPE"),
            "Dividend Yield": info.get("dividendYield"),
            "Beta": info.get("beta"),
        }
    except Exception:
        return {}
# ============================================================
# 17) FEATURE REGISTRY (AUTHORITATIVE)
# ============================================================

FREE_PAGES = [
    "Market Scenario",
    "Portfolio Analyzer",
    "Live Stocks",
    "Stock Research",
]

PRO_PAGES = [
    "Portfolio Tracker",
    "Dividend Tracker",
    "Stock Screener",
    "Portfolio Health",
    "What-If Simulator",
    "AI Rebalancing",
    "Income Forecast",
    "Risk Alerts",
    "Teen Explainer",
    "Factor Exposure",
    "Goal Probability",
    "Market Commentary",
    "Tax Optimization",
]


def allowed_pages() -> List[str]:
    return FREE_PAGES + PRO_PAGES if is_pro() else FREE_PAGES


# ============================================================
# 18) SIDEBAR NAVIGATION (LEFT SIDE)
# ============================================================

def sidebar_nav() -> str:
    with st.sidebar:
        st.markdown("## 📂 Navigation")
        st.caption(f"User: {st.session_state.current_user}")
        st.caption(f"Plan: {'Pro' if is_pro() else 'Free'}")

    page = st.radio(
        "Go to",
        allowed_pages(),
        index=0,
        key="main_sidebar_nav",   # ← ADD THIS
    )


    st.markdown("---")

        if not is_pro():
            st.info("🔒 Upgrade to Pro to unlock advanced features")

        if st.button("Log out"):
            st.session_state.current_user = None
            st.session_state.tier = "Free"
            cookie_clear_user()
            st.rerun()

        if not is_pro():
            if st.button("💎 Upgrade to Pro (Demo)"):
                upgrade_current_user_to_pro()
                st.success("You are now Pro!")
                st.rerun()

        return page


# ============================================================
# 19) PAGE RENDERERS — FREE
# ============================================================

def render_market_scenario():
    st.header("Market Scenario")

    move = st.slider("Market move (%)", -20, 20, 0)
    sector = st.selectbox("Primary sector", SECTORS)

    df = sector_impact(move, sector)
    st.session_state["scenario"] = df

    st.dataframe(df, use_container_width=True)


def render_portfolio_analyzer():
    st.header("Portfolio Analyzer")

    st.download_button(
        "Download CSV Template",
        portfolio_template_csv(),
        file_name="portfolio_template.csv",
        mime="text/csv",
    )

    f = st.file_uploader("Upload Portfolio CSV", type="csv")
    if not f:
        return

    df = pd.read_csv(f)
    ok, msg = validate_portfolio_df(df)
    if not ok:
        st.error(msg)
        return

    st.session_state["portfolio"] = df
    st.dataframe(df, use_container_width=True)

    if is_pro():
        div, hhi = diversification_and_hhi(df)
        st.metric("Diversification Score", div)
        st.metric("Concentration Risk (HHI)", hhi)


def render_live_stocks():
    st.header("Live Stocks")

    tickers = st.text_input("Tickers (comma separated)", "AAPL,MSFT,NVDA")
    cols = st.columns(3)

    for i, t in enumerate([x.strip().upper() for x in tickers.split(",")]):
        data = get_live_price(t)
        cols[i % 3].metric(t, data["price"], data["change"])


def render_stock_research():
    st.header("Stock Research")

    ticker = st.text_input("Ticker", "AAPL").upper()
    data = get_stock_fundamentals(ticker)

    if data:
        st.json(data)


# ============================================================
# 20) PAGE RENDERERS — PRO
# ============================================================

def render_portfolio_tracker():
    st.header("Portfolio Tracker")

    st.download_button(
        "Download Holdings CSV Template",
        holdings_template_csv(),
        file_name="holdings_template.csv",
        mime="text/csv",
    )

    f = st.file_uploader("Upload Holdings CSV", type="csv")
    if not f:
        return

    df = pd.read_csv(f)
    ok, msg = validate_holdings_df(df)
    if not ok:
        st.error(msg)
        return

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


def render_dividend_tracker():
    st.header("Dividend Tracker")

    ticker = st.text_input("Dividend Ticker", "MSFT").upper()
    divs = get_dividend_history(ticker)

    if divs.empty:
        st.info("No dividend data found.")
        return

    st.dataframe(divs.tail(10), use_container_width=True)
    st.metric("Trailing 12M Dividend", f"${annual_dividend(divs)}")


def render_stock_screener():
    st.header("Stock Screener")

    universe = st.text_area(
        "Ticker Universe",
        "AAPL,MSFT,GOOGL,AMZN,META,NVDA"
    )

    max_pe = st.slider("Max PE", 5, 50, 25)
    min_div = st.slider("Min Dividend Yield", 0.0, 0.1, 0.0)

    df = screen_stocks(
        [x.strip().upper() for x in universe.split(",")],
        max_pe=max_pe,
        min_div_yield=min_div,
    )

    st.dataframe(df, use_container_width=True)


# ============================================================
# 21) MAIN ROUTER (ONLY ONE — CORRECT)
# ============================================================

def main():
    if not logged_in():
        auth_ui()
        return

    flush_alerts()
    page = sidebar_nav()

    # -------- FREE --------
    if page == "Market Scenario":
        render_market_scenario()

    elif page == "Portfolio Analyzer":
        render_portfolio_analyzer()

    elif page == "Live Stocks":
        render_live_stocks()

    elif page == "Stock Research":
        render_stock_research()

    # -------- PRO --------
    elif is_pro() and page == "Portfolio Tracker":
        render_portfolio_tracker()

    elif is_pro() and page == "Dividend Tracker":
        render_dividend_tracker()

    elif is_pro() and page == "Stock Screener":
        render_stock_screener()

    elif is_pro() and page == "Portfolio Health":
        render_portfolio_health_ai()

    elif is_pro() and page == "What-If Simulator":
        render_what_if_ai()

    elif is_pro() and page == "AI Rebalancing":
        render_ai_rebalancing()

    elif is_pro() and page == "Income Forecast":
        render_income_forecast_ai()

    elif is_pro() and page == "Risk Alerts":
        render_risk_alerts_ai()

    elif is_pro() and page == "Teen Explainer":
        render_teen_explainer_ai()

    elif is_pro() and page == "Factor Exposure":
        render_factor_exposure_ai()

    elif is_pro() and page == "Goal Probability":
        render_goal_probability_ai()

    elif is_pro() and page == "Market Commentary":
        render_market_commentary_ai()

    elif is_pro() and page == "Tax Optimization":
        render_tax_optimization_ai()

===================================================
# 22) PERFORMANCE & RISK
# ============================================================

@st.cache_data(ttl=3600)
def get_price_history(ticker: str, period: str = "1y") -> pd.Series:
    if yf is None:
        return pd.Series(dtype=float)
    try:
        return yf.Ticker(ticker).history(period=period)["Close"]
    except Exception:
        return pd.Series(dtype=float)


def compute_returns(prices: pd.Series) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    return prices.pct_change().dropna()


def portfolio_volatility(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return round(float(returns.std() * np.sqrt(252)), 4)


def max_drawdown(prices: pd.Series) -> float:
    if prices.empty:
        return 0.0
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    return round(float(drawdown.min()), 4)


def render_performance_risk():
    st.header("Performance & Risk")

    if st.session_state.get("portfolio") is None:
        st.info("Upload a portfolio first.")
        return

    holdings = st.session_state["portfolio"]

    tickers = holdings["Ticker"].dropna().unique()
    returns_list = []

    for t in tickers:
        prices = get_price_history(t)
        returns = compute_returns(prices)
        returns_list.append(returns)

    if not returns_list:
        st.info("Not enough data.")
        return

    portfolio_returns = pd.concat(returns_list, axis=1).mean(axis=1)

    st.metric("Volatility (Annualized)", portfolio_volatility(portfolio_returns))
    st.metric(
        "Max Drawdown",
        max_drawdown((1 + portfolio_returns).cumprod())
    )


# ============================================================
# 23) BACKTESTING
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


def render_backtesting():
    st.header("Backtesting")

    if st.session_state.get("portfolio") is None:
        st.info("Upload portfolio first.")
        return

    bt = backtest_portfolio(st.session_state["portfolio"])
    st.line_chart(bt["Portfolio Value"])


# ============================================================
# 24) WATCHLIST
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
        "INSERT OR REPLACE INTO artifacts "
        "(id, email, kind, payload_json, created_at) "
        "VALUES (?,?,?,?,?)",
        (str(uuid.uuid4()), email, "watchlist", json.dumps(tickers), now_iso()),
    )
    conn.commit()
    conn.close()


def render_watchlist():
    st.header("Watchlist")

    email = st.session_state.current_user
    watchlist = get_watchlist(email)

    new = st.text_input("Add ticker")
    if st.button("Add"):
        watchlist.append(new.upper())
        save_watchlist(email, list(set(watchlist)))
        st.rerun()

    for t in watchlist:
        p = get_live_price(t)
        st.metric(t, p["price"], p["change"])


# ============================================================
# 25) GOAL PLANNING
# ============================================================

def goal_projection(
    current: float,
    annual: float,
    years: int,
    expected_return: float,
) -> float:
    fv = current
    for _ in range(years):
        fv = fv * (1 + expected_return) + annual
    return round(fv, 2)


def render_goals():
    st.header("Goal Planning")

    current = st.number_input("Current Portfolio Value", 0.0)
    contrib = st.number_input("Annual Contribution", 0.0)
    years = st.slider("Years", 1, 40, 10)
    ret = st.slider("Expected Return (%)", 1, 15, 7) / 100

    fv = goal_projection(current, contrib, years, ret)
    st.metric("Projected Value", f"${fv}")


# ============================================================
# 26) TAX ESTIMATOR
# ============================================================

def estimate_capital_gains_tax(
    holdings_df: pd.DataFrame,
    tax_rate: float = 0.15,
) -> float:
    if holdings_df.empty:
        return 0.0
    pnl = holdings_df.loc["TOTAL", "PnL"]
    return round(max(pnl, 0) * tax_rate, 2)


def render_taxes():
    st.header("Tax Estimator")

    if st.session_state.get("portfolio") is None:
        st.info("Upload holdings.")
        return

    tax = estimate_capital_gains_tax(st.session_state["portfolio"])
    st.metric("Estimated Capital Gains Tax", f"${tax}")
# ============================================================
# 27) AI CORE — GROQ CLIENT
# ============================================================

_groq_client = None


def ai(prompt: str) -> str:
    """
    Central AI helper for all AI-powered features.
    Pro-only, educational, no investment advice.
    """
    global _groq_client

    if not is_pro():
        return "🔒 This AI feature is available in Pro."

    if Groq is None or not GROQ_API_KEY:
        return "⚠️ AI is not configured. Missing GROQ_API_KEY."

    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)

    try:
        resp = _groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial education AI. "
                        "Do not provide investment advice. "
                        "Explain concepts clearly and responsibly."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.4,
            max_tokens=800,
        )

        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"AI error: {str(e)}"
# ============================================================
# 28) PORTFOLIO HEALTH (AI)
# ============================================================

def render_portfolio_health_ai():
    st.header("🧠 Portfolio Health Score")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    total_value = portfolio.loc["TOTAL", "Market Value"]
    total_pnl = portfolio.loc["TOTAL", "PnL"]

    score = max(
        30,
        min(
            95,
            int(100 - abs(total_pnl) / max(total_value, 1) * 100),
        ),
    )

    st.metric("Health Score", f"{score} / 100")

    st.markdown(
        ai(
            f"Explain what a portfolio health score of {score} means "
            f"for a long-term investor."
        )
    )


# ============================================================
# 29) WHAT-IF SIMULATOR (AI)
# ============================================================

def render_what_if_ai():
    st.header("📉 What-If Scenario Simulator")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    shock = st.slider("Market shock (%)", -50, 10, -20)
    impact = round(
        portfolio.loc["TOTAL", "Market Value"] * shock / 100,
        2,
    )

    st.metric("Estimated Impact", f"${impact}")

    st.markdown(
        ai(
            f"If markets fall {shock}%, explain what typically happens "
            f"to diversified portfolios."
        )
    )


# ============================================================
# 30) AI REBALANCING
# ============================================================

def render_ai_rebalancing():
    st.header("⚖️ AI Rebalancing Suggestions")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    ctx = build_portfolio_insights_context(portfolio)

    st.markdown(
        ai(
            "Analyze this portfolio and suggest educational "
            "rebalancing ideas (no advice):\n\n"
            + safe_json(ctx)
        )
    )


# ============================================================
# 31) INCOME FORECAST (AI)
# ============================================================

def render_income_forecast_ai():
    st.header("💵 Income Forecast")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    income = round(
        portfolio.loc["TOTAL", "Market Value"] * 0.025,
        2,
    )

    st.metric("Estimated Annual Income", f"${income}")

    st.markdown(
        ai(
            f"Explain dividend income investing using an estimated "
            f"annual income of ${income}."
        )
    )


# ============================================================
# 32) RISK ALERTS (AI)
# ============================================================

def render_risk_alerts_ai():
    st.header("🚨 Risk Alerts (Educational)")

    drawdown = st.slider(
        "Alert if drawdown exceeds (%)",
        5,
        40,
        15,
    )

    st.markdown(
        ai(
            f"Explain why monitoring a {drawdown}% drawdown "
            f"is important for risk management."
        )
    )


# ============================================================
# 33) TEEN MODE EXPLAINER (AI)
# ============================================================

def render_teen_explainer_ai():
    st.header("🎓 Explain My Portfolio (Teen Mode)")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    ctx = build_portfolio_insights_context(portfolio)

    st.markdown(
        ai(
            "Explain this portfolio to a high-school student "
            "who is interested in finance:\n\n"
            + safe_json(ctx)
        )
    )


# ============================================================
# 34) FACTOR EXPOSURE (AI)
# ============================================================

def render_factor_exposure_ai():
    st.header("📊 Factor Exposure (AI)")

    st.markdown(
        ai(
            "Explain value, growth, and momentum exposure "
            "in simple terms for a beginner investor."
        )
    )


# ============================================================
# 35) GOAL PROBABILITY (AI)
# ============================================================

def render_goal_probability_ai():
    st.header("🎯 Goal Probability (AI)")

    st.markdown(
        ai(
            "Explain how investors estimate the probability of "
            "reaching long-term financial goals."
        )
    )


# ============================================================
# 36) MARKET COMMENTARY (AI)
# ============================================================

def render_market_commentary_ai():
    st.header("📰 AI Market Commentary")

    st.markdown(
        ai(
            "Write a short daily market commentary for "
            "retail investors."
        )
    )


# ============================================================
# 37) TAX OPTIMIZATION (AI)
# ============================================================

def render_tax_optimization_ai():
    st.header("🧾 AI Tax Optimization (Educational)")

    st.markdown(
        ai(
            "Explain tax-efficient investing strategies "
            "at a high level without giving advice."
        )
    )
# ============================================================
# 38) OPTIONAL DEBUG PANEL (DEV ONLY)
# ============================================================

def render_debug_panel():
    if not DEV_MODE:
        return

    with st.expander("🛠 Debug Panel"):
        st.write("Session State:")
        st.json({k: str(v) for k, v in st.session_state.items()})


# ============================================================
# 39) FINAL SAFETY CHECKS
# ============================================================

def ensure_portfolio_totals():
    """
    Ensures TOTAL row exists for portfolio-based features.
    Prevents runtime errors in AI & analytics layers.
    """
    port = st.session_state.get("portfolio")
    if port is None or port.empty:
        return

    if "TOTAL" not in port.index and "Market Value" in port.columns:
        port.loc["TOTAL", "Market Value"] = port["Market Value"].sum()
        if "PnL" in port.columns:
            port.loc["TOTAL", "PnL"] = port["PnL"].sum()


# ============================================================
# 40) APP ENTRYPOINT GUARD
# ============================================================

def run_app():
    """
    Single safe entrypoint.
    Ensures no duplicate execution and clean startup.
    """
    ensure_portfolio_totals()
    main()
    render_debug_panel()


# ============================================================
# 41) EXECUTION
# ============================================================

if __name__ == "__main__":
    run_app()
