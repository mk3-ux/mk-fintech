# ============================================================
# KATTA WEALTH INSIGHTS — FULL APP
# PART 1 / 8 — CORE SETUP, CONFIG, SESSION, UTILITIES
# ============================================================

from __future__ import annotations

import os
import json
import uuid
import hashlib
import sqlite3
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

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
# APP CONFIG
# ============================================================

APP_NAME = "Katta Wealth Insights"
APP_VERSION = "1.2.0"

DEV_MODE = False
MODEL_NAME = "llama-3.1-8b-instant"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

USE_SQLITE = True
SQLITE_PATH = "kwi_app.db"

USE_COOKIES = True
COOKIE_PASSWORD = os.getenv("KWI_COOKIE_PASSWORD", "change-this-secret")

st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE
# ============================================================

# ============================================================
# DIVIDEND DATA HELPERS (REQUIRED)
# ============================================================
# ============================================================
# DIVIDEND DATA HELPERS — FIXED
# ============================================================

@st.cache_data(ttl=3600)
def get_dividend_history(ticker: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()

    try:
        divs = yf.Ticker(ticker).dividends

        if divs is None or divs.empty:
            return pd.DataFrame()

        # ✅ df is CREATED FIRST
        df = divs.reset_index()
        df.columns = ["Date", "Dividend"]

        # ✅ THEN Date conversion
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Safety: remove invalid dates
        df = df.dropna(subset=["Date"])

        return df

    except Exception:
        return pd.DataFrame()



def annual_dividend(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0

    # ✅ FIX: use timezone-safe now
    # ✅ ALWAYS use pandas Timestamp
    cutoff = pd.Timestamp.now().tz_localize(None) - pd.DateOffset(years=1)



    last_year = df[df["Date"] >= cutoff]

    return round(float(last_year["Dividend"].sum()), 2)

def init_session() -> None:
    defaults = {
        "current_user": None,
        "tier": "Free",
        "ai_uses": 0,
        "scenario": None,
        "portfolio": None,
        "alerts": [],
        "debug": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_session()

# ============================================================
# UTILITIES
# ============================================================

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def hash_pw(password: str) -> str:
    return sha256("kwi_salt_" + password)

def logged_in() -> bool:
    return st.session_state.current_user is not None

def push_alert(msg: str) -> None:
    st.session_state.alerts.append(msg)

def flush_alerts() -> None:
    for msg in st.session_state.alerts[-5:]:
        st.info(msg)
    st.session_state.alerts.clear()

def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)
# ============================================================
# PART 2 / 8 — COOKIES, SQLITE, AUTH, TIER LOGIC
# ============================================================

# ============================================================
# COOKIES (OPTIONAL, SAFE FALLBACK)
# ============================================================

cookies = None

def cookies_ready() -> bool:
    global cookies
    if not USE_COOKIES or EncryptedCookieManager is None:
        return False
    cookies = EncryptedCookieManager(
        prefix="kwi_",
        password=COOKIE_PASSWORD
    )
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
# SQLITE — DATABASE SETUP
# ============================================================

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def db_init() -> bool:
    if not USE_SQLITE:
        return False
    try:
        conn = db_connect()

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
        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)

        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

DB_OK = db_init()


# ============================================================
# DATABASE HELPERS
# ============================================================

def db_get_user(email: str) -> Optional[Dict[str, Any]]:
    if not DB_OK:
        return None
    conn = db_connect()
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
        conn = db_connect()
        conn.execute(
            "INSERT INTO users VALUES (?,?,?,?)",
            (email, pw_hash, "Free", now_iso())
        )
        conn.execute(
            "INSERT INTO usage VALUES (?,?,?)",
            (email, 0, now_iso())
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def db_get_usage(email: str) -> int:
    if not DB_OK:
        return st.session_state.ai_uses
    conn = db_connect()
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
    conn = db_connect()
    conn.execute(
        "INSERT OR REPLACE INTO usage VALUES (?,?,?)",
        (email, uses, now_iso())
    )
    conn.commit()
    conn.close()
def upgrade_current_user_to_pro():
    """
    Upgrades the currently logged-in user to Pro.
    Demo-only (no Stripe).
    """
    email = st.session_state.current_user
    if not email:
        return

    # Update database
    if DB_OK:
        db_set_tier(email, "Pro")

    # Update session immediately
    st.session_state.tier = "Pro"


def db_set_tier(email: str, tier: str) -> None:
    if not DB_OK:
        return
    conn = db_connect()
    conn.execute(
        "UPDATE users SET tier=? WHERE email=?",
        (tier, email)
    )
    conn.commit()
    conn.close()


# ============================================================
# TIER LOGIC (AUTHORITATIVE)
# ============================================================

def effective_tier() -> str:
    if DEV_MODE:
        return "Pro"
    if not logged_in():
        return "Free"
    user = db_get_user(st.session_state.current_user)
    return user["tier"] if user else "Free"

def is_pro() -> bool:
    return effective_tier() == "Pro"


# ============================================================
# AUTO LOGIN (COOKIE)
# ============================================================

def auto_login() -> None:
    if logged_in():
        return
    saved = cookie_get_user()
    if not saved:
        return
    user = db_get_user(saved)
    if user:
        st.session_state.current_user = saved
        st.session_state.tier = user["tier"]
        st.session_state.ai_uses = db_get_usage(saved)

auto_login()


# ============================================================
# AUTH UI (LOGIN / SIGNUP)
# ============================================================

def auth_ui() -> None:
    st.title(APP_NAME)
    st.caption(f"Version {APP_VERSION}")

    login_tab, signup_tab = st.tabs(["Log In", "Sign Up"])

    with login_tab:
        email = st.text_input("Email", key="login_email")
        pw = st.text_input("Password", type="password", key="login_pw")
        remember = st.checkbox("Remember me", value=True)

        if st.button("Log In"):
            if not email or not pw:
                st.error("Missing email or password.")
                return

            user = db_get_user(email)
            if user and user["pw"] == hash_pw(pw):
                st.session_state.current_user = email
                st.session_state.tier = user["tier"]
                st.session_state.ai_uses = db_get_usage(email)
                if remember:
                    cookie_set_user(email)
                st.rerun()
            else:
                st.error("Invalid credentials.")

    with signup_tab:
        email = st.text_input("New Email", key="signup_email")
        pw = st.text_input("New Password", type="password", key="signup_pw")

        if st.button("Create Account"):
            if not email or not pw:
                st.error("Missing email or password.")
                return

            if db_create_user(email, hash_pw(pw)):
                st.success("Account created. Please log in.")
            else:
                st.error("Account already exists.")
# ============================================================
# PART 3 / 8 — CORE FINANCE LOGIC
# Market Scenarios, Portfolio Analysis, Live Stocks
# ============================================================

# ============================================================
# MARKET SCENARIO ENGINE
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

def sector_impact(move: float, primary_sector: str) -> pd.DataFrame:
    """
    Computes relative sector impact given a market move.
    Normalized to a -5 to +5 scale.
    """
    rows = []

    for sector in SECTORS:
        impact = move if sector == primary_sector else move * 0.35
        rows.append({"Sector": sector, "Score": round(impact, 2)})

    df = pd.DataFrame(rows)

    max_abs = df["Score"].abs().max()
    if max_abs > 0:
        df["Score"] = (df["Score"] / max_abs * 5).round(2)

    return df


# ============================================================
# PORTFOLIO ANALYZER (SECTOR-BASED)
# ============================================================

REQUIRED_PORTFOLIO_COLUMNS = ["Sector", "Allocation"]

def validate_portfolio_df(df: pd.DataFrame) -> Tuple[bool, str]:
    if df.empty:
        return False, "Portfolio is empty."

    missing = [c for c in REQUIRED_PORTFOLIO_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"

    try:
        df["Allocation"] = pd.to_numeric(df["Allocation"])
    except Exception:
        return False, "Allocation must be numeric."

    if (df["Allocation"] < 0).any():
        return False, "Allocation must be non-negative."

    total = df["Allocation"].sum()
    if total <= 0:
        return False, "Total allocation must be greater than zero."

    return True, "OK"

def portfolio_template_csv() -> bytes:
    sample = pd.DataFrame({
        "Sector": ["Technology", "Financials", "Fixed Income"],
        "Allocation": [40, 30, 30],
    })
    return sample.to_csv(index=False).encode("utf-8")

def diversification_and_hhi(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Returns diversification score and HHI concentration metric.
    """
    weights = df["Allocation"] / df["Allocation"].sum()
    hhi = float((weights ** 2).sum())
    diversification = 1 - hhi
    return round(diversification, 2), round(hhi, 2)


# ============================================================
# LIVE STOCK DATA (YFINANCE)
# ============================================================

@st.cache_data(ttl=60)
def get_live_price(ticker: str) -> Dict[str, Optional[float]]:
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
# HOLDINGS-BASED PORTFOLIO TRACKER (PRO)
# ============================================================

REQUIRED_HOLDINGS_COLUMNS = ["Ticker", "Shares", "Cost_Basis"]

def validate_holdings_df(df: pd.DataFrame) -> Tuple[bool, str]:
    if df.empty:
        return False, "Holdings file is empty."

    missing = [c for c in REQUIRED_HOLDINGS_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"

    try:
        df["Shares"] = pd.to_numeric(df["Shares"])
        df["Cost_Basis"] = pd.to_numeric(df["Cost_Basis"])
    except Exception:
        return False, "Shares and Cost_Basis must be numeric."

    if (df["Shares"] <= 0).any():
        return False, "Shares must be greater than zero."

    return True, "OK"

def holdings_template_csv() -> bytes:
    sample = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "NVDA"],
        "Shares": [10, 5, 3],
        "Cost_Basis": [150, 280, 400],
    })
    return sample.to_csv(index=False).encode("utf-8")

def compute_portfolio_holdings(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).upper().strip()
        live = get_live_price(ticker)

        if live["price"] is None:
            continue

        shares = float(row["Shares"])
        cost = float(row["Cost_Basis"])
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


# ============================================================
# STOCK RESEARCH — FUNDAMENTALS
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
# PART 4 / 8 — FEATURE REGISTRY & SIDEBAR NAVIGATION
# ============================================================

# ============================================================
# FEATURE REGISTRY (AUTHORITATIVE SOURCE OF TRUTH)
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
    "AI Chatbot",
]

def allowed_pages() -> List[str]:
    """
    Returns pages user is allowed to see based on tier.
    Single source of truth for navigation.
    """
    if is_pro():
        return FREE_PAGES + PRO_PAGES
    return FREE_PAGES


# ============================================================
# SIDEBAR NAVIGATION (LEFT PANEL)
# ============================================================

def sidebar_nav() -> str:
    """
    Renders sidebar navigation and returns selected page.
    This function MUST be called exactly once.
    """
    with st.sidebar:
        st.markdown("## 📂 Navigation")

        if logged_in():
            st.caption(f"User: {st.session_state.current_user}")
            st.caption(f"Plan: {'Pro' if is_pro() else 'Free'}")
        else:
            st.caption("Not logged in")

        page = st.radio(
            "Go to",
            allowed_pages(),
            index=0,
            key="nav_radio",
        )

        st.markdown("---")

        # Upgrade CTA
        if not is_pro():
            st.info("🔒 Upgrade to Pro to unlock AI & advanced tools")

            if st.button("💎 Upgrade to Pro (Demo)"):
                upgrade_current_user_to_pro()
                st.success("You are now Pro!")
                st.rerun()

        # Logout
        if logged_in():
            if st.button("Log out"):
                st.session_state.current_user = None
                st.session_state.tier = "Free"
                cookie_clear_user()
                st.rerun()

        return page
# ============================================================
# PART 5 / 8 — PAGE RENDERERS (FREE + PRO, NON-AI)
# ============================================================

# ============================================================
# FREE FEATURES
# ============================================================

def render_market_scenario():
    st.header("Market Scenario")

    move = st.slider("Market move (%)", -20, 20, 0)
    sector = st.selectbox("Primary sector", SECTORS)

    df = sector_impact(move, sector)
    st.session_state.scenario = df

    st.dataframe(df, use_container_width=True)


def render_portfolio_analyzer():
    st.header("Portfolio Analyzer")

    st.download_button(
        "Download CSV Template",
        portfolio_template_csv(),
        file_name="portfolio_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload Portfolio CSV", type="csv")
    if not uploaded:
        return

    df = pd.read_csv(uploaded)
    ok, msg = validate_portfolio_df(df)
    if not ok:
        st.error(msg)
        return

    st.session_state.portfolio = df
    st.dataframe(df, use_container_width=True)

    if is_pro():
        div, hhi = diversification_and_hhi(df)
        st.metric("Diversification Score", div)
        st.metric("Concentration Risk (HHI)", hhi)


def render_live_stocks():
    st.header("Live Stocks")

    tickers = st.text_input("Tickers (comma separated)", "AAPL,MSFT,NVDA")
    cols = st.columns(3)

    for i, ticker in enumerate([t.strip().upper() for t in tickers.split(",") if t]):
        data = get_live_price(ticker)
        cols[i % 3].metric(
            ticker,
            data["price"] if data["price"] is not None else "—",
            data["change"],
        )


def render_stock_research():
    st.header("Stock Research")

    ticker = st.text_input("Ticker", "AAPL").upper()
    data = get_stock_fundamentals(ticker)

    if not data:
        st.info("No data available.")
        return

    st.json(data)


# ============================================================
# PRO FEATURES (NON-AI)
# ============================================================

def render_portfolio_tracker():
    st.header("Portfolio Tracker")

    st.download_button(
        "Download Holdings Template",
        holdings_template_csv(),
        file_name="holdings_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload Holdings CSV", type="csv")
    if not uploaded:
        return

    df = pd.read_csv(uploaded)
    ok, msg = validate_holdings_df(df)
    if not ok:
        st.error(msg)
        return

    holdings = compute_portfolio_holdings(df)
    if holdings.empty:
        st.info("No valid holdings.")
        return

    st.session_state.portfolio = holdings
    st.dataframe(holdings, use_container_width=True)

    if "TOTAL" in holdings.index:
        st.metric(
            "Total Portfolio Value",
            f"${round(holdings.loc['TOTAL', 'Market Value'], 2)}",
        )
        st.metric(
            "Total P&L",
            f"${round(holdings.loc['TOTAL', 'PnL'], 2)}",
        )

def render_dividend_tracker():
    st.header("Dividend Tracker")

    ticker = st.text_input("Dividend Ticker", "MSFT").upper().strip()

    # --------------------
    # 📊 DATA SECTION
    # --------------------
    df = get_dividend_history(ticker)

    if df.empty:
        st.info("No dividend data available.")
        return

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["Date"])

    cutoff = pd.Timestamp.now().tz_localize(None) - pd.DateOffset(years=1)

    last_year = df[df["Date"] >= cutoff]

    # --------------------
    # 🖥️ UI SECTION
    # --------------------
    st.subheader("Recent Dividends")
    st.dataframe(df.tail(12), use_container_width=True)

    total = float(last_year["Dividend"].sum()) if not last_year.empty else 0.0
    st.metric("Trailing 12-Month Dividend", f"${round(total, 2)}")

def render_stock_screener():
    st.header("Stock Screener")

    universe = st.text_area(
        "Ticker Universe",
        "AAPL,MSFT,GOOGL,AMZN,META,NVDA",
    )

    max_pe = st.slider("Max PE", 5, 50, 25)
    min_div = st.slider("Min Dividend Yield", 0.0, 0.1, 0.0)

    rows = []
    for t in [x.strip().upper() for x in universe.split(",") if x]:
        f = get_stock_fundamentals(t)
        if not f:
            continue

        pe = f.get("PE Ratio")
        dy = f.get("Dividend Yield") or 0

        if pe is not None and pe <= max_pe and dy >= min_div:
            rows.append(f)

    if not rows:
        st.info("No matches found.")
        return

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
# ============================================================
# PART 6 / 8 — PERFORMANCE, BACKTESTING, WATCHLIST, GOALS, TAXES
# ============================================================

# ============================================================
# PERFORMANCE & RISK
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

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    tickers = portfolio.get("Ticker")
    if tickers is None:
        st.info("Performance requires ticker-based holdings.")
        return

    returns_list = []
    for t in tickers.dropna().unique():
        prices = get_price_history(t)
        returns = compute_returns(prices)
        returns_list.append(returns)

    if not returns_list:
        st.info("Not enough data.")
        return

    port_returns = pd.concat(returns_list, axis=1).mean(axis=1)

    st.metric("Volatility (Annualized)", portfolio_volatility(port_returns))
    st.metric(
        "Max Drawdown",
        max_drawdown((1 + port_returns).cumprod()),
    )


# ============================================================
# BACKTESTING
# ============================================================

def backtest_portfolio(
    holdings_df: pd.DataFrame,
    period: str = "3y",
) -> pd.DataFrame:
    if yf is None or holdings_df.empty:
        return pd.DataFrame()

    prices = {}
    for _, row in holdings_df.drop(index="TOTAL", errors="ignore").iterrows():
        try:
            hist = yf.Ticker(row["Ticker"]).history(period=period)["Close"]
            prices[row["Ticker"]] = hist * row["Shares"]
        except Exception:
            continue

    if not prices:
        return pd.DataFrame()

    df = pd.DataFrame(prices).dropna()
    df["Portfolio Value"] = df.sum(axis=1)
    df["Returns"] = df["Portfolio Value"].pct_change()
    return df.dropna()

def render_backtesting():
    st.header("Backtesting")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload holdings first.")
        return

    bt = backtest_portfolio(portfolio)
    if bt.empty:
        st.info("Backtest unavailable.")
        return

    st.line_chart(bt["Portfolio Value"])


# ============================================================
# WATCHLIST
# ============================================================

def get_watchlist(email: str) -> List[str]:
    if not DB_OK:
        return st.session_state.setdefault("watchlist", [])

    conn = db_connect()
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

    conn = db_connect()
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
    if st.button("Add to Watchlist") and new:
        watchlist.append(new.upper())
        save_watchlist(email, sorted(set(watchlist)))
        st.rerun()

    if not watchlist:
        st.info("No tickers yet.")
        return

    for t in watchlist:
        p = get_live_price(t)
        st.metric(t, p["price"], p["change"])


# ============================================================
# GOAL PLANNING
# ============================================================

def goal_projection(
    current: float,
    annual: float,
    years: int,
    expected_return: float,
) -> float:
    value = current
    for _ in range(years):
        value = value * (1 + expected_return) + annual
    return round(value, 2)

def render_goals():
    st.header("Goal Planning")

    current = st.number_input("Current Portfolio Value", min_value=0.0)
    contrib = st.number_input("Annual Contribution", min_value=0.0)
    years = st.slider("Years", 1, 40, 10)
    ret = st.slider("Expected Return (%)", 1, 15, 7) / 100

    fv = goal_projection(current, contrib, years, ret)
    st.metric("Projected Value", f"${fv}")


# ============================================================
# TAX ESTIMATOR
# ============================================================

def estimate_capital_gains_tax(
    holdings_df: pd.DataFrame,
    tax_rate: float = 0.15,
) -> float:
    if holdings_df.empty or "PnL" not in holdings_df.columns:
        return 0.0

    pnl = holdings_df.loc["TOTAL", "PnL"] if "TOTAL" in holdings_df.index else 0.0
    return round(max(pnl, 0) * tax_rate, 2)

def render_taxes():
    st.header("Tax Estimator")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload holdings first.")
        return

    tax = estimate_capital_gains_tax(portfolio)
    st.metric("Estimated Capital Gains Tax", f"${tax}")
# ============================================================
# PART 7 / 8 — AI CORE + PRO AI FEATURES
# ============================================================

# ============================================================
# AI CORE (GROQ CLIENT)
# ============================================================

_groq_client = None

def ai(prompt: str) -> str:
    """
    Central AI helper.
    - Pro-only
    - Educational only (no investment advice)
    """
    global _groq_client

    if not is_pro():
        return "🔒 This AI feature is available in Pro."

    if Groq is None or not GROQ_API_KEY:
        return "⚠️ AI is not configured. Missing GROQ_API_KEY."

    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)

    try:
        response = _groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial education assistant. "
                        "Do not give investment advice. "
                        "Explain concepts clearly for learning purposes."
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

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"AI error: {str(e)}"


# ============================================================
# AI FEATURE HELPERS
# ============================================================

def build_portfolio_insights_context(portfolio: pd.DataFrame) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}

    if portfolio is None or portfolio.empty:
        return ctx

    if "TOTAL" in portfolio.index:
        ctx["total_value"] = float(portfolio.loc["TOTAL", "Market Value"])
        ctx["total_pnl"] = float(portfolio.loc["TOTAL", "PnL"])

    positions = []
    for _, row in portfolio.drop(index="TOTAL", errors="ignore").iterrows():
        positions.append({
            "ticker": row.get("Ticker"),
            "market_value": row.get("Market Value"),
            "pnl": row.get("PnL"),
        })

    ctx["positions"] = positions
    return ctx


# ============================================================
# AI FEATURES (PRO ONLY)
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
        min(95, int(100 - abs(total_pnl) / max(total_value, 1) * 100)),
    )

    st.metric("Health Score", f"{score} / 100")

    st.markdown(
        ai(
            f"Explain what a portfolio health score of {score} means "
            f"for a long-term investor."
        )
    )


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
            f"If markets fall {shock}%, explain what typically "
            f"happens to diversified portfolios."
        )
    )


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


def render_risk_alerts_ai():
    st.header("🚨 Risk Alerts")

    drawdown = st.slider("Alert if drawdown exceeds (%)", 5, 40, 15)

    st.markdown(
        ai(
            f"Explain why monitoring a {drawdown}% drawdown "
            f"is important for portfolio risk management."
        )
    )


def render_teen_explainer_ai():
    st.header("🎓 Teen Mode — Explain My Portfolio")

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


def render_factor_exposure_ai():
    st.header("📊 Factor Exposure")

    st.markdown(
        ai(
            "Explain value, growth, momentum, and quality factors "
            "in simple terms for beginner investors."
        )
    )


def render_goal_probability_ai():
    st.header("🎯 Goal Probability")

    st.markdown(
        ai(
            "Explain how investors estimate the probability "
            "of reaching long-term financial goals."
        )
    )


def render_market_commentary_ai():
    st.header("📰 Market Commentary")

    st.markdown(
        ai(
            "Write a short daily market commentary for "
            "retail investors."
        )
    )


def render_tax_optimization_ai():
    st.header("🧾 Tax Optimization")

    st.markdown(
        ai(
            "Explain tax-efficient investing strategies "
            "at a high level without giving advice."
        )
    )
# ============================================================
# AI CHATBOT (PRO)
# ============================================================

def render_ai_chatbot():
    st.header("💬 AI Chatbot")

    st.caption(
        "Ask questions about investing, portfolios, markets, or concepts. "
        "Educational only — no investment advice."
    )

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**AI:** {msg['content']}")

    # User input
    user_input = st.text_input("Type your question", key="chat_input")

    if st.button("Send") and user_input:
        # Save user message
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        # Build context-aware prompt
        portfolio = st.session_state.get("portfolio")
        context = ""

        if portfolio is not None and not portfolio.empty:
            context = (
                "User portfolio context:\n"
                + safe_json(build_portfolio_insights_context(portfolio))
                + "\n\n"
            )

        ai_response = ai(
            context
            + "User question:\n"
            + user_input
        )

        # Save AI response
        st.session_state.chat_history.append(
            {"role": "assistant", "content": ai_response}
        )

        st.rerun()

# ============================================================
# PART 8 / 8 — MAIN ROUTER & APP ENTRYPOINT
# ============================================================

def main_router(page: str) -> None:
    """
    Central page router.
    ALL rendering flows through this function.
    """

    # ---------------- FREE FEATURES ----------------
    if page == "Market Scenario":
        render_market_scenario()

    elif page == "Portfolio Analyzer":
        render_portfolio_analyzer()

    elif page == "Live Stocks":
        render_live_stocks()

    elif page == "Stock Research":
        render_stock_research()

    elif page == "AI Chatbot":
        render_ai_chatbot()


    # ---------------- PRO FEATURES (NON-AI) ----------------
    elif is_pro() and page == "Portfolio Tracker":
        render_portfolio_tracker()

    elif is_pro() and page == "Dividend Tracker":
        render_dividend_tracker()

    elif is_pro() and page == "Stock Screener":
        render_stock_screener()

    elif is_pro() and page == "Performance & Risk":
        render_performance_risk()

    elif is_pro() and page == "Backtesting":
        render_backtesting()

    elif is_pro() and page == "Watchlist":
        render_watchlist()

    elif is_pro() and page == "Goal Planning":
        render_goals()

    elif is_pro() and page == "Tax Estimator":
        render_taxes()

    # ---------------- PRO FEATURES (AI) ----------------
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

    else:
        st.info("Select a feature from the left menu.")


# ============================================================
# SAFE ENTRYPOINT
# ============================================================

def run_app() -> None:
    """
    Single safe entrypoint for the app.
    Prevents duplicate execution & indentation issues.
    """

    # Authentication gate
    if not logged_in():
        auth_ui()
        return

    # Sidebar navigation
    page = sidebar_nav()

    # Render selected page
    main_router(page)


# ============================================================
# EXECUTION GUARD
# ============================================================

if __name__ == "__main__":
    run_app()
