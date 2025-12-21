# ============================================================
# KATTA WEALTH INSIGHTS — FULL APP (PART 1 / 5)
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
# PART 2 / 5 — COOKIES, SQLITE, AUTH, PRO LOGIC
# ============================================================

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
# 6) TIER / PRO LOGIC (AUTHORITATIVE)
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
# PART 3 / 5 — CORE FEATURES (NO UI ROUTING YET)
# ============================================================

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
        t = str(r["Ticker"]).upper().strip()
        live = get_live_price(t)
        if live["price"] is None:
            continue

        shares = float(r["Shares"])
        cost = float(r["Cost_Basis"])
        value = shares * live["price"]
        pnl = value - (shares * cost)

        rows.append({
            "Ticker": t,
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
# 14) DIVIDEND TRACKER (PRO)
# ============================================================

@st.cache_data(ttl=3600)
def get_dividend_history(ticker: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()

    try:
        divs = yf.Ticker(ticker).dividends
        if divs is None or divs.empty:
            return pd.DataFrame()
        df = divs.reset_index()
        df.columns = ["Date", "Dividend"]
        return df
    except Exception:
        return pd.DataFrame()


def annual_dividend(div_df: pd.DataFrame) -> float:
    if div_df.empty:
        return 0.0
    last_year = div_df[
        div_df["Date"] >= (pd.Timestamp.now() - pd.DateOffset(years=1))
    ]
    return round(float(last_year["Dividend"].sum()), 2)


# ============================================================
# 15) STOCK SCREENER (PRO)
# ============================================================

def screen_stocks(
    tickers: List[str],
    max_pe: float = 25.0,
    min_div_yield: float = 0.0,
) -> pd.DataFrame:
    rows = []

    for t in tickers:
        f = get_stock_fundamentals(t)
        if not f:
            continue

        pe = f.get("PE Ratio")
        dy = f.get("Dividend Yield") or 0

        if pe is not None and pe <= max_pe and dy >= min_div_yield:
            rows.append(f)

    return pd.DataFrame(rows)


# ============================================================
# 16) AI PORTFOLIO CONTEXT BUILDER
# ============================================================

def build_portfolio_insights_context(
    holdings_df: pd.DataFrame,
    fundamentals_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    ctx = {"positions": []}

    if holdings_df is not None and not holdings_df.empty:
        ctx["total_value"] = float(holdings_df.loc["TOTAL", "Market Value"])
        ctx["total_pnl"] = float(holdings_df.loc["TOTAL", "PnL"])

        for _, r in holdings_df.drop(index="TOTAL", errors="ignore").iterrows():
            ctx["positions"].append({
                "ticker": r["Ticker"],
                "market_value": r["Market Value"],
                "pnl": r["PnL"],
            })

    if fundamentals_df is not None and not fundamentals_df.empty:
        ctx["fundamentals"] = fundamentals_df.to_dict(orient="records")

    return ctx
# ============================================================
# PART 4 / 5 — SIDEBAR NAVIGATION & PAGE ROUTER
# ============================================================

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
        )

        st.markdown("---")

        if not is_pro():
            st.info("🔒 Upgrade to Pro to unlock advanced features")

        if st.button("Log out"):
            st.session_state.current_user = None
            st.session_state.tier = "Free"
            cookie_clear_user()
            st.rerun()

        return page


# ============================================================
# 19) PAGE RENDERERS (CENTER CONTENT)
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
# 20) MAIN ROUTER
# ============================================================

def main():
    if not logged_in():
        auth_ui()
        return

    flush_alerts()

    page = sidebar_nav()

    # ---------- FREE ----------
    if page == "Market Scenario":
        render_market_scenario()

    elif page == "Portfolio Analyzer":
        render_portfolio_analyzer()

    elif page == "Live Stocks":
        render_live_stocks()

    elif page == "Stock Research":
        render_stock_research()

    # ---------- PRO ----------
    elif is_pro() and page == "Portfolio Tracker":
        render_portfolio_tracker()

    elif is_pro() and page == "Dividend Tracker":
        render_dividend_tracker()

    elif is_pro() and page == "Stock Screener":
        render_stock_screener()


if __name__ == "__main__":
    main()
# ============================================================
# PART 5 / 5 — ADVANCED FEATURES
# ============================================================

# ============================================================
# 21) PERFORMANCE & RISK
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
    combined = []

    for t in tickers:
        prices = get_price_history(t)
        returns = compute_returns(prices)
        combined.append(returns)

    if not combined:
        st.info("Not enough data.")
        return

    port_returns = pd.concat(combined, axis=1).mean(axis=1)
    st.metric("Volatility (Annualized)", portfolio_volatility(port_returns))
    st.metric("Max Drawdown", max_drawdown((1 + port_returns).cumprod()))


# ============================================================
# 22) BACKTESTING
# ============================================================

def backtest_portfolio(holdings_df: pd.DataFrame, period: str = "3y") -> pd.DataFrame:
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
# 23) WATCHLIST
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
# 24) GOALS
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
# 25) TAXES
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
# 26) ADVISOR LETTER (AI)
# ============================================================

def advisor_letter_ai(context: Dict[str, Any]) -> str:
    prompt = (
        "Write a professional advisor letter summarizing performance, "
        "risk, and outlook. No investment advice.\n\n"
        + safe_json(context)
    )
    return ai(prompt)


def render_advisor_letter():
    st.header("Advisor Letter")

    if st.button("Generate Advisor Letter"):
        ctx = {
            "portfolio": st.session_state.get("portfolio"),
            "client": st.session_state.get("client"),
        }
        letter = advisor_letter_ai(ctx)
        st.text_area("Advisor Letter", letter, height=300)
