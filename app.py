# ============================================================
# KATTA WEALTH INSIGHTS — FULL SINGLE-FILE APP
# PART 1 / 8 — CORE SETUP, CONFIG, SESSION, UI FOUNDATION
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
APP_VERSION = "1.4.0"

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
# 🎨 GLOBAL UI / AESTHETIC HELPERS (STREAMLIT-SAFE)
# ============================================================

def page_header(title: str, subtitle: str | None = None, icon: str = ""):
    st.markdown(
        f"""
        <div style="padding: 0.5rem 0 1.2rem 0;">
            <h2 style="margin-bottom: 0.2rem;">{icon} {title}</h2>
            <p style="color: #6b7280; margin-top: 0;">{subtitle or ""}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str | None = None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")


def divider():
    st.markdown(
        "<hr style='margin: 1.5rem 0; border-color: #e5e7eb;'>",
        unsafe_allow_html=True,
    )


def spacer(lines: int = 1):
    for _ in range(lines):
        st.write("")


def card_metric(label: str, value: Any, delta: Any = None):
    with st.container(border=True):
        st.metric(label, value, delta)


def metric_grid(metrics: List[Tuple[str, str, Optional[str]]]):
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            card_metric(label, value, delta)


# ============================================================
# SESSION STATE
# ============================================================

def init_session() -> None:
    defaults = {
        "current_user": None,
        "tier": "Free",
        "ai_uses": 0,

        # Canonical stock-based portfolio
        "portfolio_raw": None,
        "portfolio": None,
        "portfolio_meta": {},

        "alerts": [],
        "debug": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_session()


# ============================================================
# CORE UTILITIES
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

# ===== END PART 1 / 8 =====
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


def upgrade_current_user_to_pro():
    email = st.session_state.current_user
    if not email:
        return

    if DB_OK:
        db_set_tier(email, "Pro")

    st.session_state.tier = "Pro"


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
    page_header(
        APP_NAME,
        f"Version {APP_VERSION} — Secure login",
        icon="🔐"
    )

    login_tab, signup_tab = st.tabs(["Log In", "Sign Up"])

    with login_tab:
        email = st.text_input("Email", key="login_email")
        pw = st.text_input("Password", type="password", key="login_pw")
        remember = st.checkbox("Remember me", value=True)

        if st.button("Log In", use_container_width=True):
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

        if st.button("Create Account", use_container_width=True):
            if not email or not pw:
                st.error("Missing email or password.")
                return

            if db_create_user(email, hash_pw(pw)):
                st.success("Account created. Please log in.")
            else:
                st.error("Account already exists.")

# ===== END PART 2 / 8 =====
# ============================================================
# PART 3 / 8 — PORTFOLIO ENGINE + ETF LOOK-THROUGH
# ============================================================

# ============================================================
# CANONICAL PORTFOLIO MODEL (STOCK-BASED)
# ============================================================

REQUIRED_PORTFOLIO_COLUMNS = ["Ticker", "Shares", "Cost_Basis"]

def portfolio_template_csv() -> bytes:
    sample = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "NVDA", "VOO"],
        "Shares": [10, 5, 3, 2],
        "Cost_Basis": [150, 280, 400, 350],
    })
    return sample.to_csv(index=False).encode("utf-8")


def validate_portfolio_df(df: pd.DataFrame) -> Tuple[bool, str]:
    if df.empty:
        return False, "Portfolio is empty."

    missing = [c for c in REQUIRED_PORTFOLIO_COLUMNS if c not in df.columns]
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


# ============================================================
# LIVE PRICE FETCH
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
# PORTFOLIO COMPUTATION
# ============================================================

def compute_stock_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).upper().strip()
        live = get_live_price(ticker)

        if live["price"] is None:
            continue

        shares = float(row["Shares"])
        cost = float(row["Cost_Basis"])

        market_value = shares * live["price"]
        cost_value = shares * cost
        pnl = market_value - cost_value

        rows.append({
            "Ticker": ticker,
            "Shares": shares,
            "Live Price": live["price"],
            "Market Value": round(market_value, 2),
            "Cost Basis": round(cost_value, 2),
            "PnL": round(pnl, 2),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out.loc["TOTAL", "Market Value"] = out["Market Value"].sum()
    out.loc["TOTAL", "PnL"] = out["PnL"].sum()

    return out


# ============================================================
# ETF LOOK-THROUGH ENGINE
# ============================================================

KNOWN_ETFS = {
    "SPY", "VOO", "VTI", "QQQ", "IVV", "DIA",
    "SCHB", "SCHX", "IWM", "VT"
}

def is_etf(ticker: str) -> bool:
    if yf is None:
        return False
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("quoteType") == "ETF" or ticker in KNOWN_ETFS
    except Exception:
        return ticker in KNOWN_ETFS


@st.cache_data(ttl=24 * 3600)
def get_etf_holdings(ticker: str, limit: int = 10) -> pd.DataFrame:
    try:
        etf = yf.Ticker(ticker)
        holdings = getattr(etf, "fund_holdings", None)

        if holdings is None or holdings.empty:
            return pd.DataFrame()

        df = holdings[["symbol", "holdingPercent"]].dropna()
        df = df.rename(
            columns={"symbol": "Ticker", "holdingPercent": "Weight"}
        )
        df["Weight"] = df["Weight"] / 100.0

        return df.head(limit)
    except Exception:
        return pd.DataFrame()


def compute_lookthrough_exposure(portfolio: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    stock_rows = []
    sector_rows = []

    total_value = portfolio.loc["TOTAL", "Market Value"]

    for _, row in portfolio.drop(index="TOTAL").iterrows():
        ticker = row["Ticker"]
        position_value = row["Market Value"]
        weight = position_value / total_value

        if is_etf(ticker):
            holdings = get_etf_holdings(ticker)

            for _, h in holdings.iterrows():
                lt_weight = weight * h["Weight"]
                stock_rows.append({
                    "Ticker": h["Ticker"],
                    "Weight": lt_weight
                })

                try:
                    info = yf.Ticker(h["Ticker"]).info or {}
                    sector = info.get("sector", "Other")
                except Exception:
                    sector = "Other"

                sector_rows.append({
                    "Sector": sector,
                    "Weight": lt_weight
                })
        else:
            stock_rows.append({"Ticker": ticker, "Weight": weight})

            try:
                info = yf.Ticker(ticker).info or {}
                sector = info.get("sector", "Other")
            except Exception:
                sector = "Other"

            sector_rows.append({"Sector": sector, "Weight": weight})

    stock_df = (
        pd.DataFrame(stock_rows)
        .groupby("Ticker", as_index=False)
        .sum()
        .sort_values("Weight", ascending=False)
    )
    sector_df = (
        pd.DataFrame(sector_rows)
        .groupby("Sector", as_index=False)
        .sum()
        .sort_values("Weight", ascending=False)
    )

    stock_df["Weight %"] = (stock_df["Weight"] * 100).round(2)
    sector_df["Weight %"] = (sector_df["Weight"] * 100).round(2)

    return {"stocks": stock_df, "sectors": sector_df}


# ============================================================
# PORTFOLIO OVERVIEW PAGE (UNIFIED)
# ============================================================

def render_portfolio_overview():
    page_header(
        "Portfolio Overview",
        "Upload once. Track holdings, exposure, income & goals.",
        icon="📊",
    )

    with st.container(border=True):
        st.markdown("#### 📁 Upload Portfolio")

        st.download_button(
            "⬇️ Download CSV Template",
            portfolio_template_csv(),
            file_name="portfolio_template.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader(
            "Upload CSV (Ticker, Shares, Cost_Basis)",
            type="csv",
            key="portfolio_upload",
        )

    if not uploaded:
        st.info("Upload a portfolio to continue.")
        return

    raw_df = pd.read_csv(uploaded)
    ok, msg = validate_portfolio_df(raw_df)
    if not ok:
        st.error(msg)
        return

    portfolio = compute_stock_portfolio(raw_df)
    if portfolio.empty:
        st.warning("No valid holdings.")
        return

    st.session_state.portfolio_raw = raw_df
    st.session_state.portfolio = portfolio

    divider()

    total_value = portfolio.loc["TOTAL", "Market Value"]
    total_pnl = portfolio.loc["TOTAL", "PnL"]

    metric_grid([
        ("Total Value", f"${round(total_value, 2):,}", None),
        ("Total P&L", f"${round(total_pnl, 2):,}", None),
        ("Holdings", str(len(portfolio.drop(index='TOTAL'))), None),
    ])

    spacer()
    section("📄 Holdings")

    st.dataframe(portfolio, use_container_width=True, height=420)

    spacer()
    section("🧬 Look-Through Exposure")

    exposure = compute_lookthrough_exposure(portfolio)
    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("**Top Stock Exposure**")
            st.dataframe(exposure["stocks"].head(10), use_container_width=True)

    with c2:
        with st.container(border=True):
            st.markdown("**Sector Exposure**")
            st.dataframe(exposure["sectors"], use_container_width=True)

# ===== END PART 3 / 8 =====
# ============================================================
# PART 4 / 8 — DIVIDENDS & INCOME ENGINE
# ============================================================

# ============================================================
# DIVIDEND HISTORY (TTM, TIMEZONE-SAFE)
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
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["Date"])

        return df
    except Exception:
        return pd.DataFrame()


def trailing_12m_dividend(div_df: pd.DataFrame) -> float:
    if div_df.empty:
        return 0.0

    cutoff = pd.Timestamp.now().tz_localize(None) - pd.DateOffset(years=1)
    return round(float(div_df[div_df["Date"] >= cutoff]["Dividend"].sum()), 4)


# ============================================================
# DIVIDEND INCOME PER HOLDING
# ============================================================

def compute_dividend_income(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in portfolio.drop(index="TOTAL", errors="ignore").iterrows():
        ticker = row["Ticker"]
        shares = float(row["Shares"])
        live_price = float(row["Live Price"])
        market_value = float(row["Market Value"])

        div_df = get_dividend_history(ticker)
        annual_div_per_share = trailing_12m_dividend(div_df)

        annual_income = annual_div_per_share * shares
        yield_pct = (
            (annual_div_per_share / live_price) * 100
            if live_price > 0 else 0.0
        )

        rows.append({
            "Ticker": ticker,
            "Annual Dividend / Share": round(annual_div_per_share, 4),
            "Dividend Yield %": round(yield_pct, 2),
            "Annual Income ($)": round(annual_income, 2),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.loc["TOTAL", "Annual Income ($)"] = df["Annual Income ($)"].sum()

    return df


# ============================================================
# EXTEND: PORTFOLIO OVERVIEW — INCOME SECTION
# ============================================================

def render_portfolio_income_section(portfolio: pd.DataFrame):
    divider()
    page_header(
        "Dividend Income",
        "Real trailing-12-month income from your holdings",
        icon="💵",
    )

    income_df = compute_dividend_income(portfolio)

    if income_df.empty:
        st.info("No dividend data available for current holdings.")
        st.session_state.portfolio_meta["income"] = {}
        return

    c1, c2 = st.columns([2, 1])

    with c1:
        with st.container(border=True):
            st.markdown("**Income by Holding**")
            st.dataframe(
                income_df,
                use_container_width=True,
                height=320,
            )

    total_income = float(income_df.loc["TOTAL", "Annual Income ($)"])
    total_value = float(portfolio.loc["TOTAL", "Market Value"])
    portfolio_yield = (total_income / total_value * 100) if total_value > 0 else 0.0

    with c2:
        card_metric(
            "Annual Portfolio Income",
            f"${round(total_income, 2):,}"
        )
        card_metric(
            "Portfolio Yield",
            f"{round(portfolio_yield, 2)}%"
        )

    # Persist for AI + goals
    st.session_state.portfolio_meta["income"] = {
        "total_income": round(total_income, 2),
        "portfolio_yield": round(portfolio_yield, 2),
        "by_holding": income_df.to_dict(),
    }


# ============================================================
# HOOK INCOME SECTION INTO PORTFOLIO OVERVIEW
# ============================================================

# Call this at the END of render_portfolio_overview()
# (Placed here so definition exists before router)

def render_portfolio_overview_with_income():
    render_portfolio_overview()
    portfolio = st.session_state.get("portfolio")
    if portfolio is not None and not portfolio.empty:
        render_portfolio_income_section(portfolio)

# ===== END PART 4 / 8 =====
# ============================================================
# PART 5 / 8 — MONTE CARLO GOAL PROBABILITY ENGINE
# ============================================================

# ============================================================
# RETURN & VOLATILITY ESTIMATION (FROM REAL HOLDINGS)
# ============================================================

@st.cache_data(ttl=3600)
def estimate_portfolio_stats(portfolio: pd.DataFrame) -> Tuple[float, float]:
    """
    Estimate expected annual return and volatility from holdings.
    Uses historical prices (3Y) and equal-weight aggregation.
    """
    if yf is None or portfolio.empty:
        return 0.07, 0.15  # conservative defaults

    returns = []

    for _, row in portfolio.drop(index="TOTAL", errors="ignore").iterrows():
        try:
            prices = yf.Ticker(row["Ticker"]).history(period="3y")["Close"]
            r = prices.pct_change().dropna()
            if not r.empty:
                returns.append(r)
        except Exception:
            continue

    if not returns:
        return 0.07, 0.15

    combined = pd.concat(returns, axis=1).mean(axis=1)

    exp_return = float(combined.mean() * 252)
    volatility = float(combined.std() * np.sqrt(252))

    return round(exp_return, 4), round(volatility, 4)


# ============================================================
# MONTE CARLO SIMULATION CORE
# ============================================================

def monte_carlo_simulation(
    start_value: float,
    annual_contribution: float,
    years: int,
    exp_return: float,
    volatility: float,
    simulations: int = 3000,
) -> np.ndarray:
    """
    Monte Carlo simulation using lognormal-style returns.
    """
    steps = years
    results = np.zeros((simulations, steps))

    for i in range(simulations):
        value = start_value
        for y in range(steps):
            shock = np.random.normal(exp_return, volatility)
            value = value * (1 + shock) + annual_contribution
            results[i, y] = value

    return results


def goal_success_probability(simulations: np.ndarray, goal: float) -> float:
    final_values = simulations[:, -1]
    return round((final_values >= goal).mean() * 100, 2)


# ============================================================
# GOAL PROBABILITY PAGE (PRO)
# ============================================================

def render_goal_probability():
    page_header(
        "Goal Probability",
        "Estimate the likelihood of reaching your long-term goal",
        icon="🎯",
    )

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    total_value = float(portfolio.loc["TOTAL", "Market Value"])

    c1, c2, c3 = st.columns(3)

    with c1:
        goal = st.number_input(
            "Target Goal ($)",
            min_value=0.0,
            value=1_000_000.0,
            step=50_000.0,
        )

    with c2:
        annual = st.number_input(
            "Annual Contribution ($)",
            min_value=0.0,
            value=10_000.0,
            step=1_000.0,
        )

    with c3:
        years = st.slider("Years", 1, 40, 20)

    divider()

    exp_return, volatility = estimate_portfolio_stats(portfolio)

    sims = monte_carlo_simulation(
        start_value=total_value,
        annual_contribution=annual,
        years=years,
        exp_return=exp_return,
        volatility=volatility,
    )

    probability = goal_success_probability(sims, goal)

    metric_grid([
        ("Expected Return", f"{round(exp_return * 100, 2)}%", None),
        ("Volatility", f"{round(volatility * 100, 2)}%", None),
        ("Goal Success Probability", f"{probability}%", None),
    ])

    final_vals = sims[:, -1]

    summary = pd.DataFrame({
        "Scenario": ["Pessimistic (10%)", "Median (50%)", "Optimistic (90%)"],
        "Ending Value ($)": [
            round(np.percentile(final_vals, 10), 0),
            round(np.percentile(final_vals, 50), 0),
            round(np.percentile(final_vals, 90), 0),
        ],
    })

    spacer()
    with st.container(border=True):
        st.markdown("**Outcome Distribution**")
        st.dataframe(summary, use_container_width=True)

    # Persist for AI
    st.session_state.portfolio_meta["goal_probability"] = {
        "goal": goal,
        "years": years,
        "probability": probability,
        "expected_return": exp_return,
        "volatility": volatility,
    }

# ===== END PART 5 / 8 =====
# ============================================================
# PART 6 / 8 — AI ENGINE & AI FEATURES (PRO)
# ============================================================

# ============================================================
# AI CORE (GROQ CLIENT)
# ============================================================

_groq_client = None

def ai(prompt: str) -> str:
    """
    Central AI helper.
    Educational only — no investment advice.
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
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI error: {str(e)}"


# ============================================================
# AI CONTEXT BUILDER
# ============================================================

def build_ai_context() -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}

    portfolio = st.session_state.get("portfolio")
    meta = st.session_state.get("portfolio_meta", {})

    if portfolio is not None and not portfolio.empty:
        if "TOTAL" in portfolio.index:
            ctx["total_value"] = float(portfolio.loc["TOTAL", "Market Value"])
            ctx["total_pnl"] = float(portfolio.loc["TOTAL", "PnL"])

        ctx["positions"] = [
            {
                "ticker": row.get("Ticker"),
                "market_value": row.get("Market Value"),
                "pnl": row.get("PnL"),
            }
            for _, row in portfolio.drop(index="TOTAL", errors="ignore").iterrows()
        ]

    if meta:
        ctx["meta"] = meta

    return ctx


def ai_block(title: str, prompt: str):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(ai(prompt))


# ============================================================
# AI PAGES
# ============================================================

def render_portfolio_health_ai():
    page_header(
        "Portfolio Health",
        "High-level educational assessment of portfolio structure",
        icon="🧠",
    )

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

    metric_grid([
        ("Health Score", f"{score} / 100", None),
        ("Total Value", f"${round(total_value, 2):,}", None),
        ("Total P&L", f"${round(total_pnl, 2):,}", None),
    ])

    ai_block(
        "What this score means",
        f"Explain what a portfolio health score of {score} means "
        f"for a long-term investor.\n\nContext:\n{safe_json(build_ai_context())}",
    )


def render_ai_rebalancing():
    page_header(
        "AI Rebalancing",
        "Educational rebalancing ideas (not advice)",
        icon="⚖️",
    )

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    ai_block(
        "Rebalancing Insights",
        "Analyze this portfolio and suggest educational "
        "rebalancing concepts (no advice):\n\n"
        + safe_json(build_ai_context()),
    )


def render_income_forecast_ai():
    page_header(
        "Income Forecast",
        "Understanding dividend-based income",
        icon="💵",
    )

    meta = st.session_state.get("portfolio_meta", {})
    income = meta.get("income", {}).get("total_income")

    if income is None:
        st.info("Upload a dividend-paying portfolio first.")
        return

    metric_grid([
        ("Estimated Annual Income", f"${round(income, 2):,}", None),
    ])

    ai_block(
        "Income Explanation",
        f"Explain dividend income investing using an estimated "
        f"annual income of ${round(income, 2)}.\n\n"
        f"Context:\n{safe_json(build_ai_context())}",
    )


def render_teen_explainer_ai():
    page_header(
        "Teen Explainer",
        "Your portfolio explained simply",
        icon="🎓",
    )

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    ai_block(
        "Explain it like I'm in high school",
        "Explain this portfolio to a high-school student "
        "who is learning about investing:\n\n"
        + safe_json(build_ai_context()),
    )


def render_ai_chatbot():
    page_header(
        "AI Chatbot",
        "Ask questions about markets, portfolios, and finance concepts",
        icon="💬",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        role = "You" if msg["role"] == "user" else "AI"
        st.markdown(f"**{role}:** {msg['content']}")

    user_input = st.text_input("Ask a question")

    if st.button("Send", use_container_width=True) and user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        response = ai(
            "Context:\n"
            + safe_json(build_ai_context())
            + "\n\nUser question:\n"
            + user_input
        )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )
        st.rerun()

# ===== END PART 6 / 8 =====
# ============================================================
# PART 7 / 8 — FEATURE REGISTRY, SIDEBAR, ROUTER
# ============================================================

# ============================================================
# FEATURE REGISTRY
# ============================================================

FREE_PAGES = [
    "Portfolio Overview",
]

PRO_PAGES = [
    "Goal Probability",
    "Portfolio Health",
    "AI Rebalancing",
    "Income Forecast",
    "Teen Explainer",
    "AI Chatbot",
]


def allowed_pages() -> List[str]:
    if is_pro():
        return FREE_PAGES + PRO_PAGES
    return FREE_PAGES


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

def sidebar_nav() -> str:
    with st.sidebar:
        st.markdown("## 💎 Katta Wealth")

        if logged_in():
            st.caption(f"👤 {st.session_state.current_user}")
            st.caption(f"Plan: {'Pro' if is_pro() else 'Free'}")
        else:
            st.caption("Not logged in")

        divider()

        st.markdown("### 📍 Explore")

        page = st.radio(
            "",
            allowed_pages(),
            key="nav_radio",
        )

        divider()

        if not is_pro():
            st.info("Unlock AI insights, simulations & planning")

            if st.button("💎 Upgrade to Pro (Demo)", use_container_width=True):
                upgrade_current_user_to_pro()
                st.success("You are now Pro!")
                st.rerun()

        if logged_in():
            if st.button("🚪 Log out", use_container_width=True):
                st.session_state.current_user = None
                st.session_state.tier = "Free"
                cookie_clear_user()
                st.rerun()

        return page


# ============================================================
# MAIN ROUTER
# ============================================================

def main_router(page: str) -> None:

    # ---------- FREE ----------
    if page == "Portfolio Overview":
        render_portfolio_overview_with_income()

    # ---------- PRO ----------
    elif is_pro() and page == "Goal Probability":
        render_goal_probability()

    elif is_pro() and page == "Portfolio Health":
        render_portfolio_health_ai()

    elif is_pro() and page == "AI Rebalancing":
        render_ai_rebalancing()

    elif is_pro() and page == "Income Forecast":
        render_income_forecast_ai()

    elif is_pro() and page == "Teen Explainer":
        render_teen_explainer_ai()

    elif is_pro() and page == "AI Chatbot":
        render_ai_chatbot()

    else:
        st.info("Select a feature from the left menu.")

# ===== END PART 7 / 8 =====
# ============================================================
# PART 8 / 8 — SAFE ENTRYPOINT & EXECUTION
# ============================================================

def run_app() -> None:
    """
    Single safe entrypoint for the app.
    Prevents duplicate execution and keeps auth gating clean.
    """

    # Authentication gate
    if not logged_in():
        auth_ui()
        return

    # Sidebar navigation
    page = sidebar_nav()

    # Route to selected page
    main_router(page)


# ============================================================
# EXECUTION GUARD
# ============================================================

if __name__ == "__main__":
    run_app()

# ===== END PART 8 / 8 =====
# ============================================================
# PART 9 / 9 — VISUAL ANALYTICS & INSIGHTS DASHBOARD
# ============================================================

# ============================================================
# CHART HELPERS (STREAMLIT-NATIVE, SAFE)
# ============================================================

def plot_allocation_pie(portfolio: pd.DataFrame):
    df = portfolio.drop(index="TOTAL", errors="ignore").copy()
    df = df[["Ticker", "Market Value"]]

    st.subheader("📊 Allocation by Holding")
    st.pyplot(
        df.set_index("Ticker")
          .plot.pie(
              y="Market Value",
              figsize=(5, 5),
              autopct="%1.1f%%",
              legend=False
          )
          .get_figure()
    )


def plot_monte_carlo_paths(simulations: np.ndarray, years: int):
    st.subheader("📈 Monte Carlo Portfolio Paths")

    df = pd.DataFrame(simulations.T)
    st.line_chart(df.sample(min(50, df.shape[1]), axis=1))


# ============================================================
# PORTFOLIO INSIGHTS DASHBOARD (NEW PAGE)
# ============================================================

def render_portfolio_insights():
    page_header(
        "Portfolio Insights",
        "Visual summary of allocation, risk, income, and goals",
        icon="📊",
    )

    portfolio = st.session_state.get("portfolio")
    meta = st.session_state.get("portfolio_meta", {})

    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    # ---------------------------
    # Snapshot Metrics
    # ---------------------------
    total_value = portfolio.loc["TOTAL", "Market Value"]
    total_pnl = portfolio.loc["TOTAL", "PnL"]

    income = meta.get("income", {}).get("total_income", 0)
    goal_prob = meta.get("goal_probability", {}).get("probability")

    metric_grid([
        ("Portfolio Value", f"${round(total_value, 2):,}", None),
        ("Total P&L", f"${round(total_pnl, 2):,}", None),
        ("Annual Income", f"${round(income, 2):,}", None),
        ("Goal Success", f"{goal_prob}%" if goal_prob else "—", None),
    ])

    divider()

    # ---------------------------
    # Allocation Visualization
    # ---------------------------
    with st.container(border=True):
        plot_allocation_pie(portfolio)

    divider()

    # ---------------------------
    # Monte Carlo Visualization (If Available)
    # ---------------------------
    if "goal_probability" in meta:
        gp = meta["goal_probability"]

        exp_return = gp.get("expected_return", 0.07)
        volatility = gp.get("volatility", 0.15)
        years = gp.get("years", 20)

        sims = monte_carlo_simulation(
            start_value=total_value,
            annual_contribution=0,
            years=years,
            exp_return=exp_return,
            volatility=volatility,
            simulations=2000,
        )

        with st.container(border=True):
            plot_monte_carlo_paths(sims, years)

    divider()

    # ---------------------------
    # AI Summary (Optional, Pro)
    # ---------------------------
    if is_pro():
        ai_block(
            "Portfolio Summary",
            "Summarize this portfolio’s allocation, income profile, "
            "risk posture, and goal outlook in plain English:\n\n"
            + safe_json(build_ai_context()),
        )


# ============================================================
# REGISTER INSIGHTS PAGE
# ============================================================

# Add page dynamically without touching earlier lists
if "Portfolio Insights" not in FREE_PAGES:
    FREE_PAGES.append("Portfolio Insights")


# ============================================================
# EXTEND ROUTER (SAFE PATCH)
# ============================================================

_old_main_router = main_router

def main_router(page: str) -> None:
    if page == "Portfolio Insights":
        render_portfolio_insights()
    else:
        _old_main_router(page)

# ===== END PART 9 / 9 =====
# ============================================================
# PART 10 / 13 — RISK ALERTS & DRAWDOWN MONITORING
# ============================================================

def compute_drawdown(series: pd.Series) -> float:
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return round(drawdown.min() * 100, 2)


def render_risk_alerts():
    page_header(
        "Risk Alerts",
        "Monitor volatility and drawdown risk",
        icon="🚨",
    )

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    returns = []

    for _, row in portfolio.drop(index="TOTAL", errors="ignore").iterrows():
        try:
            prices = yf.Ticker(row["Ticker"]).history(period="1y")["Close"]
            returns.append(prices)
        except Exception:
            continue

    if not returns:
        st.info("Not enough data for risk analysis.")
        return

    combined = pd.concat(returns, axis=1).mean(axis=1)
    dd = compute_drawdown(combined)

    metric_grid([
        ("Max Drawdown (1Y)", f"{dd}%", None),
        ("Risk Level", "High" if dd < -20 else "Moderate", None),
    ])

    if is_pro():
        ai_block(
            "Risk Explanation",
            f"Explain what a {dd}% drawdown means for a long-term investor.\n\n"
            + safe_json(build_ai_context()),
        )
# ============================================================
# PART 11 / 13 — TAX OPTIMIZATION (EDUCATIONAL)
# ============================================================

def estimate_capital_gains_tax(portfolio: pd.DataFrame, tax_rate: float) -> float:
    pnl = portfolio.loc["TOTAL", "PnL"]
    return round(max(pnl, 0) * tax_rate, 2)


def render_tax_optimization():
    page_header(
        "Tax Optimization",
        "Understand potential capital gains impact",
        icon="🧾",
    )

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    tax_rate = st.slider("Estimated Capital Gains Tax Rate", 0.0, 0.4, 0.15)

    tax = estimate_capital_gains_tax(portfolio, tax_rate)

    metric_grid([
        ("Unrealized Gains", f"${round(portfolio.loc['TOTAL', 'PnL'], 2):,}", None),
        ("Estimated Tax", f"${tax:,}", None),
    ])

    if is_pro():
        ai_block(
            "Tax Awareness",
            "Explain tax-efficient investing strategies at a high level "
            "without giving advice.\n\n"
            + safe_json(build_ai_context()),
        )
# ============================================================
# PART 12 / 13 — PERFORMANCE & BENCHMARKING
# ============================================================

def render_performance_benchmark():
    page_header(
        "Performance",
        "Compare your portfolio vs the market",
        icon="📈",
    )

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    prices = []

    for _, row in portfolio.drop(index="TOTAL", errors="ignore").iterrows():
        try:
            p = yf.Ticker(row["Ticker"]).history(period="1y")["Close"]
            prices.append(p)
        except Exception:
            continue

    if not prices:
        st.info("Not enough data.")
        return

    portfolio_series = pd.concat(prices, axis=1).mean(axis=1)
    benchmark = yf.Ticker("SPY").history(period="1y")["Close"]

    df = pd.DataFrame({
        "Portfolio": (portfolio_series / portfolio_series.iloc[0]) - 1,
        "S&P 500 (SPY)": (benchmark / benchmark.iloc[0]) - 1,
    })

    st.line_chart(df)
# ============================================================
# PART 13 / 13 — EXPORTS & REPORTING
# ============================================================

def render_exports():
    page_header(
        "Exports & Reports",
        "Download portfolio data and summaries",
        icon="📤",
    )

    portfolio = st.session_state.get("portfolio")
    meta = st.session_state.get("portfolio_meta", {})

    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    st.download_button(
        "⬇️ Download Portfolio CSV",
        portfolio.to_csv().encode("utf-8"),
        "portfolio_snapshot.csv",
        "text/csv",
    )

    st.download_button(
        "⬇️ Download Portfolio JSON",
        json.dumps(
            {
                "portfolio": portfolio.to_dict(),
                "meta": meta,
            },
            indent=2,
        ).encode("utf-8"),
        "portfolio_snapshot.json",
        "application/json",
    )

    st.success("Exports are advisor- and audit-ready.")
# ============================================================
# PART 14 / 14 — AI-NATIVE LANDING PAGE (KATTA WEALTH INSIGHTS)
# ============================================================

def render_landing_page():
    st.markdown(
        """
        <style>
        .hero-container {
            display: grid;
            grid-template-columns: 1.1fr 1fr;
            gap: 3rem;
            padding: 4rem 2rem;
        }

        .hero-badge {
            display: inline-block;
            background: #e8f0ff;
            color: #2563eb;
            padding: 0.4rem 0.9rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 1rem;
        }

        .hero-title span {
            color: #2563eb;
        }

        .hero-subtitle {
            font-size: 1.1rem;
            color: #4b5563;
            max-width: 520px;
            margin-bottom: 2rem;
        }

        .hero-actions {
            display: flex;
            gap: 1rem;
        }

        .btn-primary {
            background: #2563eb;
            color: white;
            padding: 0.75rem 1.4rem;
            border-radius: 0.6rem;
            font-weight: 600;
            text-decoration: none;
        }

        .btn-secondary {
            background: #f3f4f6;
            color: #111827;
            padding: 0.75rem 1.4rem;
            border-radius: 0.6rem;
            font-weight: 600;
            text-decoration: none;
        }

        .feature-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2rem;
            margin-top: 3.5rem;
            padding: 0 2rem;
        }

        .feature-card {
            background: #f9fafb;
            border-radius: 1rem;
            padding: 2rem;
            border: 1px solid #e5e7eb;
        }

        .feature-icon {
            font-size: 1.8rem;
            margin-bottom: 0.7rem;
        }

        .feature-title {
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .feature-desc {
            color: #4b5563;
            font-size: 0.95rem;
        }

        @media (max-width: 900px) {
            .hero-container {
                grid-template-columns: 1fr;
            }
            .feature-row {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-container">
            <div>
                <div class="hero-badge">AI-Native Wealth Platform</div>

                <div class="hero-title">
                    <span>Katta Wealth Insights</span><br>
                    Research Workspace for<br>
                    Modern Investors
                </div>

                <div class="hero-subtitle">
                    Katta Wealth Insights harnesses AI, market data, and probabilistic
                    modeling to help investors understand portfolios, income, risk,
                    and long-term outcomes — all in one intelligent workspace.
                </div>

                <div class="hero-actions">
                    <a class="btn-primary" href="#">Get Started →</a>
                    <a class="btn-secondary" href="#">Learn More</a>
                </div>
            </div>

            <div>
                <img src="https://raw.githubusercontent.com/streamlit/brand/master/logos/mark/streamlit-mark-primary.png"
                     style="width:100%; border-radius: 1rem; box-shadow: 0 20px 50px rgba(0,0,0,0.15);" />
            </div>
        </div>

        <div class="feature-row">
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">AI Portfolio Intelligence</div>
                <div class="feature-desc">
                    Unified portfolio tracking with ETF look-through, income analysis,
                    and risk diagnostics.
                </div>
            </div>

            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Goal Probability Engine</div>
                <div class="feature-desc">
                    Monte Carlo simulations quantify the probability of reaching
                    financial goals — not guesses.
                </div>
            </div>

            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">AI Research Assistant</div>
                <div class="feature-desc">
                    Context-aware AI explains portfolios, income, and risks in
                    clear, investor-friendly language.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAKE LANDING PAGE THE DEFAULT HOME
# ============================================================

_old_run_app = run_app

def run_app():
    if not logged_in():
        render_landing_page()
        spacer(2)
        st.markdown("### 🔐 Sign in to continue")
        auth_ui()
        return

    page = sidebar_nav()
    main_router(page)
# ============================================================
# PART 15 / 15 — WEBSITE TABS & DEMO GATING
# ============================================================

def render_home_tab():
    page_header(
        "Katta Wealth Insights",
        "AI-native research and planning workspace for modern investors",
        icon="💎",
    )

    st.markdown(
        """
        Katta Wealth Insights is an **AI-powered wealth intelligence platform**
        designed to help investors understand portfolios, income, risk,
        and long-term outcomes — all in one unified workspace.
        """
    )

    metric_grid([
        ("AI Insights", "Context-Aware", None),
        ("Portfolio Types", "Stocks & ETFs", None),
        ("Planning Horizon", "Probabilistic", None),
    ])


def render_about_tab():
    page_header(
        "About Us",
        "Built for clarity, discipline, and long-term thinking",
        icon="🏛️",
    )

    st.markdown(
        """
        **Katta Wealth Insights** was created to bridge the gap between
        raw financial data and real investor understanding.

        We believe:
        - Markets are uncertain — probabilities matter
        - Investors need explanations, not noise
        - AI should clarify, not confuse

        Our platform blends **data, modeling, and AI** to help investors
        reason about decisions with confidence.
        """
    )


def render_features_tab():
    page_header(
        "Features",
        "A complete AI-native wealth research stack",
        icon="✨",
    )

    st.markdown(
        """
        **Portfolio Intelligence**
        - Stock & ETF tracking
        - Look-through exposure
        - Dividend income analysis

        **Planning & Risk**
        - Monte Carlo goal probability
        - Drawdown & volatility monitoring
        - Benchmark comparisons

        **AI Research**
        - Portfolio explanations
        - Teen-friendly education mode
        - Context-aware AI chatbot

        **Enterprise-Grade**
        - Secure authentication
        - Audit-ready exports
        - Scalable architecture
        """
    )


def render_how_it_works_tab():
    page_header(
        "How It Works",
        "From data to insight in minutes",
        icon="⚙️",
    )

    st.markdown(
        """
        **1. Upload Your Portfolio**
        Upload a simple CSV with tickers and shares.

        **2. Analyze Automatically**
        Katta Wealth Insights calculates value, income,
        exposure, and risk in real time.

        **3. Simulate Outcomes**
        Run Monte Carlo simulations to understand
        the probability of reaching your goals.

        **4. Ask AI**
        Get clear explanations — not advice —
        grounded in your actual data.
        """
    )
def render_demo_tab():
    page_header(
        "Live Demo",
        "Explore the full Katta Wealth Insights platform",
        icon="🚀",
    )

    st.info(
        "All features are available inside the demo environment. "
        "Sign in to explore portfolios, simulations, and AI insights."
    )

    # 🔐 AUTH + FULL APP
    if not logged_in():
        auth_ui()
        return

    page = sidebar_nav()
    main_router(page)
def run_app():
    # Top-level website tabs
    tabs = st.tabs([
        "Home",
        "About Us",
        "Features",
        "How It Works",
        "Demo",
    ])

    with tabs[0]:
        render_home_tab()

    with tabs[1]:
        render_about_tab()

    with tabs[2]:
        render_features_tab()

    with tabs[3]:
        render_how_it_works_tab()

    with tabs[4]:
        render_demo_tab()
# ============================================================
# GLOBAL APP MODE
# ============================================================

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "marketing"


# ============================================================
# TOP NAV BAR (ALWAYS VISIBLE)
# ============================================================

def render_top_nav():
    st.markdown(
        """
        <style>
        .top-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.8rem 2rem;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 1.5rem;
        }
        .nav-left {
            font-weight: 800;
            font-size: 1.1rem;
        }
        .nav-right button {
            background: none;
            border: none;
            font-weight: 600;
            margin-left: 1.2rem;
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns([3, 5])

    with cols[0]:
        st.markdown("### 💎 Katta Wealth Insights")

    with cols[1]:
        nav_cols = st.columns(5)
        labels = ["About Us", "Features", "How It Works", "Benefits", "Demo"]

        for i, label in enumerate(labels):
            if nav_cols[i].button(label, use_container_width=True):
                if label == "Demo":
                    st.session_state.app_mode = "demo"
                else:
                    st.session_state.app_mode = label.lower().replace(" ", "_")
                st.rerun()


# ============================================================
# MARKETING PAGES
# ============================================================

def render_about():
    page_header("About Us", "Built for long-term thinking", "🏛️")
    st.write(
        """
        Katta Wealth Insights is designed to help investors reason clearly
        about portfolios, income, risk, and long-term outcomes.

        We focus on **probabilities, transparency, and education** —
        not hype or predictions.
        """
    )


def render_features():
    page_header("Features", "AI-native wealth intelligence", "✨")
    st.markdown("""
    - Stock & ETF portfolio tracking  
    - ETF look-through exposure  
    - Dividend income analytics  
    - Monte Carlo goal probability  
    - Risk & drawdown monitoring  
    - AI explanations & education mode  
    """)


def render_how_it_works():
    page_header("How It Works", "From data to insight", "⚙️")
    st.markdown("""
    1. Upload your portfolio  
    2. Analyze exposure, income, and risk  
    3. Simulate future outcomes  
    4. Ask AI for explanations  
    """)


def render_benefits():
    page_header("Benefits", "Why investors use KWI", "🎯")
    st.markdown("""
    - Clarity over complexity  
    - Probabilities instead of guesses  
    - AI explanations grounded in your data  
    - Built for investors, students, and families  
    """)


# ============================================================
# FINAL ENTRYPOINT (THIS IS THE KEY)
# ============================================================

def run_app():
    # 🔝 Always render top navigation
    render_top_nav()

    # 🟦 MARKETING MODE (NO SIDEBAR, NO APP)
    if st.session_state.app_mode != "demo":
        if st.session_state.app_mode == "about_us":
            render_about()
        elif st.session_state.app_mode == "features":
            render_features()
        elif st.session_state.app_mode == "how_it_works":
            render_how_it_works()
        elif st.session_state.app_mode == "benefits":
            render_benefits()
        else:
            render_about()  # default landing
        return

    # 🚀 DEMO MODE — FULL APP ENABLED
    if not logged_in():
        auth_ui()
        return

    page = sidebar_nav()
    main_router(page)


# ============================================================
# EXECUTION GUARD
# ============================================================

if __name__ == "__main__":
    run_app()
