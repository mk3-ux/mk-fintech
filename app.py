# ============================================================
# KATTA WEALTH INSIGHTS — APP ENTRYPOINT
# FILE 1 / 9 — app.py
# ============================================================

from __future__ import annotations

import streamlit as st

# Core imports
from core.session import init_session
from core.nav import render_top_nav
from core.routing import route_app

# ============================================================
# STREAMLIT CONFIG (ONLY PLACE THIS EXISTS)
# ============================================================

st.set_page_config(
    page_title="Katta Wealth Insights",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# APP BOOTSTRAP
# ============================================================

def run_app() -> None:
    """
    Single, authoritative entrypoint.
    No widgets should be defined here.
    """

    # Initialize session state once
    init_session()

    # Always render top navigation
    render_top_nav()

    # Route application based on mode (marketing vs demo)
    route_app()


# ============================================================
# EXECUTION GUARD (ONLY ONE IN ENTIRE CODEBASE)
# ============================================================

if __name__ == "__main__":
    run_app()
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 2 / 9 — core/session.py
# ============================================================

from __future__ import annotations

import os
import sqlite3
import hashlib
import datetime as dt
from typing import Optional, Dict, Any

import streamlit as st

try:
    from streamlit_cookies_manager import EncryptedCookieManager
except Exception:
    EncryptedCookieManager = None


# ============================================================
# GLOBAL CONSTANTS
# ============================================================

USE_SQLITE = True
SQLITE_PATH = "kwi_app.db"

USE_COOKIES = True
COOKIE_PASSWORD = os.getenv("KWI_COOKIE_PASSWORD", "change-this-secret")

# Payment (stub — real Stripe later)
PAID_PLAN_NAME = "Pro"
PAID_PRICE_MONTHLY = 24


# ============================================================
# SESSION INITIALIZATION (AUTHORITATIVE)
# ============================================================

def init_session() -> None:
    """
    Initialize ALL session keys exactly once.
    This function is idempotent and safe.
    """

    defaults = {
        # App mode
        "app_mode": "marketing",  # marketing | demo

        # Auth
        "current_user": None,
        "is_paid": False,

        # Portfolio state
        "portfolio_raw": None,
        "portfolio": None,
        "portfolio_meta": {},

        # AI
        "chat_history": [],

        # Alerts / debug
        "alerts": [],
        "debug": False,
    }

    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ============================================================
# TIME + HASH HELPERS
# ============================================================

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def hash_pw(password: str) -> str:
    return sha256("kwi_salt_" + password)


# ============================================================
# COOKIE MANAGER
# ============================================================

cookies = None

def cookies_ready() -> bool:
    global cookies
    if not USE_COOKIES or EncryptedCookieManager is None:
        return False

    cookies = EncryptedCookieManager(
        prefix="kwi_",
        password=COOKIE_PASSWORD,
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
    cookies["user"] = email
    cookies.save()


def cookie_clear_user() -> None:
    if not _COOKIES_OK:
        return
    cookies["user"] = ""
    cookies.save()


# ============================================================
# SQLITE DATABASE
# ============================================================

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def db_init() -> None:
    if not USE_SQLITE:
        return

    conn = db_connect()

    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        pw_hash TEXT NOT NULL,
        is_paid INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    );
    """)

    conn.commit()
    conn.close()


db_init()


# ============================================================
# USER HELPERS
# ============================================================

def db_get_user(email: str) -> Optional[Dict[str, Any]]:
    conn = db_connect()
    row = conn.execute(
        "SELECT email, pw_hash, is_paid FROM users WHERE email=?",
        (email,),
    ).fetchone()
    conn.close()

    if not row:
        return None

    return {
        "email": row[0],
        "pw": row[1],
        "is_paid": bool(row[2]),
    }


def db_create_user(email: str, pw_hash: str) -> bool:
    try:
        conn = db_connect()
        conn.execute(
            "INSERT INTO users VALUES (?,?,?,?)",
            (email, pw_hash, 0, now_iso()),
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def db_set_paid(email: str, paid: bool) -> None:
    conn = db_connect()
    conn.execute(
        "UPDATE users SET is_paid=? WHERE email=?",
        (1 if paid else 0, email),
    )
    conn.commit()
    conn.close()


# ============================================================
# AUTH STATE HELPERS
# ============================================================

def logged_in() -> bool:
    return st.session_state.current_user is not None


def is_paid_user() -> bool:
    return bool(st.session_state.get("is_paid", False))


def login_user(email: str, remember: bool = True) -> None:
    user = db_get_user(email)
    if not user:
        return

    st.session_state.current_user = email
    st.session_state.is_paid = user["is_paid"]

    if remember:
        cookie_set_user(email)


def logout_user() -> None:
    st.session_state.current_user = None
    st.session_state.is_paid = False
    cookie_clear_user()


def auto_login() -> None:
    if logged_in():
        return

    saved = cookie_get_user()
    if not saved:
        return

    user = db_get_user(saved)
    if user:
        st.session_state.current_user = saved
        st.session_state.is_paid = user["is_paid"]


# Run auto-login once per session
auto_login()


# ============================================================
# PAYMENT STUB (NO STRIPE YET)
# ============================================================

def mark_user_as_paid() -> None:
    """
    Demo-only payment unlock.
    """
    email = st.session_state.current_user
    if not email:
        return

    db_set_paid(email, True)
    st.session_state.is_paid = True
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 3 / 9 — core/nav.py
# ============================================================

from __future__ import annotations

import streamlit as st


# ============================================================
# TOP NAVIGATION BAR
# ============================================================

def render_top_nav() -> None:
    """
    Always-visible top navigation bar.
    This is the ONLY place where app_mode is changed.
    """

    st.markdown(
        """
        <style>
        .kwi-top-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 2rem;
            border-bottom: 1px solid #1f2933;
            background-color: #0b0f19;
            margin-bottom: 1.5rem;
        }

        .kwi-brand {
            font-weight: 800;
            font-size: 1.15rem;
            color: #e5e7eb;
        }

        .kwi-nav-buttons button {
            background: none;
            border: 1px solid #374151;
            color: #e5e7eb;
            padding: 0.35rem 0.9rem;
            border-radius: 0.5rem;
            font-weight: 600;
            margin-left: 0.6rem;
            cursor: pointer;
        }

        .kwi-nav-buttons button:hover {
            border-color: #6366f1;
            color: #c7d2fe;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 7])

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------
    with left:
        st.markdown(
            "<div class='kwi-brand'>💎 Katta Wealth Insights</div>",
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # NAVIGATION BUTTONS
    # --------------------------------------------------------
    with right:
        cols = st.columns(5)

        labels = [
            ("About Us", "about"),
            ("Features", "features"),
            ("How It Works", "how_it_works"),
            ("Benefits", "benefits"),
            ("Demo", "demo"),
        ]

        for i, (label, mode) in enumerate(labels):
            if cols[i].button(label, key=f"top_nav_{mode}"):
                st.session_state.app_mode = mode
                st.rerun()
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 4 / 9 — core/routing.py
# ============================================================

from __future__ import annotations

import streamlit as st

from core.session import (
    logged_in,
    is_paid_user,
)

# Marketing pages
from ui.marketing import (
    render_about,
    render_features,
    render_how_it_works,
    render_benefits,
)

# Auth & payment UI
from ui.auth import render_auth
from ui.payment import render_payment

# Demo sidebar + pages
from ui.sidebar import sidebar_nav
from ui.router import demo_router


# ============================================================
# MAIN ROUTER (SINGLE SOURCE OF TRUTH)
# ============================================================

def route_app() -> None:
    """
    Routes application based on st.session_state.app_mode.

    MODES:
    - about
    - features
    - how_it_works
    - benefits
    - demo
    """

    mode = st.session_state.get("app_mode", "about")

    # ========================================================
    # MARKETING MODE (NO SIDEBAR, NO APP UI)
    # ========================================================

    if mode != "demo":
        if mode == "about":
            render_about()
        elif mode == "features":
            render_features()
        elif mode == "how_it_works":
            render_how_it_works()
        elif mode == "benefits":
            render_benefits()
        else:
            render_about()

        return  # 🔐 STOP — no sidebar, no demo widgets

    # ========================================================
    # DEMO MODE (FULL APP)
    # ========================================================

    # ---- Authentication gate ----
    if not logged_in():
        render_auth()
        return

    # ---- Payment gate ----
    if not is_paid_user():
        render_payment()
        return

    # ---- FULL APP ENABLED ----
    page = sidebar_nav()
    demo_router(page)
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 5 / 9 — ui/sidebar.py
# ============================================================

from __future__ import annotations

import streamlit as st

from core.session import (
    logout_user,
)


# ============================================================
# DEMO SIDEBAR (ONLY ONE IN APP)
# ============================================================

def sidebar_nav() -> str:
    """
    Renders the demo sidebar and returns selected page.
    This function MUST be called exactly once per run.
    """

    with st.sidebar:
        st.markdown(
            """
            <style>
            .kwi-sidebar-title {
                font-weight: 800;
                font-size: 1.05rem;
                margin-bottom: 0.3rem;
            }
            .kwi-sidebar-section {
                margin-top: 1.2rem;
                font-weight: 700;
                font-size: 0.85rem;
                color: #6b7280;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='kwi-sidebar-title'>📂 Navigation</div>",
                    unsafe_allow_html=True)

        # ----------------------------------------------------
        # PRIMARY NAVIGATION
        # ----------------------------------------------------

        pages = [
            "Portfolio Overview",
            "Portfolio Insights",
            "Goal Probability",
            "Portfolio Health",
            "AI Rebalancing",
            "Income Forecast",
            "Teen Explainer",
            "AI Chatbot",
            "Risk Alerts",
            "Tax Optimization",
            "Performance",
            "Exports",
        ]

        page = st.radio(
            label="",
            options=pages,
            index=0,
            key="sidebar_nav_radio_unique",  # 🔐 UNIQUE KEY
        )

        # ----------------------------------------------------
        # ACCOUNT ACTIONS
        # ----------------------------------------------------

        st.markdown("<div class='kwi-sidebar-section'>Account</div>",
                    unsafe_allow_html=True)

        if st.button("🚪 Log out", key="sidebar_logout_btn"):
            logout_user()
            st.session_state.app_mode = "about"
            st.rerun()

        return page
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 6 / 9 — ui/router.py
# ============================================================

from __future__ import annotations

import streamlit as st

# Portfolio & analytics
from features.portfolio import (
    render_portfolio_overview,
    render_portfolio_insights,
)

# Goals & simulations
from features.goals import (
    render_goal_probability,
)

# AI features
from features.ai import (
    render_portfolio_health_ai,
    render_ai_rebalancing,
    render_income_forecast_ai,
    render_teen_explainer_ai,
    render_ai_chatbot,
)

# Risk, tax, performance, exports
from features.risk import (
    render_risk_alerts,
    render_tax_optimization,
    render_performance_benchmark,
    render_exports,
)


# ============================================================
# DEMO ROUTER
# ============================================================

def demo_router(page: str) -> None:
    """
    Routes sidebar page selection to feature renderers.
    This function does not create widgets itself.
    """

    if page == "Portfolio Overview":
        render_portfolio_overview()

    elif page == "Portfolio Insights":
        render_portfolio_insights()

    elif page == "Goal Probability":
        render_goal_probability()

    elif page == "Portfolio Health":
        render_portfolio_health_ai()

    elif page == "AI Rebalancing":
        render_ai_rebalancing()

    elif page == "Income Forecast":
        render_income_forecast_ai()

    elif page == "Teen Explainer":
        render_teen_explainer_ai()

    elif page == "AI Chatbot":
        render_ai_chatbot()

    elif page == "Risk Alerts":
        render_risk_alerts()

    elif page == "Tax Optimization":
        render_tax_optimization()

    elif page == "Performance":
        render_performance_benchmark()

    elif page == "Exports":
        render_exports()

    else:
        st.info("Select a feature from the sidebar.")
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 7 / 9 — features/portfolio.py
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None


# ============================================================
# UI HELPERS
# ============================================================

def page_header(title: str, subtitle: str | None = None, icon: str = ""):
    st.markdown(
        f"""
        <div style="padding-bottom:1rem;">
            <h2>{icon} {title}</h2>
            <p style="color:#6b7280;">{subtitle or ""}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider():
    st.markdown("---")


# ============================================================
# PORTFOLIO INPUT MODEL
# ============================================================

REQUIRED_COLUMNS = ["Ticker", "Shares", "Cost_Basis"]


def portfolio_template_csv() -> bytes:
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "NVDA", "VOO"],
            "Shares": [10, 5, 3, 2],
            "Cost_Basis": [150, 280, 400, 350],
        }
    )
    return df.to_csv(index=False).encode("utf-8")


def validate_portfolio(df: pd.DataFrame):
    if df.empty:
        return False, "Portfolio is empty."

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"

    try:
        df["Shares"] = pd.to_numeric(df["Shares"])
        df["Cost_Basis"] = pd.to_numeric(df["Cost_Basis"])
    except Exception:
        return False, "Shares and Cost_Basis must be numeric."

    if (df["Shares"] <= 0).any():
        return False, "Shares must be positive."

    return True, "OK"


# ============================================================
# LIVE MARKET DATA
# ============================================================

@st.cache_data(ttl=60)
def get_live_price(ticker: str):
    if yf is None:
        return None, None

    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="1m")
        if hist.empty:
            return None, None

        price = float(hist["Close"].iloc[-1])
        change = price - float(hist["Open"].iloc[0])
        return round(price, 2), round(change, 2)
    except Exception:
        return None, None


# ============================================================
# PORTFOLIO CALCULATION
# ============================================================

def compute_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in df.iterrows():
        ticker = str(r["Ticker"]).upper()
        shares = float(r["Shares"])
        cost = float(r["Cost_Basis"])

        price, _ = get_live_price(ticker)
        if price is None:
            continue

        mv = shares * price
        pnl = mv - (shares * cost)

        rows.append(
            {
                "Ticker": ticker,
                "Shares": shares,
                "Live Price": price,
                "Market Value": round(mv, 2),
                "Cost Basis": round(shares * cost, 2),
                "PnL": round(pnl, 2),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out.loc["TOTAL", "Market Value"] = out["Market Value"].sum()
    out.loc["TOTAL", "PnL"] = out["PnL"].sum()

    return out


# ============================================================
# ETF LOOK-THROUGH
# ============================================================

KNOWN_ETFS = {"SPY", "VOO", "VTI", "QQQ", "IVV", "DIA"}


def is_etf(ticker: str) -> bool:
    if ticker in KNOWN_ETFS:
        return True
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("quoteType") == "ETF"
    except Exception:
        return False


@st.cache_data(ttl=86400)
def get_etf_holdings(ticker: str, limit: int = 10) -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        h = getattr(t, "fund_holdings", None)
        if h is None or h.empty:
            return pd.DataFrame()

        df = h[["symbol", "holdingPercent"]].dropna()
        df.columns = ["Ticker", "Weight"]
        df["Weight"] = df["Weight"] / 100
        return df.head(limit)
    except Exception:
        return pd.DataFrame()


def compute_lookthrough(portfolio: pd.DataFrame):
    stocks, sectors = [], []
    total_value = portfolio.loc["TOTAL", "Market Value"]

    for _, r in portfolio.drop(index="TOTAL").iterrows():
        ticker = r["Ticker"]
        weight = r["Market Value"] / total_value

        if is_etf(ticker):
            holdings = get_etf_holdings(ticker)
            for _, h in holdings.iterrows():
                stocks.append(
                    {"Ticker": h["Ticker"], "Weight": weight * h["Weight"]}
                )
        else:
            stocks.append({"Ticker": ticker, "Weight": weight})

    stock_df = pd.DataFrame(stocks).groupby("Ticker").sum().reset_index()
    stock_df["Weight %"] = (stock_df["Weight"] * 100).round(2)

    return stock_df.sort_values("Weight %", ascending=False)


# ============================================================
# DIVIDENDS
# ============================================================

@st.cache_data(ttl=3600)
def get_dividends(ticker: str):
    try:
        d = yf.Ticker(ticker).dividends
        if d is None or d.empty:
            return 0.0
        last_year = d[d.index >= (pd.Timestamp.now() - pd.DateOffset(years=1))]
        return float(last_year.sum())
    except Exception:
        return 0.0


def compute_dividend_table(portfolio: pd.DataFrame):
    rows = []

    for _, r in portfolio.drop(index="TOTAL").iterrows():
        div = get_dividends(r["Ticker"])
        income = div * r["Shares"]
        yield_pct = (div / r["Live Price"]) * 100 if r["Live Price"] > 0 else 0

        rows.append(
            {
                "Ticker": r["Ticker"],
                "Annual Dividend / Share": round(div, 2),
                "Yield %": round(yield_pct, 2),
                "Annual Income": round(income, 2),
            }
        )

    df = pd.DataFrame(rows)
    df.loc["TOTAL", "Annual Income"] = df["Annual Income"].sum()
    return df


# ============================================================
# PAGES
# ============================================================

def render_portfolio_overview():
    page_header("Portfolio Overview", "Upload once. Everything flows from here.", "📊")

    st.download_button(
        "Download CSV Template",
        portfolio_template_csv(),
        file_name="portfolio_template.csv",
    )

    uploaded = st.file_uploader(
        "Upload Portfolio CSV",
        type="csv",
        key="portfolio_upload_main",
    )

    if not uploaded:
        st.info("Upload a portfolio to continue.")
        return

    raw = pd.read_csv(uploaded)
    ok, msg = validate_portfolio(raw)
    if not ok:
        st.error(msg)
        return

    portfolio = compute_portfolio(raw)
    if portfolio.empty:
        st.warning("No valid holdings.")
        return

    st.session_state.portfolio_raw = raw
    st.session_state.portfolio = portfolio

    divider()

    st.dataframe(portfolio, use_container_width=True)

    income = compute_dividend_table(portfolio)
    st.session_state.portfolio_meta["income"] = income.to_dict()

    divider()

    st.subheader("Dividend Income")
    st.dataframe(income, use_container_width=True)


def render_portfolio_insights():
    page_header("Portfolio Insights", "Allocation and look-through exposure", "🧬")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    exposure = compute_lookthrough(portfolio)
    st.subheader("Look-Through Stock Exposure")
    st.dataframe(exposure.head(15), use_container_width=True)
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 8 / 9 — features/goals.py
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None


# ============================================================
# UI HELPERS
# ============================================================

def page_header(title: str, subtitle: str | None = None, icon: str = ""):
    st.markdown(
        f"""
        <div style="padding-bottom:1rem;">
            <h2>{icon} {title}</h2>
            <p style="color:#6b7280;">{subtitle or ""}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider():
    st.markdown("---")


# ============================================================
# PORTFOLIO RETURN ESTIMATION
# ============================================================

@st.cache_data(ttl=3600)
def estimate_portfolio_stats(portfolio: pd.DataFrame):
    """
    Estimate expected annual return and volatility
    using historical daily returns (3Y lookback).
    """
    if yf is None or portfolio is None or portfolio.empty:
        return 0.07, 0.15  # conservative defaults

    returns = []

    for _, r in portfolio.drop(index="TOTAL", errors="ignore").iterrows():
        try:
            prices = yf.Ticker(r["Ticker"]).history(period="3y")["Close"]
            daily = prices.pct_change().dropna()
            if not daily.empty:
                returns.append(daily)
        except Exception:
            continue

    if not returns:
        return 0.07, 0.15

    combined = pd.concat(returns, axis=1).mean(axis=1)

    exp_return = float(combined.mean() * 252)
    volatility = float(combined.std() * np.sqrt(252))

    return round(exp_return, 4), round(volatility, 4)


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

def monte_carlo_simulation(
    start_value: float,
    annual_contribution: float,
    years: int,
    expected_return: float,
    volatility: float,
    simulations: int = 3000,
):
    """
    Monte Carlo simulation of portfolio value paths.
    """
    results = np.zeros((simulations, years))

    for i in range(simulations):
        value = start_value
        for y in range(years):
            shock = np.random.normal(expected_return, volatility)
            value = value * (1 + shock) + annual_contribution
            results[i, y] = value

    return results


def goal_success_probability(simulations: np.ndarray, goal: float) -> float:
    final_values = simulations[:, -1]
    return round((final_values >= goal).mean() * 100, 2)


# ============================================================
# VISUAL HELPERS
# ============================================================

def plot_simulation_paths(simulations: np.ndarray, max_paths: int = 50):
    df = pd.DataFrame(simulations.T)
    st.line_chart(df.sample(min(max_paths, df.shape[1]), axis=1))


# ============================================================
# GOAL PROBABILITY PAGE
# ============================================================

def render_goal_probability():
    page_header(
        "Goal Probability",
        "Estimate the likelihood of reaching your long-term goal",
        "🎯",
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
        expected_return=exp_return,
        volatility=volatility,
    )

    probability = goal_success_probability(sims, goal)

    st.metric("Goal Success Probability", f"{probability}%")
    st.metric("Expected Return", f"{round(exp_return * 100, 2)}%")
    st.metric("Volatility", f"{round(volatility * 100, 2)}%")

    divider()

    st.subheader("Outcome Distribution")

    final_vals = sims[:, -1]

    summary = pd.DataFrame({
        "Scenario": ["Pessimistic (10%)", "Median (50%)", "Optimistic (90%)"],
        "Ending Value ($)": [
            round(np.percentile(final_vals, 10), 0),
            round(np.percentile(final_vals, 50), 0),
            round(np.percentile(final_vals, 90), 0),
        ],
    })

    st.dataframe(summary, use_container_width=True)

    divider()

    st.subheader("Monte Carlo Paths")
    plot_simulation_paths(sims)

    # Persist for AI context
    st.session_state.portfolio_meta["goal_probability"] = {
        "goal": goal,
        "years": years,
        "probability": probability,
        "expected_return": exp_return,
        "volatility": volatility,
    }
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 8 / 9 — features/goals.py
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None


# ============================================================
# UI HELPERS
# ============================================================

def page_header(title: str, subtitle: str | None = None, icon: str = ""):
    st.markdown(
        f"""
        <div style="padding-bottom:1rem;">
            <h2>{icon} {title}</h2>
            <p style="color:#6b7280;">{subtitle or ""}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider():
    st.markdown("---")


# ============================================================
# PORTFOLIO RETURN ESTIMATION
# ============================================================

@st.cache_data(ttl=3600)
def estimate_portfolio_stats(portfolio: pd.DataFrame):
    """
    Estimate expected annual return and volatility
    using historical daily returns (3Y lookback).
    """
    if yf is None or portfolio is None or portfolio.empty:
        return 0.07, 0.15  # conservative defaults

    returns = []

    for _, r in portfolio.drop(index="TOTAL", errors="ignore").iterrows():
        try:
            prices = yf.Ticker(r["Ticker"]).history(period="3y")["Close"]
            daily = prices.pct_change().dropna()
            if not daily.empty:
                returns.append(daily)
        except Exception:
            continue

    if not returns:
        return 0.07, 0.15

    combined = pd.concat(returns, axis=1).mean(axis=1)

    exp_return = float(combined.mean() * 252)
    volatility = float(combined.std() * np.sqrt(252))

    return round(exp_return, 4), round(volatility, 4)


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

def monte_carlo_simulation(
    start_value: float,
    annual_contribution: float,
    years: int,
    expected_return: float,
    volatility: float,
    simulations: int = 3000,
):
    """
    Monte Carlo simulation of portfolio value paths.
    """
    results = np.zeros((simulations, years))

    for i in range(simulations):
        value = start_value
        for y in range(years):
            shock = np.random.normal(expected_return, volatility)
            value = value * (1 + shock) + annual_contribution
            results[i, y] = value

    return results


def goal_success_probability(simulations: np.ndarray, goal: float) -> float:
    final_values = simulations[:, -1]
    return round((final_values >= goal).mean() * 100, 2)


# ============================================================
# VISUAL HELPERS
# ============================================================

def plot_simulation_paths(simulations: np.ndarray, max_paths: int = 50):
    df = pd.DataFrame(simulations.T)
    st.line_chart(df.sample(min(max_paths, df.shape[1]), axis=1))


# ============================================================
# GOAL PROBABILITY PAGE
# ============================================================

def render_goal_probability():
    page_header(
        "Goal Probability",
        "Estimate the likelihood of reaching your long-term goal",
        "🎯",
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
        expected_return=exp_return,
        volatility=volatility,
    )

    probability = goal_success_probability(sims, goal)

    st.metric("Goal Success Probability", f"{probability}%")
    st.metric("Expected Return", f"{round(exp_return * 100, 2)}%")
    st.metric("Volatility", f"{round(volatility * 100, 2)}%")

    divider()

    st.subheader("Outcome Distribution")

    final_vals = sims[:, -1]

    summary = pd.DataFrame({
        "Scenario": ["Pessimistic (10%)", "Median (50%)", "Optimistic (90%)"],
        "Ending Value ($)": [
            round(np.percentile(final_vals, 10), 0),
            round(np.percentile(final_vals, 50), 0),
            round(np.percentile(final_vals, 90), 0),
        ],
    })

    st.dataframe(summary, use_container_width=True)

    divider()

    st.subheader("Monte Carlo Paths")
    plot_simulation_paths(sims)

    # Persist for AI context
    st.session_state.portfolio_meta["goal_probability"] = {
        "goal": goal,
        "years": years,
        "probability": probability,
        "expected_return": exp_return,
        "volatility": volatility,
    }
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 10 / 11 — ui/auth.py
# ============================================================

from __future__ import annotations

import streamlit as st

from core.session import (
    hash_pw,
    db_get_user,
    db_create_user,
    login_user,
)


# ============================================================
# AUTH UI
# ============================================================

def render_auth() -> None:
    """
    Login / Signup UI.
    No routing, no sidebar, no redirects here.
    """

    st.markdown("## 🔐 Sign in to Katta Wealth Insights")

    login_tab, signup_tab = st.tabs(["Log In", "Create Account"])

    # ----------------------------
    # LOGIN
    # ----------------------------
    with login_tab:
        email = st.text_input("Email", key="auth_login_email")
        pw = st.text_input("Password", type="password", key="auth_login_pw")
        remember = st.checkbox("Remember me", value=True)

        if st.button("Log In", key="auth_login_btn", use_container_width=True):
            if not email or not pw:
                st.error("Please enter email and password.")
                return

            user = db_get_user(email)
            if not user or user["pw"] != hash_pw(pw):
                st.error("Invalid credentials.")
                return

            login_user(email, remember=remember)
            st.success("Logged in successfully.")
            st.rerun()

    # ----------------------------
    # SIGNUP
    # ----------------------------
    with signup_tab:
        email = st.text_input("Email", key="auth_signup_email")
        pw = st.text_input("Password", type="password", key="auth_signup_pw")

        if st.button("Create Account", key="auth_signup_btn", use_container_width=True):
            if not email or not pw:
                st.error("Please enter email and password.")
                return

            created = db_create_user(email, hash_pw(pw))
            if not created:
                st.error("Account already exists.")
                return

            st.success("Account created. Please log in.")
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 11 / 11 — ui/payment.py
# ============================================================

from __future__ import annotations

import streamlit as st

from core.session import (
    mark_user_as_paid,
)


# ============================================================
# PAYMENT UI (DEMO + STRIPE-READY)
# ============================================================

def render_payment() -> None:
    """
    Paywall screen.
    Blocks demo features until user upgrades.
    """

    st.markdown("## 💎 Upgrade to Katta Wealth Pro")

    st.markdown(
        """
        ### What you unlock:
        - 📊 Full portfolio analytics
        - 🧬 ETF look-through exposure
        - 💵 Dividend income engine
        - 🎯 Monte Carlo goal probability
        - 🧠 AI explanations & chatbot
        - 🚨 Risk & drawdown alerts
        - 📤 Exportable reports
        """
    )

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            **Pro Plan**
            - $24 / month  
            - Cancel anytime  
            - Educational, no advice
            """
        )

    with col2:
        if st.button("💳 Upgrade Now (Demo)", key="paywall_upgrade_btn"):
            # Demo-only unlock
            mark_user_as_paid()
            st.success("Welcome to Pro 🎉")
            st.rerun()

    st.caption(
        "⚠️ This is a demo paywall. Stripe Checkout can be enabled later "
        "with no architectural changes."
    )
