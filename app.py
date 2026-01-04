# ============================================================
# KATTA WEALTH INSIGHTS — APP ENTRYPOINT
# FILE 1 / 9 — app.py
# ============================================================

from __future__ import annotations

import streamlit as st

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
    Single authoritative entrypoint.
    - No widgets here
    - No business logic
    """

    # Initialize session state once
    init_session()

    # Always render global navigation
    render_top_nav()

    # Route based on app mode
    route_app()


# ============================================================
# EXECUTION GUARD (ONLY ONE IN CODEBASE)
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

PAID_PLAN_NAME = "Pro"
PAID_PRICE_MONTHLY = 24


# ============================================================
# SESSION INITIALIZATION (AUTHORITATIVE)
# ============================================================

def init_session() -> None:
    """
    Initialize ALL session keys exactly once.
    Idempotent and Streamlit-safe.
    """

    defaults = {
        # App mode
        "app_mode": "about",  # about | features | how_it_works | benefits | demo

        # Auth
        "current_user": None,
        "is_paid": False,

        # Portfolio state
        "portfolio_raw": None,
        "portfolio": None,
        "portfolio_meta": {},

        # AI
        "chat_history": [],

        # UI guards
        "_top_nav_rendered": False,

        # Debug
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
# PAYMENT STUB (STRIPE-READY)
# ============================================================

def mark_user_as_paid() -> None:
    """
    Demo-only payment unlock.
    Stripe Checkout can replace this without refactor.
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
    Renders the global top navigation bar exactly once.
    This is UI-only and stateless.
    """

    # Prevent double-render on reruns
    if st.session_state.get("_top_nav_rendered"):
        return
    st.session_state["_top_nav_rendered"] = True

    st.markdown(
        """
        <style>
        .kwi-top-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.8rem 2rem;
            border-bottom: 1px solid #222;
            background-color: #0f172a;
        }
        .kwi-brand {
            font-size: 1.4rem;
            font-weight: 800;
            color: #e5e7eb;
        }
        .kwi-nav button {
            margin-left: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns([3, 7])

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------
    with cols[0]:
        st.markdown("<div class='kwi-brand'>💎 Katta Wealth Insights</div>",
                    unsafe_allow_html=True)

    # --------------------------------------------------------
    # NAVIGATION BUTTONS
    # --------------------------------------------------------
    with cols[1]:
        nav_cols = st.columns(5)
        labels = [
            ("About", "about"),
            ("Features", "features"),
            ("How It Works", "how_it_works"),
            ("Benefits", "benefits"),
            ("Demo", "demo"),
        ]

        for i, (label, mode) in enumerate(labels):
            if nav_cols[i].button(
                label,
                key=f"topnav_{mode}",
                use_container_width=True,
            ):
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

# Auth & payment
from ui.auth import render_auth
from ui.payment import render_payment

# Demo UI
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
    # MARKETING MODE (NO SIDEBAR, NO DEMO UI)
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

        # 🔒 HARD STOP — prevents sidebar bleed
        return

    # ========================================================
    # DEMO MODE (FULL APP)
    # ========================================================

    # ---- Authentication Gate ----
    if not logged_in():
        render_auth()
        return

    # ---- Payment Gate ----
    if not is_paid_user():
        render_payment()
        return

    # ---- Full App Enabled ----
    page = sidebar_nav()
    demo_router(page)
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 5 / 9 — ui/auth.py
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
    - No routing
    - No sidebar
    - No redirects except rerun
    """

    st.markdown("## 🔐 Sign in to Katta Wealth Insights")

    login_tab, signup_tab = st.tabs(["Log In", "Create Account"])

    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------
    with login_tab:
        email = st.text_input(
            "Email",
            key="auth_login_email",
            placeholder="you@example.com",
        )
        pw = st.text_input(
            "Password",
            type="password",
            key="auth_login_pw",
        )
        remember = st.checkbox(
            "Remember me",
            value=True,
            key="auth_login_remember",
        )

        if st.button(
            "Log In",
            key="auth_login_btn",
            use_container_width=True,
        ):
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

    # --------------------------------------------------------
    # SIGNUP
    # --------------------------------------------------------
    with signup_tab:
        email = st.text_input(
            "Email",
            key="auth_signup_email",
            placeholder="you@example.com",
        )
        pw = st.text_input(
            "Password",
            type="password",
            key="auth_signup_pw",
        )

        if st.button(
            "Create Account",
            key="auth_signup_btn",
            use_container_width=True,
        ):
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
# FILE 6 / 9 — ui/payment.py
# ============================================================

from __future__ import annotations

import streamlit as st

from core.session import mark_user_as_paid


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
            - Educational use only  
            """
        )

    with col2:
        if st.button(
            "💳 Upgrade Now (Demo)",
            key="paywall_upgrade_btn",
            use_container_width=True,
        ):
            # Demo-only unlock
            mark_user_as_paid()
            st.success("Welcome to Pro 🎉")
            st.rerun()

    st.caption(
        "⚠️ This is a demo paywall. Stripe Checkout can be enabled later "
        "with no architectural changes."
    )
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 7 / 9 — ui/sidebar.py
# ============================================================

from __future__ import annotations

import streamlit as st

from core.session import logout_user


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
                margin-bottom: 0.4rem;
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

        # ----------------------------------------------------
        # NAV TITLE
        # ----------------------------------------------------
        st.markdown(
            "<div class='kwi-sidebar-title'>📂 Navigation</div>",
            unsafe_allow_html=True,
        )

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
            key="sidebar_nav_radio_unique",
        )

        # ----------------------------------------------------
        # ACCOUNT ACTIONS
        # ----------------------------------------------------
        st.markdown(
            "<div class='kwi-sidebar-section'>Account</div>",
            unsafe_allow_html=True,
        )

        if st.button(
            "🚪 Log out",
            key="sidebar_logout_btn",
            use_container_width=True,
        ):
            logout_user()
            st.session_state.app_mode = "about"
            st.rerun()

        return page
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 8 / 9 — ui/router.py
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

# AI-driven features
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
# FILE 9 / 9 — features/portfolio.py
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

def page_header(title: str, subtitle: str | None = None, icon: str = "") -> None:
    st.markdown(
        f"""
        <div style="padding-bottom:1rem;">
            <h2>{icon} {title}</h2>
            <p style="color:#6b7280;">{subtitle or ""}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider() -> None:
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


def validate_portfolio(df: pd.DataFrame) -> tuple[bool, str]:
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
def get_live_price(ticker: str) -> tuple[float | None, float | None]:
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
        ticker = str(r["Ticker"]).upper().strip()
        shares = float(r["Shares"])
        cost = float(r["Cost_Basis"])

        price, _ = get_live_price(ticker)
        if price is None:
            continue

        market_value = shares * price
        pnl = market_value - (shares * cost)

        rows.append(
            {
                "Ticker": ticker,
                "Shares": shares,
                "Live Price": price,
                "Market Value": round(market_value, 2),
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

    if yf is None:
        return False

    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("quoteType") == "ETF"
    except Exception:
        return False


@st.cache_data(ttl=86400)
def get_etf_holdings(ticker: str, limit: int = 10) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()

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


def compute_lookthrough(portfolio: pd.DataFrame) -> pd.DataFrame:
    stocks = []
    total_value = portfolio.loc["TOTAL", "Market Value"]

    for _, r in portfolio.drop(index="TOTAL").iterrows():
        ticker = r["Ticker"]
        weight = r["Market Value"] / total_value

        if is_etf(ticker):
            holdings = get_etf_holdings(ticker)
            for _, h in holdings.iterrows():
                stocks.append(
                    {
                        "Ticker": h["Ticker"],
                        "Weight": weight * h["Weight"],
                    }
                )
        else:
            stocks.append({"Ticker": ticker, "Weight": weight})

    df = pd.DataFrame(stocks).groupby("Ticker").sum().reset_index()
    df["Weight %"] = (df["Weight"] * 100).round(2)

    return df.sort_values("Weight %", ascending=False)


# ============================================================
# DIVIDENDS
# ============================================================

@st.cache_data(ttl=3600)
def get_dividends(ticker: str) -> float:
    if yf is None:
        return 0.0

    try:
        d = yf.Ticker(ticker).dividends
        if d is None or d.empty:
            return 0.0

        last_year = d[d.index >= (pd.Timestamp.now() - pd.DateOffset(years=1))]
        return float(last_year.sum())
    except Exception:
        return 0.0


def compute_dividend_table(portfolio: pd.DataFrame) -> pd.DataFrame:
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

def render_portfolio_overview() -> None:
    page_header(
        "Portfolio Overview",
        "Upload once. Everything flows from here.",
        "📊",
    )

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


def render_portfolio_insights() -> None:
    page_header(
        "Portfolio Insights",
        "Allocation and look-through exposure",
        "🧬",
    )

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    exposure = compute_lookthrough(portfolio)

    st.subheader("Look-Through Stock Exposure")
    st.dataframe(exposure.head(15), use_container_width=True)
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 10 / 9 — features/goals.py
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

def page_header(title: str, subtitle: str | None = None, icon: str = "") -> None:
    st.markdown(
        f"""
        <div style="padding-bottom:1rem;">
            <h2>{icon} {title}</h2>
            <p style="color:#6b7280;">{subtitle or ""}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown("---")


# ============================================================
# PORTFOLIO RETURN ESTIMATION
# ============================================================

@st.cache_data(ttl=3600)
def estimate_portfolio_stats(
    portfolio: pd.DataFrame,
) -> tuple[float, float]:
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
) -> np.ndarray:
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


def goal_success_probability(
    simulations: np.ndarray,
    goal: float,
) -> float:
    final_values = simulations[:, -1]
    return round((final_values >= goal).mean() * 100, 2)


# ============================================================
# VISUAL HELPERS
# ============================================================

def plot_simulation_paths(
    simulations: np.ndarray,
    max_paths: int = 50,
) -> None:
    df = pd.DataFrame(simulations.T)
    st.line_chart(
        df.sample(
            min(max_paths, df.shape[1]),
            axis=1,
        )
    )


# ============================================================
# GOAL PROBABILITY PAGE
# ============================================================

def render_goal_probability() -> None:
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

    summary = pd.DataFrame(
        {
            "Scenario": [
                "Pessimistic (10%)",
                "Median (50%)",
                "Optimistic (90%)",
            ],
            "Ending Value ($)": [
                round(np.percentile(final_vals, 10), 0),
                round(np.percentile(final_vals, 50), 0),
                round(np.percentile(final_vals, 90), 0),
            ],
        }
    )

    st.dataframe(summary, use_container_width=True)

    divider()

    st.subheader("Monte Carlo Paths")
    plot_simulation_paths(sims)

    # Persist for AI / context usage
    st.session_state.portfolio_meta["goal_probability"] = {
        "goal": goal,
        "years": years,
        "probability": probability,
        "expected_return": exp_return,
        "volatility": volatility,
    }
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 11 / 9 — features/ai.py
# ============================================================

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np


# ============================================================
# UI HELPERS
# ============================================================

def page_header(title: str, subtitle: str | None = None, icon: str = "") -> None:
    st.markdown(
        f"""
        <div style="padding-bottom:1rem;">
            <h2>{icon} {title}</h2>
            <p style="color:#6b7280;">{subtitle or ""}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown("---")


# ============================================================
# PORTFOLIO HEALTH AI
# ============================================================

def render_portfolio_health_ai() -> None:
    page_header(
        "Portfolio Health",
        "AI-style diagnostics of diversification and risk",
        "🧠",
    )

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    holdings = portfolio.drop(index="TOTAL")
    weights = holdings["Market Value"] / holdings["Market Value"].sum()

    max_weight = weights.max()
    concentration = (weights ** 2).sum()

    st.subheader("Key Diagnostics")

    c1, c2 = st.columns(2)
    c1.metric("Largest Holding Weight", f"{round(max_weight * 100, 1)}%")
    c2.metric("Concentration Index", round(concentration, 3))

    divider()

    st.subheader("AI Assessment")

    if max_weight > 0.4:
        st.warning(
            "⚠️ Your portfolio is highly concentrated in a single holding. "
            "This increases volatility and downside risk."
        )
    elif concentration > 0.25:
        st.info(
            "ℹ️ Your portfolio has moderate concentration. "
            "Consider additional diversification."
        )
    else:
        st.success(
            "✅ Your portfolio appears well diversified based on current holdings."
        )


# ============================================================
# AI REBALANCING (RULE-BASED)
# ============================================================

def render_ai_rebalancing() -> None:
    page_header(
        "AI Rebalancing",
        "Suggested actions to improve diversification",
        "⚖️",
    )

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    holdings = portfolio.drop(index="TOTAL")
    weights = holdings["Market Value"] / holdings["Market Value"].sum()

    suggestions = []

    for ticker, weight in zip(holdings["Ticker"], weights):
        if weight > 0.35:
            suggestions.append(
                f"Consider trimming **{ticker}** to reduce concentration risk."
            )

    if not suggestions:
        suggestions.append(
            "Your portfolio weights look balanced. No immediate rebalancing needed."
        )

    st.subheader("Suggested Actions")
    for s in suggestions:
        st.markdown(f"- {s}")


# ============================================================
# INCOME FORECAST AI
# ============================================================

def render_income_forecast_ai() -> None:
    page_header(
        "Income Forecast",
        "Projected dividend income growth",
        "💵",
    )

    income = st.session_state.get("portfolio_meta", {}).get("income")
    if not income:
        st.info("Upload a dividend-paying portfolio first.")
        return

    df = pd.DataFrame(income).T
    annual_income = df.loc["TOTAL", "Annual Income"]

    growth = st.slider("Assumed Annual Growth (%)", 0.0, 10.0, 4.0)

    years = list(range(1, 11))
    projected = [
        round(annual_income * ((1 + growth / 100) ** y), 2)
        for y in years
    ]

    forecast = pd.DataFrame(
        {
            "Year": years,
            "Projected Income ($)": projected,
        }
    )

    st.metric("Current Annual Income", f"${round(annual_income, 2)}")
    st.line_chart(forecast.set_index("Year"))


# ============================================================
# TEEN EXPLAINER AI
# ============================================================

def render_teen_explainer_ai() -> None:
    page_header(
        "Teen Explainer",
        "Simple explanations for students and beginners",
        "🎒",
    )

    st.markdown(
        """
        ### What is a portfolio?
        A **portfolio** is a collection of investments like stocks and ETFs.

        ### Why diversify?
        Diversification reduces risk by spreading money across many investments.

        ### What is risk?
        Risk is how much your investment value can go up **or down**.

        ### Long-term investing
        Time + consistency is usually more important than timing the market.
        """
    )


# ============================================================
# AI CHATBOT (RULE-BASED STUB)
# ============================================================

def render_ai_chatbot() -> None:
    page_header(
        "AI Chatbot",
        "Ask questions about your portfolio",
        "🤖",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question")

    if st.button("Ask"):
        response = _generate_response(user_input)
        st.session_state.chat_history.append(
            {"q": user_input, "a": response}
        )

    divider()

    for msg in reversed(st.session_state.chat_history[-5:]):
        st.markdown(f"**You:** {msg['q']}")
        st.markdown(f"**AI:** {msg['a']}")
        st.markdown("---")


def _generate_response(question: str) -> str:
    q = question.lower()

    if "risk" in q:
        return "Risk refers to how much your investment value can fluctuate over time."
    if "divers" in q:
        return "Diversification means spreading investments to reduce risk."
    if "goal" in q:
        return "Your goal probability shows how likely you are to reach your target based on assumptions."

    return "I can help explain portfolio concepts, risk, diversification, and goals."
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 12 / 9 — features/risk.py
# ============================================================

from __future__ import annotations

import io
import pandas as pd
import numpy as np
import streamlit as st


# ============================================================
# UI HELPERS
# ============================================================

def page_header(title: str, subtitle: str | None = None, icon: str = "") -> None:
    st.markdown(
        f"""
        <div style="padding-bottom:1rem;">
            <h2>{icon} {title}</h2>
            <p style="color:#6b7280;">{subtitle or ""}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown("---")


# ============================================================
# RISK ALERTS
# ============================================================

def render_risk_alerts() -> None:
    page_header(
        "Risk Alerts",
        "Identify concentration and downside risks",
        "🚨",
    )

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    holdings = portfolio.drop(index="TOTAL")
    weights = holdings["Market Value"] / holdings["Market Value"].sum()

    alerts = []

    for ticker, w in zip(holdings["Ticker"], weights):
        if w > 0.4:
            alerts.append(
                f"⚠️ **{ticker}** exceeds 40% of portfolio value."
            )

    if not alerts:
        st.success("✅ No major concentration risks detected.")
    else:
        for a in alerts:
            st.warning(a)


# ============================================================
# TAX OPTIMIZATION
# ============================================================

def render_tax_optimization() -> None:
    page_header(
        "Tax Optimization",
        "Identify potential tax-loss harvesting opportunities",
        "🧾",
    )

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    holdings = portfolio.drop(index="TOTAL")
    losses = holdings[holdings["PnL"] < 0]

    if losses.empty:
        st.success("✅ No unrealized losses detected.")
        return

    st.subheader("Potential Tax-Loss Harvesting")
    st.dataframe(
        losses[["Ticker", "PnL"]],
        use_container_width=True,
    )


# ============================================================
# PERFORMANCE BENCHMARKING
# ============================================================

def render_performance_benchmark() -> None:
    page_header(
        "Performance",
        "Portfolio performance snapshot",
        "📈",
    )

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    total_value = portfolio.loc["TOTAL", "Market Value"]
    total_pnl = portfolio.loc["TOTAL", "PnL"]

    st.metric("Total Portfolio Value", f"${round(total_value, 2)}")
    st.metric("Total P&L", f"${round(total_pnl, 2)}")

    divider()

    st.caption(
        "Benchmark comparison (e.g., S&P 500) can be added later "
        "using historical return data."
    )


# ============================================================
# EXPORTS
# ============================================================

def render_exports() -> None:
    page_header(
        "Exports",
        "Download portfolio reports",
        "📤",
    )

    portfolio = st.session_state.get("portfolio")
    if portfolio is None or portfolio.empty:
        st.info("Upload a portfolio first.")
        return

    buffer = io.StringIO()
    portfolio.to_csv(buffer)
    csv_bytes = buffer.getvalue().encode("utf-8")

    st.download_button(
        label="Download Portfolio CSV",
        data=csv_bytes,
        file_name="portfolio_report.csv",
        mime="text/csv",
    )

    st.caption(
        "PDF and client-ready reports can be added later "
        "without changing this interface."
    )
# ============================================================
# KATTA WEALTH INSIGHTS
# FILE 13 / 9 — ui/marketing.py
# ============================================================

from __future__ import annotations

import streamlit as st


# ============================================================
# SHARED UI HELPERS
# ============================================================

def page_header(title: str, subtitle: str, icon: str = "") -> None:
    st.markdown(
        f"""
        <div style="padding:2rem 0 1.5rem 0;">
            <h1>{icon} {title}</h1>
            <p style="font-size:1.1rem; color:#6b7280;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div style="margin-top:2rem;">
            <h3>{title}</h3>
            <p style="font-size:1rem; color:#e5e7eb;">
                {body}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ABOUT
# ============================================================

def render_about() -> None:
    page_header(
        "About Katta Wealth Insights",
        "An educational platform for understanding portfolios, risk, and long-term investing.",
        "💎",
    )

    section(
        "Our Mission",
        """
        Katta Wealth Insights was built to help individuals and students
        understand investing concepts clearly — without jargon, hype,
        or hidden incentives.
        """
    )

    section(
        "What We Are (and Aren’t)",
        """
        We are an educational analytics platform.
        We are not a broker, advisor, or trading system.
        All insights are informational and learning-focused.
        """
    )

    section(
        "Who It’s For",
        """
        • Long-term investors<br>
        • Students & teens learning finance<br>
        • Families exploring portfolio concepts together
        """
    )


# ============================================================
# FEATURES
# ============================================================

def render_features() -> None:
    page_header(
        "Features",
        "Powerful analytics explained simply.",
        "✨",
    )

    section(
        "Portfolio Analytics",
        """
        Upload your holdings once and instantly see market value,
        diversification, dividends, and exposure.
        """
    )

    section(
        "ETF Look-Through",
        """
        Understand what you actually own inside ETFs —
        not just the ticker symbol.
        """
    )

    section(
        "Goal Probability",
        """
        Monte Carlo simulations estimate the likelihood
        of reaching long-term financial goals.
        """
    )

    section(
        "AI-Style Explanations",
        """
        Plain-English insights help users understand
        risk, diversification, and rebalancing.
        """
    )


# ============================================================
# HOW IT WORKS
# ============================================================

def render_how_it_works() -> None:
    page_header(
        "How It Works",
        "A simple flow from upload to insight.",
        "🛠️",
    )

    section(
        "1. Upload Your Portfolio",
        """
        Upload a CSV of your holdings — stocks or ETFs —
        using a simple, clean template.
        """
    )

    section(
        "2. Analyze Automatically",
        """
        The platform computes live market values,
        diversification metrics, and income projections.
        """
    )

    section(
        "3. Explore Insights",
        """
        Use interactive tools to explore risk,
        goal probabilities, and portfolio health.
        """
    )

    section(
        "4. Learn & Iterate",
        """
        Designed for education and understanding —
        not day trading or speculation.
        """
    )


# ============================================================
# BENEFITS
# ============================================================

def render_benefits() -> None:
    page_header(
        "Benefits",
        "Why learners and families choose Katta Wealth Insights.",
        "🎯",
    )

    section(
        "Clarity Over Complexity",
        """
        Financial concepts are explained visually
        and in clear, everyday language.
        """
    )

    section(
        "Education-First Design",
        """
        Built for learning — especially for students
        and long-term thinkers.
        """
    )

    section(
        "Safe & Transparent",
        """
        No trading, no execution, no financial advice —
        just insights and understanding.
        """
    )

    section(
        "Future-Ready",
        """
        Designed to grow with AI explainers,
        exports, and institutional-grade analytics.
        """
    )
