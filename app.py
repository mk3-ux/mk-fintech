# ============================================================
# KATTA WEALTH INSIGHTS — SINGLE FILE STREAMLIT APP
# PART 01 / 14 — IMPORTS & STREAMLIT CONFIG
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import datetime as dt
import io
import math
import random
from typing import Dict, Any, List, Optional

# ============================================================
# STREAMLIT CONFIG (MUST BE FIRST STREAMLIT CALL)
# ============================================================

st.set_page_config(
    page_title="Katta Wealth Insights",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# ============================================================
# PART 02 / 14 — SESSION STATE (AUTHORITATIVE)
# ============================================================

def init_session_state():
    defaults = {
        "app_mode": "about",
        "current_user": None,
        "is_paid": False,
        "portfolio_raw": None,
        "portfolio": None,
        "portfolio_meta": {},
        "chat_history": [],
        "alerts": [],
        "snapshots": [],
        "learning_notes": "",
        "learning_checklist": {},
        "show_tips": True,
        "theme": "Default",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()
# ============================================================
# PART 03 / 14 — LEGAL & ABOUT (ALWAYS VISIBLE)
# ============================================================

def _legal_section(title: str, body: str):
    st.markdown(
        f"""
        <div style="margin-top:1.4rem;">
            <h4>{title}</h4>
            <p style="font-size:0.9rem; color:#d1d5db; line-height:1.6;">
                {body}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about_and_legal():
    st.markdown("## 💎 About Katta Wealth Insights")

    st.markdown(
        """
        **Katta Wealth Insights** is an education-first financial analytics platform.
        It is designed to help users understand portfolios, diversification,
        risk, and long-term investing concepts.
        """
    )

    _legal_section(
        "Educational Purpose Only",
        "This platform is strictly for educational and informational purposes."
    )

    _legal_section(
        "Not Investment Advice",
        "Nothing here constitutes financial, investment, tax, or legal advice."
    )

    _legal_section(
        "Risk Disclosure",
        "Investing involves risk, including possible loss of principal."
    )

    _legal_section(
        "Simulations Disclaimer",
        "Monte Carlo simulations are hypothetical and not guarantees."
    )

    _legal_section(
        "User Responsibility",
        "Users assume full responsibility for decisions they make."
    )

    st.caption("© Katta Wealth Insights — Educational use only")


def render_legal_expander():
    with st.expander("ℹ️ About & Legal (Always Available)", expanded=False):
        render_about_and_legal()


def render_legal_banner():
    st.markdown(
        """
        <div style="
            background:#111827;
            border-top:1px solid #374151;
            padding:0.6rem;
            font-size:0.8rem;
            text-align:center;
            color:#9ca3af;
        ">
            Educational use only · No financial advice · Investing involves risk
        </div>
        """,
        unsafe_allow_html=True,
    )
# ============================================================
# PART 04 / 14 — TOP NAVIGATION (GLOBAL)
# ============================================================

def render_top_nav():
    st.markdown(
        """
        <style>
        .kwi-topnav {
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding:0.8rem 2rem;
            border-bottom:1px solid #1f2937;
            background:#0f172a;
        }
        .kwi-brand {
            font-size:1.4rem;
            font-weight:800;
            color:#e5e7eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 7])

    with left:
        st.markdown(
            "<div class='kwi-brand'>💎 Katta Wealth Insights</div>",
            unsafe_allow_html=True,
        )

    with right:
        cols = st.columns(6)
        nav_items = [
            ("About", "about"),
            ("Features", "features"),
            ("How It Works", "how"),
            ("Benefits", "benefits"),
            ("About & Legal", "legal"),
            ("Demo", "demo"),
        ]

        for i, (label, mode) in enumerate(nav_items):
            if cols[i].button(label, use_container_width=True, key=f"nav_{mode}"):
                st.session_state.app_mode = mode
                st.rerun()
# ============================================================
# PART 05 / 14 — MARKETING PAGES
# ============================================================

def _marketing_header(title: str, subtitle: str, icon: str = ""):
    st.markdown(
        f"""
        <div style="padding:2rem 0;">
            <h1>{icon} {title}</h1>
            <p style="font-size:1.1rem;color:#9ca3af;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _marketing_section(title: str, body: str):
    st.markdown(
        f"""
        <div style="margin-top:1.6rem;">
            <h3>{title}</h3>
            <p style="color:#e5e7eb;">
                {body}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about_page():
    _marketing_header(
        "About Katta Wealth Insights",
        "An education-first platform for understanding investing.",
        "💡",
    )

    _marketing_section(
        "Our Mission",
        "To help families, students, and long-term investors understand "
        "portfolios, diversification, and risk without hype or fear."
    )

    _marketing_section(
        "What This Is",
        "An educational analytics platform. No trading. No execution. No advice."
    )

    _marketing_section(
        "Who It’s For",
        "• Families<br>• Students & teens<br>• Long-term investors"
    )


def render_features_page():
    _marketing_header(
        "Features",
        "Professional-grade analytics explained simply.",
        "✨",
    )

    _marketing_section("Portfolio Analytics", "Understand value, P&L, and diversification.")
    _marketing_section("ETF Look-Through", "See what ETFs really hold.")
    _marketing_section("Goal Probability", "Monte Carlo success estimates.")
    _marketing_section("AI-Style Insights", "Plain-English explanations.")


def render_how_page():
    _marketing_header(
        "How It Works",
        "A simple flow from upload to insight.",
        "🛠️",
    )

    _marketing_section("1. Upload", "Upload a CSV of your investments.")
    _marketing_section("2. Analyze", "We compute value, risk, and income.")
    _marketing_section("3. Explore", "Simulations and insights.")
    _marketing_section("4. Learn", "Education-first design.")


def render_benefits_page():
    _marketing_header(
        "Benefits",
        "Why learners choose Katta Wealth Insights.",
        "🎯",
    )

    _marketing_section("Clarity", "No hype or fear-driven noise.")
    _marketing_section("Safety", "No trading, leverage, or execution.")
    _marketing_section("Education", "Built for learning and long-term thinking.")
# ============================================================
# PART 06 / 14 — DATABASE + AUTH (LOGIN / SIGNUP)
# ============================================================

# ------------------------------------------------------------
# DATABASE SETUP (LOCAL SQLITE)
# ------------------------------------------------------------

DB_PATH = "kwi_users.db"


def _db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _db_init():
    conn = _db_connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            pw_hash TEXT NOT NULL,
            is_paid INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


_db_init()


# ------------------------------------------------------------
# SECURITY HELPERS
# ------------------------------------------------------------

def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat()


def _hash_pw(pw: str) -> str:
    """
    Simple salted hash for demo / education.
    (Replace with bcrypt/argon2 in production)
    """
    return hashlib.sha256(("kwi_salt_" + pw).encode()).hexdigest()


# ------------------------------------------------------------
# USER DATA ACCESS
# ------------------------------------------------------------

def db_get_user(email: str) -> Optional[Dict[str, Any]]:
    conn = _db_connect()
    row = conn.execute(
        "SELECT email, pw_hash, is_paid FROM users WHERE email=?",
        (email,),
    ).fetchone()
    conn.close()

    if not row:
        return None

    return {
        "email": row[0],
        "pw_hash": row[1],
        "is_paid": bool(row[2]),
    }


def db_create_user(email: str, pw_hash: str) -> bool:
    try:
        conn = _db_connect()
        conn.execute(
            "INSERT INTO users VALUES (?,?,?,?)",
            (email, pw_hash, 0, _now_iso()),
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def db_mark_user_paid(email: str):
    conn = _db_connect()
    conn.execute(
        "UPDATE users SET is_paid=1 WHERE email=?",
        (email,),
    )
    conn.commit()
    conn.close()


# ------------------------------------------------------------
# AUTH UI
# ------------------------------------------------------------

def render_auth_block():
    st.header("🔐 Sign in to Katta Wealth Insights")

    login_tab, signup_tab = st.tabs(
        ["Log In", "Create Account"]
    )

    # ----------------------------
    # LOGIN
    # ----------------------------
    with login_tab:
        email = st.text_input(
            "Email",
            key="auth_login_email",
        )
        pw = st.text_input(
            "Password",
            type="password",
            key="auth_login_pw",
        )

        if st.button(
            "Log In",
            use_container_width=True,
            key="auth_login_btn",
        ):
            if not email or not pw:
                st.error("Please enter email and password.")
                return

            user = db_get_user(email)
            if not user or user["pw_hash"] != _hash_pw(pw):
                st.error("Invalid credentials.")
                return

            st.session_state.current_user = email
            st.session_state.is_paid = user["is_paid"]
            st.success("Logged in successfully.")
            st.rerun()

    # ----------------------------
    # SIGN UP
    # ----------------------------
    with signup_tab:
        email = st.text_input(
            "Email",
            key="auth_signup_email",
        )
        pw = st.text_input(
            "Password",
            type="password",
            key="auth_signup_pw",
        )

        if st.button(
            "Create Account",
            use_container_width=True,
            key="auth_signup_btn",
        ):
            if not email or not pw:
                st.error("Please enter email and password.")
                return

            created = db_create_user(email, _hash_pw(pw))
            if not created:
                st.error("Account already exists.")
                return

            st.success("Account created. Please log in.")


# ------------------------------------------------------------
# AUTH + PAYWALL ENFORCEMENT (NO UI YET)
# ------------------------------------------------------------

def enforce_auth_only() -> bool:
    """
    Returns True if user is logged in.
    Renders auth UI if not.
    """
    if not st.session_state.get("current_user"):
        render_auth_block()
        return False

    return True
# ============================================================
# PART 07 / 14 — PAYWALL + PRO UPGRADE
# ============================================================

def render_paywall_block():
    st.header("💎 Upgrade to Katta Wealth Pro")

    st.markdown(
        """
        ### What you unlock:
        - 📊 Full portfolio analytics
        - 🧬 ETF look-through exposure
        - 🎯 Monte Carlo goal probability
        - 🧠 AI-style insights
        - 🚨 Risk & tax education
        - 📤 Exportable reports

        **Educational use only · No investment advice**
        """
    )

    st.markdown("---")

    left, right = st.columns([2, 1])

    with left:
        st.markdown(
            """
            **Pro Plan (Demo)**
            - $24 / month  
            - Cancel anytime  
            - Education-only access  
            """
        )

    with right:
        if st.button(
            "Upgrade (Demo Unlock)",
            use_container_width=True,
            key="paywall_upgrade_btn",
        ):
            user = st.session_state.get("current_user")
            if user:
                db_mark_user_paid(user)
                st.session_state.is_paid = True
                st.success("Pro unlocked (demo).")
                st.rerun()


def enforce_auth_and_paywall() -> bool:
    """
    Enforces:
    1. Logged-in user
    2. Paid access
    """
    if not enforce_auth_only():
        return False

    if not st.session_state.get("is_paid", False):
        render_paywall_block()
        return False

    return True
# ============================================================
# PART 08 / 14 — SIDEBAR NAVIGATION + DEMO ROUTER
# ============================================================

def render_demo_sidebar() -> str:
    """
    Left sidebar navigation for demo features.
    Returns selected page name.
    """
    with st.sidebar:
        st.markdown(
            """
            <style>
            .kwi-sidebar-title {
                font-weight:800;
                font-size:1.1rem;
                margin-bottom:0.6rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='kwi-sidebar-title'>📂 Demo Navigation</div>",
            unsafe_allow_html=True,
        )

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
            "Learning Notes",
            "Scenario Comparison",
            "Snapshots",
            "Settings",
            "Learning Checklist",
        ]

        page = st.radio(
            "",
            pages,
            key="demo_sidebar_radio",
        )

        st.markdown("---")

        if st.button(
            "🚪 Log out",
            use_container_width=True,
            key="demo_logout_btn",
        ):
            st.session_state.current_user = None
            st.session_state.is_paid = False
            st.session_state.app_mode = "about"
            st.rerun()

    return page


def demo_router(page: str):
    """
    Dispatches demo pages.
    Feature renderers are defined later.
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

    elif page == "Learning Notes":
        render_learning_notes()

    elif page == "Scenario Comparison":
        render_scenario_comparison()

    elif page == "Snapshots":
        render_portfolio_snapshots()

    elif page == "Settings":
        render_settings()

    elif page == "Learning Checklist":
        render_learning_checklist()

    else:
        st.info("Select a page from the sidebar.")
# ============================================================
# PART 09 / 14 — PORTFOLIO ENGINE
# ============================================================

REQUIRED_COLUMNS = ["Ticker", "Shares", "Cost_Basis"]

# ------------------------------------------------------------
# PORTFOLIO TEMPLATE
# ------------------------------------------------------------

def portfolio_template_csv() -> bytes:
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "NVDA", "VOO"],
            "Shares": [10, 5, 3, 2],
            "Cost_Basis": [150, 280, 400, 350],
        }
    )
    return df.to_csv(index=False).encode("utf-8")


# ------------------------------------------------------------
# PORTFOLIO VALIDATION
# ------------------------------------------------------------

def validate_portfolio(df: pd.DataFrame) -> (bool, str):
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


# ------------------------------------------------------------
# PORTFOLIO CALCULATION (OFFLINE-SAFE)
# ------------------------------------------------------------

def compute_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in df.iterrows():
        ticker = str(r["Ticker"]).upper().strip()
        shares = float(r["Shares"])
        cost = float(r["Cost_Basis"])

        live_price = cost  # demo-safe pricing
        market_value = shares * live_price
        pnl = market_value - (shares * cost)

        rows.append(
            {
                "Ticker": ticker,
                "Shares": shares,
                "Live Price": round(live_price, 2),
                "Market Value": round(market_value, 2),
                "Cost Basis": round(shares * cost, 2),
                "PnL": round(pnl, 2),
            }
        )

    out = pd.DataFrame(rows)
    out.loc["TOTAL", "Market Value"] = out["Market Value"].sum()
    out.loc["TOTAL", "PnL"] = out["PnL"].sum()
    return out


# ------------------------------------------------------------
# ETF LOOK-THROUGH MODEL (EDUCATIONAL)
# ------------------------------------------------------------

ETF_LOOKTHROUGH = {
    "VOO": [("AAPL", 0.07), ("MSFT", 0.06), ("NVDA", 0.05)],
    "SPY": [("AAPL", 0.07), ("MSFT", 0.06), ("AMZN", 0.04)],
    "QQQ": [("AAPL", 0.11), ("MSFT", 0.09), ("NVDA", 0.08)],
}

def compute_lookthrough(portfolio: pd.DataFrame) -> pd.DataFrame:
    total_value = portfolio.loc["TOTAL", "Market Value"]
    rows = []

    for _, r in portfolio.drop(index="TOTAL").iterrows():
        ticker = r["Ticker"]
        weight = r["Market Value"] / total_value

        if ticker in ETF_LOOKTHROUGH:
            for sym, w in ETF_LOOKTHROUGH[ticker]:
                rows.append({"Ticker": sym, "Weight": weight * w})
        else:
            rows.append({"Ticker": ticker, "Weight": weight})

    df = pd.DataFrame(rows)
    df = df.groupby("Ticker").sum().reset_index()
    df["Weight %"] = (df["Weight"] * 100).round(2)
    return df.sort_values("Weight %", ascending=False)


# ------------------------------------------------------------
# DIVIDEND ENGINE
# ------------------------------------------------------------

DIVIDEND_YIELDS = {
    "AAPL": 0.006,
    "MSFT": 0.007,
    "NVDA": 0.001,
    "VOO": 0.015,
    "SPY": 0.014,
    "QQQ": 0.012,
}

def compute_dividends(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in portfolio.drop(index="TOTAL").iterrows():
        y = DIVIDEND_YIELDS.get(r["Ticker"], 0.0)
        income = r["Market Value"] * y
        rows.append(
            {
                "Ticker": r["Ticker"],
                "Yield %": round(y * 100, 2),
                "Annual Income": round(income, 2),
            }
        )

    df = pd.DataFrame(rows)
    df.loc["TOTAL", "Annual Income"] = df["Annual Income"].sum()
    return df


# ------------------------------------------------------------
# PORTFOLIO OVERVIEW PAGE
# ------------------------------------------------------------

def render_portfolio_overview():
    st.header("📊 Portfolio Overview")

    st.download_button(
        "Download CSV Template",
        data=portfolio_template_csv(),
        file_name="portfolio_template.csv",
        key="portfolio_template_dl",
    )

    uploaded = st.file_uploader(
        "Upload Portfolio CSV",
        type="csv",
        key="portfolio_upload",
    )

    if not uploaded:
        st.info("Upload a portfolio CSV to begin.")
        return

    raw = pd.read_csv(uploaded)
    ok, msg = validate_portfolio(raw)

    if not ok:
        st.error(msg)
        return

    portfolio = compute_portfolio(raw)

    st.session_state.portfolio_raw = raw
    st.session_state.portfolio = portfolio

    st.dataframe(portfolio, use_container_width=True)

    income = compute_dividends(portfolio)
    st.session_state.portfolio_meta["income"] = income.to_dict()

    st.subheader("Dividend Income")
    st.dataframe(income, use_container_width=True)


# ------------------------------------------------------------
# PORTFOLIO INSIGHTS PAGE
# ------------------------------------------------------------

def render_portfolio_insights():
    st.header("🧬 Portfolio Insights")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        st.info("Upload a portfolio first.")
        return

    exposure = compute_lookthrough(portfolio)
    st.subheader("Look-Through Exposure")
    st.dataframe(exposure.head(20), use_container_width=True)
# ============================================================
# PART 10 / 14 — MONTE CARLO + GOAL PROBABILITY ENGINE
# ============================================================

# ------------------------------------------------------------
# RETURN & VOLATILITY ASSUMPTIONS (EDUCATIONAL DEFAULTS)
# ------------------------------------------------------------

def estimate_return_and_volatility() -> Dict[str, float]:
    """
    Conservative long-term assumptions.
    These are educational defaults, not predictions.
    """
    return {
        "expected_return": 0.07,   # 7% annual
        "volatility": 0.15,        # 15% annual
    }


# ------------------------------------------------------------
# MONTE CARLO SIMULATION CORE
# ------------------------------------------------------------

def monte_carlo_simulation(
    start_value: float,
    annual_contribution: float,
    years: int,
    expected_return: float,
    volatility: float,
    simulations: int = 3000,
) -> np.ndarray:
    """
    Runs Monte Carlo simulations.
    Returns array of shape (simulations, years).
    """
    results = np.zeros((simulations, years))

    for i in range(simulations):
        value = start_value
        for y in range(years):
            random_return = np.random.normal(
                expected_return,
                volatility,
            )
            value = value * (1 + random_return) + annual_contribution
            results[i, y] = value

    return results


# ------------------------------------------------------------
# GOAL SUCCESS METRICS
# ------------------------------------------------------------

def calculate_goal_probability(
    simulations: np.ndarray,
    goal: float,
) -> float:
    final_values = simulations[:, -1]
    return round((final_values >= goal).mean() * 100, 2)


def summarize_simulation_outcomes(
    simulations: np.ndarray,
) -> Dict[str, float]:
    final_values = simulations[:, -1]

    return {
        "Pessimistic (10th %)": round(np.percentile(final_values, 10), 0),
        "Median (50th %)": round(np.percentile(final_values, 50), 0),
        "Optimistic (90th %)": round(np.percentile(final_values, 90), 0),
    }


# ------------------------------------------------------------
# MONTE CARLO VISUALIZATION HELPERS
# ------------------------------------------------------------

def monte_carlo_sample_paths(
    simulations: np.ndarray,
    max_paths: int = 50,
) -> pd.DataFrame:
    sample = simulations[:max_paths]
    return pd.DataFrame(sample.T)


def monte_carlo_distribution(
    simulations: np.ndarray,
) -> pd.Series:
    return pd.Series(simulations[:, -1])


# ------------------------------------------------------------
# GOAL PROBABILITY PAGE
# ------------------------------------------------------------

def render_goal_probability():
    st.header("🎯 Goal Probability")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        st.info("Upload a portfolio first.")
        return

    total_value = float(
        portfolio.loc["TOTAL", "Market Value"]
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        goal = st.number_input(
            "Target Goal ($)",
            min_value=0.0,
            value=1_000_000.0,
            step=50_000.0,
            key="goal_target_input",
        )

    with c2:
        annual_contribution = st.number_input(
            "Annual Contribution ($)",
            min_value=0.0,
            value=10_000.0,
            step=1_000.0,
            key="goal_contribution_input",
        )

    with c3:
        years = st.slider(
            "Years",
            min_value=1,
            max_value=40,
            value=20,
            key="goal_years_slider",
        )

    st.markdown("---")

    assumptions = estimate_return_and_volatility()

    simulations = monte_carlo_simulation(
        start_value=total_value,
        annual_contribution=annual_contribution,
        years=years,
        expected_return=assumptions["expected_return"],
        volatility=assumptions["volatility"],
    )

    probability = calculate_goal_probability(
        simulations,
        goal,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Goal Success Probability", f"{probability}%")
    m2.metric(
        "Expected Return",
        f"{round(assumptions['expected_return'] * 100, 2)}%",
    )
    m3.metric(
        "Volatility",
        f"{round(assumptions['volatility'] * 100, 2)}%",
    )

    st.markdown("---")

    summary = summarize_simulation_outcomes(simulations)

    summary_df = pd.DataFrame(
        {
            "Scenario": list(summary.keys()),
            "Ending Value ($)": list(summary.values()),
        }
    )

    st.subheader("Outcome Summary")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Monte Carlo Sample Paths")
    paths_df = monte_carlo_sample_paths(simulations)
    st.line_chart(paths_df)

    st.subheader("Distribution of Ending Values")
    dist = monte_carlo_distribution(simulations)
    st.bar_chart(dist.value_counts().sort_index())

    # Persist results for AI explanations & chatbot
    st.session_state.portfolio_meta["goal_analysis"] = {
        "goal": goal,
        "years": years,
        "annual_contribution": annual_contribution,
        "probability": probability,
        "assumptions": assumptions,
    }
# ============================================================
# PART 11 / 14 — AI INSIGHTS + CHATBOT
# ============================================================

# ------------------------------------------------------------
# PORTFOLIO HEALTH ANALYSIS
# ------------------------------------------------------------

def calculate_portfolio_health(portfolio: pd.DataFrame) -> Dict[str, Any]:
    holdings = portfolio.drop(index="TOTAL")
    total_value = holdings["Market Value"].sum()
    weights = holdings["Market Value"] / total_value

    max_weight = float(weights.max())
    concentration_index = float((weights ** 2).sum())

    risk_level = "Low"
    if max_weight > 0.4:
        risk_level = "High"
    elif concentration_index > 0.25:
        risk_level = "Medium"

    return {
        "max_weight": round(max_weight * 100, 2),
        "concentration_index": round(concentration_index, 3),
        "risk_level": risk_level,
    }


def render_portfolio_health_ai():
    st.header("🧠 Portfolio Health")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        st.info("Upload a portfolio first.")
        return

    health = calculate_portfolio_health(portfolio)

    c1, c2, c3 = st.columns(3)
    c1.metric("Largest Holding", f"{health['max_weight']}%")
    c2.metric("Concentration Index", health["concentration_index"])
    c3.metric("Risk Level", health["risk_level"])

    st.markdown("---")

    if health["risk_level"] == "High":
        st.warning(
            "Your portfolio is highly concentrated. "
            "This increases volatility and downside risk."
        )
    elif health["risk_level"] == "Medium":
        st.info(
            "Your portfolio shows moderate concentration. "
            "Diversification could reduce risk."
        )
    else:
        st.success(
            "Your portfolio appears well diversified."
        )

    st.session_state.portfolio_meta["health"] = health


# ------------------------------------------------------------
# AI REBALANCING LOGIC
# ------------------------------------------------------------

def generate_rebalancing_suggestions(portfolio: pd.DataFrame) -> List[str]:
    holdings = portfolio.drop(index="TOTAL")
    total_value = holdings["Market Value"].sum()
    weights = holdings["Market Value"] / total_value

    suggestions = []

    for ticker, weight in zip(holdings["Ticker"], weights):
        if weight > 0.35:
            suggestions.append(
                f"Reduce **{ticker}** exposure to lower concentration risk."
            )

    if not suggestions:
        suggestions.append(
            "No rebalancing needed. Portfolio weights appear balanced."
        )

    return suggestions


def render_ai_rebalancing():
    st.header("⚖️ AI Rebalancing Suggestions")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        st.info("Upload a portfolio first.")
        return

    suggestions = generate_rebalancing_suggestions(portfolio)

    for s in suggestions:
        st.markdown(f"- {s}")

    st.caption(
        "Rebalancing suggestions are educational only, not investment advice."
    )


# ------------------------------------------------------------
# INCOME FORECAST AI
# ------------------------------------------------------------

def render_income_forecast_ai():
    st.header("💵 Income Forecast")

    income_meta = st.session_state.get("portfolio_meta", {}).get("income")
    if not income_meta:
        st.info("Upload a dividend-paying portfolio first.")
        return

    df = pd.DataFrame(income_meta).T
    current_income = float(df.loc["TOTAL", "Annual Income"])

    growth = st.slider(
        "Assumed Annual Income Growth (%)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.5,
        key="income_growth_slider",
    )

    years = list(range(1, 11))
    projected = [
        round(current_income * ((1 + growth / 100) ** y), 2)
        for y in years
    ]

    forecast_df = pd.DataFrame(
        {"Year": years, "Projected Income ($)": projected}
    )

    st.metric("Current Annual Income", f"${round(current_income, 2)}")
    st.line_chart(forecast_df.set_index("Year"))


# ------------------------------------------------------------
# RULE-BASED AI CHATBOT
# ------------------------------------------------------------

def render_ai_chatbot():
    st.header("🤖 AI Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input(
        "Ask a question about your portfolio, risk, or goals",
        key="chat_input",
    )

    if st.button("Ask", key="chat_ask_btn"):
        response = generate_chatbot_response(user_input)
        st.session_state.chat_history.append(
            {"q": user_input, "a": response}
        )

    st.markdown("---")

    for msg in reversed(st.session_state.chat_history[-6:]):
        st.markdown(f"**You:** {msg['q']}")
        st.markdown(f"**AI:** {msg['a']}")
        st.markdown("---")


def generate_chatbot_response(question: str) -> str:
    q = question.lower()

    portfolio = st.session_state.get("portfolio")
    goal_meta = st.session_state.get("portfolio_meta", {}).get("goal_analysis")
    health = st.session_state.get("portfolio_meta", {}).get("health")

    if "risk" in q and health:
        return (
            f"Your portfolio risk level is **{health['risk_level']}**. "
            f"The largest holding is {health['max_weight']}% of your portfolio."
        )

    if "goal" in q and goal_meta:
        return (
            f"You have a **{goal_meta['probability']}%** probability of reaching "
            f"your ${int(goal_meta['goal']):,} goal in "
            f"{goal_meta['years']} years."
        )

    if "divers" in q:
        return (
            "Diversification spreads investments across assets "
            "to reduce the impact of any single loss."
        )

    if "rebalance" in q:
        return (
            "Rebalancing means adjusting holdings to maintain desired risk levels."
        )

    if "portfolio" in q and portfolio is not None:
        total_value = float(portfolio.loc["TOTAL", "Market Value"])
        return (
            f"Your portfolio total value is approximately "
            f"${round(total_value, 2):,}."
        )

    return (
        "I can help explain portfolio risk, diversification, goals, "
        "income forecasts, and long-term investing concepts."
    )
# ============================================================
# PART 12 / 14 — RISK, TAX, PERFORMANCE, EXPORTS
# ============================================================

# ------------------------------------------------------------
# RISK ALERTS (CONCENTRATION + DRAWDOWN EDUCATION)
# ------------------------------------------------------------

def render_risk_alerts():
    st.header("🚨 Risk Alerts")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        st.info("Upload a portfolio first.")
        return

    holdings = portfolio.drop(index="TOTAL")
    total_value = holdings["Market Value"].sum()
    weights = holdings["Market Value"] / total_value

    alerts = []

    for ticker, weight in zip(holdings["Ticker"], weights):
        if weight > 0.40:
            alerts.append(
                f"⚠️ **{ticker}** represents {round(weight * 100, 1)}% "
                f"of your portfolio, which increases concentration risk."
            )

    if not alerts:
        st.success("✅ No major concentration risks detected.")
    else:
        for alert in alerts:
            st.warning(alert)

    st.caption(
        "Risk alerts are informational only and not investment advice."
    )


# ------------------------------------------------------------
# TAX OPTIMIZATION (EDUCATIONAL – NO ADVICE)
# ------------------------------------------------------------

def render_tax_optimization():
    st.header("🧾 Tax Optimization (Educational)")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        st.info("Upload a portfolio first.")
        return

    holdings = portfolio.drop(index="TOTAL")
    losses = holdings[holdings["PnL"] < 0]

    if losses.empty:
        st.success("✅ No unrealized losses detected.")
        return

    st.subheader("Potential Tax-Loss Harvesting Candidates")
    st.dataframe(
        losses[["Ticker", "PnL"]],
        use_container_width=True,
    )

    st.caption(
        "This section is for learning purposes only. "
        "Consult a tax professional before making decisions."
    )


# ------------------------------------------------------------
# PERFORMANCE SUMMARY (EDUCATIONAL SNAPSHOT)
# ------------------------------------------------------------

def render_performance_benchmark():
    st.header("📈 Performance Summary")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        st.info("Upload a portfolio first.")
        return

    total_value = float(portfolio.loc["TOTAL", "Market Value"])
    total_pnl = float(portfolio.loc["TOTAL", "PnL"])

    c1, c2 = st.columns(2)
    c1.metric("Total Portfolio Value", f"${round(total_value, 2):,}")
    c2.metric("Total P&L", f"${round(total_pnl, 2):,}")

    st.markdown("---")

    st.caption(
        "Performance metrics shown are point-in-time snapshots "
        "and do not represent historical or future returns."
    )


# ------------------------------------------------------------
# EXPORTS (CSV ONLY — SAFE)
# ------------------------------------------------------------

def render_exports():
    st.header("📤 Exports")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
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
        key="export_portfolio_csv",
    )

    st.caption(
        "Exports are provided for personal education and record-keeping only."
    )
# ============================================================
# PART 13 / 14 — ADVANCED EDUCATIONAL FEATURES + CHATBOT
# ============================================================

# ------------------------------------------------------------
# LEARNING NOTES / JOURNAL
# ------------------------------------------------------------

def render_learning_notes():
    st.header("📝 Learning Notes")

    st.markdown(
        """
        Write your own understanding of investing concepts, portfolio behavior,
        or lessons learned. Notes are saved only for this session.
        """
    )

    notes = st.text_area(
        "Your notes",
        value=st.session_state.get("learning_notes", ""),
        height=220,
        key="learning_notes_textarea",
    )

    if st.button("Save Notes", key="save_learning_notes"):
        st.session_state.learning_notes = notes
        st.success("Notes saved for this session.")

    if notes:
        st.markdown("---")
        st.subheader("Your Reflection")
        st.markdown(notes)


# ------------------------------------------------------------
# SCENARIO COMPARISON (WHAT-IF ANALYSIS)
# ------------------------------------------------------------

def render_scenario_comparison():
    st.header("📊 Scenario Comparison (What-If Analysis)")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        st.info("Upload a portfolio first.")
        return

    st.markdown(
        """
        Compare **hypothetical scenarios** to understand how assumptions
        impact long-term outcomes. These are **not predictions**.
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        return_a = st.slider(
            "Scenario A – Expected Return (%)",
            0.0, 12.0, 7.0,
            key="scenario_a_return",
        )
        vol_a = st.slider(
            "Scenario A – Volatility (%)",
            5.0, 30.0, 15.0,
            key="scenario_a_vol",
        )

    with c2:
        return_b = st.slider(
            "Scenario B – Expected Return (%)",
            0.0, 12.0, 5.0,
            key="scenario_b_return",
        )
        vol_b = st.slider(
            "Scenario B – Volatility (%)",
            5.0, 30.0, 10.0,
            key="scenario_b_vol",
        )

    total_value = float(portfolio.loc["TOTAL", "Market Value"])
    years = 20

    def simulate_once(r, v):
        value = total_value
        for _ in range(years):
            value *= (1 + np.random.normal(r / 100, v / 100))
        return round(value, 0)

    results = pd.DataFrame(
        {
            "Scenario": ["A", "B"],
            "Ending Value (Sample Run)": [
                simulate_once(return_a, vol_a),
                simulate_once(return_b, vol_b),
            ],
        }
    )

    st.subheader("Illustrative Outcome (Single Run)")
    st.dataframe(results, use_container_width=True)

    st.caption(
        "Each run is random. Results vary. Educational use only."
    )


# ------------------------------------------------------------
# PORTFOLIO SNAPSHOTS (SESSION-LEVEL)
# ------------------------------------------------------------

def render_portfolio_snapshots():
    st.header("📸 Portfolio Snapshots")

    if "snapshots" not in st.session_state:
        st.session_state.snapshots = []

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        st.info("Upload a portfolio first.")
        return

    if st.button("Save Snapshot", key="save_snapshot_btn"):
        st.session_state.snapshots.append(portfolio.copy())
        st.success(f"Snapshot #{len(st.session_state.snapshots)} saved.")

    if not st.session_state.snapshots:
        st.info("No snapshots saved yet.")
        return

    st.subheader("Saved Snapshots")

    for i, snap in enumerate(st.session_state.snapshots):
        with st.expander(f"Snapshot {i + 1}", expanded=False):
            st.dataframe(snap, use_container_width=True)


# ------------------------------------------------------------
# USER PREFERENCES / SETTINGS
# ------------------------------------------------------------

def render_settings():
    st.header("⚙️ Preferences & Settings")

    show_tips = st.checkbox(
        "Show educational tips",
        value=st.session_state.get("show_tips", True),
        key="pref_show_tips",
    )

    theme = st.selectbox(
        "Preferred theme (visual only)",
        ["Default", "Dark", "Light"],
        index=0,
        key="pref_theme",
    )

    st.session_state.show_tips = show_tips
    st.session_state.theme = theme

    st.success("Preferences saved for this session.")


# ------------------------------------------------------------
# LEARNING CHECKLIST (STUDENT-FRIENDLY)
# ------------------------------------------------------------

def render_learning_checklist():
    st.header("✅ Learning Checklist")

    default_checklist = {
        "Understand what a portfolio is": False,
        "Know why diversification matters": False,
        "Understand risk vs reward": False,
        "Learn what volatility means": False,
        "Explore goal probability": False,
    }

    checklist = st.session_state.get(
        "learning_checklist",
        default_checklist.copy(),
    )

    for item in checklist:
        checklist[item] = st.checkbox(
            item,
            value=checklist[item],
            key=f"check_{item}",
        )

    st.session_state.learning_checklist = checklist

    completed = sum(checklist.values())
    total = len(checklist)

    st.metric("Progress", f"{completed} / {total} topics completed")


# ------------------------------------------------------------
# AI CHATBOT (RULE-BASED, SAFE, NO API)
# ------------------------------------------------------------

def render_ai_chatbot():
    st.header("🤖 AI Chatbot (Educational)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input(
        "Ask a question about investing or portfolios",
        key="chatbot_input",
    )

    if st.button("Ask", key="chatbot_ask"):
        response = generate_chat_response(user_input)
        st.session_state.chat_history.append(
            {"q": user_input, "a": response}
        )

    st.markdown("---")

    for msg in reversed(st.session_state.chat_history[-6:]):
        st.markdown(f"**You:** {msg['q']}")
        st.markdown(f"**AI:** {msg['a']}")
        st.markdown("---")


def generate_chat_response(question: str) -> str:
    if not question:
        return "Please ask a question."

    q = question.lower()

    if "risk" in q:
        return "Risk refers to how much investment values can fluctuate over time."
    if "divers" in q:
        return "Diversification means spreading investments to reduce risk."
    if "goal" in q:
        return "Goal probability estimates how likely you are to reach a target amount."
    if "rebalance" in q:
        return "Rebalancing helps keep portfolio risk aligned over time."
    if "etf" in q:
        return "ETFs hold many stocks, providing built-in diversification."
    if "stock" in q:
        return "Stocks represent ownership in individual companies."

    return (
        "I can explain portfolios, diversification, risk, goals, "
        "ETFs, and long-term investing concepts."
    )
# ============================================================
# PART 13 / 14 — ADVANCED EDUCATIONAL FEATURES + CHATBOT
# ============================================================

# ------------------------------------------------------------
# LEARNING NOTES / JOURNAL
# ------------------------------------------------------------

def render_learning_notes():
    st.header("📝 Learning Notes")

    st.markdown(
        """
        Write your own understanding of investing concepts, portfolio behavior,
        or lessons learned. Notes are saved only for this session.
        """
    )

    notes = st.text_area(
        "Your notes",
        value=st.session_state.get("learning_notes", ""),
        height=220,
        key="learning_notes_textarea",
    )

    if st.button("Save Notes", key="save_learning_notes"):
        st.session_state.learning_notes = notes
        st.success("Notes saved for this session.")

    if notes:
        st.markdown("---")
        st.subheader("Your Reflection")
        st.markdown(notes)


# ------------------------------------------------------------
# SCENARIO COMPARISON (WHAT-IF ANALYSIS)
# ------------------------------------------------------------

def render_scenario_comparison():
    st.header("📊 Scenario Comparison (What-If Analysis)")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        st.info("Upload a portfolio first.")
        return

    st.markdown(
        """
        Compare **hypothetical scenarios** to understand how assumptions
        impact long-term outcomes. These are **not predictions**.
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        return_a = st.slider(
            "Scenario A – Expected Return (%)",
            0.0, 12.0, 7.0,
            key="scenario_a_return",
        )
        vol_a = st.slider(
            "Scenario A – Volatility (%)",
            5.0, 30.0, 15.0,
            key="scenario_a_vol",
        )

    with c2:
        return_b = st.slider(
            "Scenario B – Expected Return (%)",
            0.0, 12.0, 5.0,
            key="scenario_b_return",
        )
        vol_b = st.slider(
            "Scenario B – Volatility (%)",
            5.0, 30.0, 10.0,
            key="scenario_b_vol",
        )

    total_value = float(portfolio.loc["TOTAL", "Market Value"])
    years = 20

    def simulate_once(r, v):
        value = total_value
        for _ in range(years):
            value *= (1 + np.random.normal(r / 100, v / 100))
        return round(value, 0)

    results = pd.DataFrame(
        {
            "Scenario": ["A", "B"],
            "Ending Value (Sample Run)": [
                simulate_once(return_a, vol_a),
                simulate_once(return_b, vol_b),
            ],
        }
    )

    st.subheader("Illustrative Outcome (Single Run)")
    st.dataframe(results, use_container_width=True)

    st.caption(
        "Each run is random. Results vary. Educational use only."
    )


# ------------------------------------------------------------
# PORTFOLIO SNAPSHOTS (SESSION-LEVEL)
# ------------------------------------------------------------

def render_portfolio_snapshots():
    st.header("📸 Portfolio Snapshots")

    if "snapshots" not in st.session_state:
        st.session_state.snapshots = []

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        st.info("Upload a portfolio first.")
        return

    if st.button("Save Snapshot", key="save_snapshot_btn"):
        st.session_state.snapshots.append(portfolio.copy())
        st.success(f"Snapshot #{len(st.session_state.snapshots)} saved.")

    if not st.session_state.snapshots:
        st.info("No snapshots saved yet.")
        return

    st.subheader("Saved Snapshots")

    for i, snap in enumerate(st.session_state.snapshots):
        with st.expander(f"Snapshot {i + 1}", expanded=False):
            st.dataframe(snap, use_container_width=True)


# ------------------------------------------------------------
# USER PREFERENCES / SETTINGS
# ------------------------------------------------------------

def render_settings():
    st.header("⚙️ Preferences & Settings")

    show_tips = st.checkbox(
        "Show educational tips",
        value=st.session_state.get("show_tips", True),
        key="pref_show_tips",
    )

    theme = st.selectbox(
        "Preferred theme (visual only)",
        ["Default", "Dark", "Light"],
        index=0,
        key="pref_theme",
    )

    st.session_state.show_tips = show_tips
    st.session_state.theme = theme

    st.success("Preferences saved for this session.")


# ------------------------------------------------------------
# LEARNING CHECKLIST (STUDENT-FRIENDLY)
# ------------------------------------------------------------

def render_learning_checklist():
    st.header("✅ Learning Checklist")

    default_checklist = {
        "Understand what a portfolio is": False,
        "Know why diversification matters": False,
        "Understand risk vs reward": False,
        "Learn what volatility means": False,
        "Explore goal probability": False,
    }

    checklist = st.session_state.get(
        "learning_checklist",
        default_checklist.copy(),
    )

    for item in checklist:
        checklist[item] = st.checkbox(
            item,
            value=checklist[item],
            key=f"check_{item}",
        )

    st.session_state.learning_checklist = checklist

    completed = sum(checklist.values())
    total = len(checklist)

    st.metric("Progress", f"{completed} / {total} topics completed")


# ------------------------------------------------------------
# AI CHATBOT (RULE-BASED, SAFE, NO API)
# ------------------------------------------------------------

def render_ai_chatbot():
    st.header("🤖 AI Chatbot (Educational)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input(
        "Ask a question about investing or portfolios",
        key="chatbot_input",
    )

    if st.button("Ask", key="chatbot_ask"):
        response = generate_chat_response(user_input)
        st.session_state.chat_history.append(
            {"q": user_input, "a": response}
        )

    st.markdown("---")

    for msg in reversed(st.session_state.chat_history[-6:]):
        st.markdown(f"**You:** {msg['q']}")
        st.markdown(f"**AI:** {msg['a']}")
        st.markdown("---")


def generate_chat_response(question: str) -> str:
    if not question:
        return "Please ask a question."

    q = question.lower()

    if "risk" in q:
        return "Risk refers to how much investment values can fluctuate over time."
    if "divers" in q:
        return "Diversification means spreading investments to reduce risk."
    if "goal" in q:
        return "Goal probability estimates how likely you are to reach a target amount."
    if "rebalance" in q:
        return "Rebalancing helps keep portfolio risk aligned over time."
    if "etf" in q:
        return "ETFs hold many stocks, providing built-in diversification."
    if "stock" in q:
        return "Stocks represent ownership in individual companies."

    return (
        "I can explain portfolios, diversification, risk, goals, "
        "ETFs, and long-term investing concepts."
    )
