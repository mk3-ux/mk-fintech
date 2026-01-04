# ============================================================
# KATTA WEALTH INSIGHTS — SINGLE FILE STREAMLIT APP
# PART 1 / 12 — CONFIG, IMPORTS, SESSION, DATABASE, SECURITY
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
from typing import Dict, Any, List

# ============================================================
# STREAMLIT CONFIG (MUST BE FIRST STREAMLIT CALL)
# ============================================================

st.set_page_config(
    page_title="Katta Wealth Insights",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SESSION INITIALIZATION (AUTHORITATIVE)
# ============================================================

def init_session():
    defaults = {
        "app_mode": "about",
        "current_user": None,
        "is_paid": False,
        "portfolio_raw": None,
        "portfolio": None,
        "portfolio_meta": {},
        "chat_history": [],
        "alerts": [],
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def legal_section(title: str, body: str):
    st.markdown(
        f"""
        <div style="margin-top:1.5rem;">
            <h3>{title}</h3>
            <p style="font-size:0.95rem; color:#d1d5db; line-height:1.6;">
                {body}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# ABOUT US + LEGAL DISCLOSURES (FULL TEXT)
# ------------------------------------------------------------

def render_about_us_legal():
    st.markdown("## 💎 About Katta Wealth Insights")

    st.markdown(
        """
        **Katta Wealth Insights** is an education-first financial analytics platform
        designed to help users understand portfolios, diversification, risk,
        and long-term investing concepts through visual tools and simulations.
        """
    )

    legal_section(
        "Educational Purpose Only",
        """
        Katta Wealth Insights is provided strictly for **educational and informational purposes**.
        The platform is intended to help users learn about general investing concepts,
        portfolio construction, diversification, and long-term planning.
        """
    )

    legal_section(
        "Not Financial, Investment, Tax, or Legal Advice",
        """
        Nothing on this platform constitutes **financial advice, investment advice,
        tax advice, or legal advice**.
        Katta Wealth Insights is **not** a registered investment adviser, broker-dealer,
        or financial planner.
        """
    )

    legal_section(
        "No Fiduciary Relationship",
        """
        Use of this platform does **not** create a fiduciary, advisory,
        or client relationship of any kind.
        All decisions based on information from this platform
        are made solely at the user’s discretion and risk.
        """
    )

    legal_section(
        "Risk Disclosure",
        """
        Investing involves risk, including the possible loss of principal.
        Past performance, simulations, projections, or hypothetical results
        do **not** guarantee future outcomes.
        """
    )

    legal_section(
        "Simulations & Hypothetical Results",
        """
        Monte Carlo simulations and forecasts shown on this platform
        are **hypothetical in nature**.
        They rely on assumptions that may not reflect real-world conditions
        and should not be relied upon as predictions or guarantees.
        """
    )

    legal_section(
        "Data Accuracy & Sources",
        """
        Any financial data used may be derived from public,
        third-party, or user-provided sources.
        Katta Wealth Insights does **not guarantee accuracy,
        completeness, or timeliness** of any data.
        """
    )

    legal_section(
        "User Responsibility",
        """
        Users are encouraged to consult qualified professionals
        before making financial decisions.
        By using this platform, you acknowledge that you assume
        full responsibility for your actions.
        """
    )

    st.markdown("---")
    st.caption(
        "© Katta Wealth Insights — Educational platform only. "
        "No advice. No guarantees."
    )


# ------------------------------------------------------------
# GLOBAL LEGAL BANNER (ALWAYS SHOWN)
# ------------------------------------------------------------

def render_legal_banner():
    st.markdown(
        """
        <div style="
            background:#111827;
            border-top:1px solid #374151;
            padding:0.6rem 1rem;
            font-size:0.8rem;
            color:#9ca3af;
            text-align:center;
            margin-top:2rem;
        ">
            Educational use only · No financial, tax, or legal advice ·
            No guarantees · Investing involves risk
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# GLOBAL LEGAL EXPANDER (ALWAYS AVAILABLE)
# ------------------------------------------------------------

def render_legal_expander():
    with st.expander(
        "ℹ️ About, Legal & Disclosures (Always Available)",
        expanded=False,
    ):
        render_about_us_legal()
# ============================================================
# PART 3 / 12 — TOP NAVIGATION + MARKETING PAGES
# ============================================================

# ------------------------------------------------------------
# TOP NAVIGATION (GLOBAL)
# ------------------------------------------------------------

def render_top_nav():
    st.markdown(
        """
        <style>
        .kwi-topnav {
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding:0.75rem 2rem;
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

    c1, c2 = st.columns([3, 7])

    with c1:
        st.markdown(
            "<div class='kwi-brand'>💎 Katta Wealth Insights</div>",
            unsafe_allow_html=True,
        )

    with c2:
        nav = st.columns(6)
        buttons = [
            ("About", "about"),
            ("Features", "features"),
            ("How It Works", "how"),
            ("Benefits", "benefits"),
            ("About & Legal", "legal"),
            ("Demo", "demo"),
        ]

        for i, (label, mode) in enumerate(buttons):
            if nav[i].button(label, use_container_width=True):
                st.session_state.app_mode = mode
                st.rerun()


# ------------------------------------------------------------
# MARKETING PAGE HELPERS
# ------------------------------------------------------------

def marketing_header(title: str, subtitle: str, icon: str = ""):
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


def marketing_section(title: str, body: str):
    st.markdown(
        f"""
        <div style="margin-top:2rem;">
            <h3>{title}</h3>
            <p style="color:#e5e7eb;">
                {body}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# ABOUT PAGE
# ------------------------------------------------------------

def render_about():
    marketing_header(
        "About Katta Wealth Insights",
        "An education-first platform for understanding investing.",
        "💡",
    )

    marketing_section(
        "Our Mission",
        """
        To help families, students, and long-term investors
        understand portfolios, diversification, and risk
        without hype, fear, or complexity.
        """,
    )

    marketing_section(
        "What This Is",
        """
        An educational analytics platform.
        No trading. No execution. No recommendations.
        """,
    )

    marketing_section(
        "Who It’s For",
        """
        • Families learning together<br>
        • Students & teens<br>
        • Long-term investors
        """,
    )


# ------------------------------------------------------------
# FEATURES PAGE
# ------------------------------------------------------------

def render_features():
    marketing_header(
        "Features",
        "Professional-grade analytics explained simply.",
        "✨",
    )

    marketing_section(
        "Portfolio Analytics",
        "Upload once and instantly see value, P&L, and diversification.",
    )

    marketing_section(
        "ETF Look-Through",
        "Understand what you really own inside ETFs.",
    )

    marketing_section(
        "Goal Probability",
        "Monte Carlo simulations estimate long-term success odds.",
    )

    marketing_section(
        "AI-Style Insights",
        "Plain-English explanations for risk and rebalancing.",
    )


# ------------------------------------------------------------
# HOW IT WORKS PAGE
# ------------------------------------------------------------

def render_how():
    marketing_header(
        "How It Works",
        "A simple flow from upload to insight.",
        "🛠️",
    )

    marketing_section(
        "1. Upload",
        "Upload a CSV of your investments using a simple template.",
    )

    marketing_section(
        "2. Analyze",
        "We compute value, diversification, income, and risk.",
    )

    marketing_section(
        "3. Explore",
        "Interact with simulations and visual explanations.",
    )

    marketing_section(
        "4. Learn",
        "Designed for education and long-term understanding.",
    )


# ------------------------------------------------------------
# BENEFITS PAGE
# ------------------------------------------------------------

def render_benefits():
    marketing_header(
        "Benefits",
        "Why learners choose Katta Wealth Insights.",
        "🎯",
    )

    marketing_section(
        "Clarity Over Noise",
        "Understand investing without media-driven fear.",
    )

    marketing_section(
        "Education First",
        "Built for students, families, and long-term thinkers.",
    )

    marketing_section(
        "Safe & Transparent",
        "No execution, no leverage, no financial advice.",
    )
# ============================================================
# PART 4 / 12 — AUTHENTICATION + PAYWALL
# ============================================================

# ------------------------------------------------------------
# DATABASE HELPERS (AUTH)
# ------------------------------------------------------------

DB_PATH = "kwi_app.db"

def db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def db_init():
    conn = db_connect()
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


db_init()


# ------------------------------------------------------------
# SECURITY HELPERS
# ------------------------------------------------------------

def now_iso():
    return dt.datetime.utcnow().isoformat()


def hash_pw(pw: str) -> str:
    return hashlib.sha256(("kwi_salt_" + pw).encode()).hexdigest()


# ------------------------------------------------------------
# USER HELPERS
# ------------------------------------------------------------

def db_get_user(email: str):
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


def mark_user_paid(email: str):
    conn = db_connect()
    conn.execute(
        "UPDATE users SET is_paid=1 WHERE email=?",
        (email,),
    )
    conn.commit()
    conn.close()


# ------------------------------------------------------------
# AUTH UI
# ------------------------------------------------------------

def render_auth():
    st.header("🔐 Sign in to Katta Wealth Insights")

    login_tab, signup_tab = st.tabs(["Log In", "Create Account"])

    # ----------------------------
    # LOGIN
    # ----------------------------
    with login_tab:
        email = st.text_input("Email", key="login_email")
        pw = st.text_input("Password", type="password", key="login_pw")

        if st.button("Log In", use_container_width=True):
            if not email or not pw:
                st.error("Please enter email and password.")
                return

            user = db_get_user(email)
            if not user or user["pw"] != hash_pw(pw):
                st.error("Invalid credentials.")
                return

            st.session_state.current_user = email
            st.session_state.is_paid = user["is_paid"]
            st.success("Logged in successfully.")
            st.rerun()

    # ----------------------------
    # SIGNUP
    # ----------------------------
    with signup_tab:
        email = st.text_input("Email", key="signup_email")
        pw = st.text_input("Password", type="password", key="signup_pw")

        if st.button("Create Account", use_container_width=True):
            if not email or not pw:
                st.error("Please enter email and password.")
                return

            created = db_create_user(email, hash_pw(pw))
            if not created:
                st.error("Account already exists.")
                return

            st.success("Account created. Please log in.")


# ------------------------------------------------------------
# PAYWALL UI
# ------------------------------------------------------------

def render_payment():
    st.header("💎 Upgrade to Katta Wealth Pro")

    st.markdown(
        """
        ### What you unlock:
        - 📊 Full portfolio analytics
        - 🧬 ETF look-through exposure
        - 💵 Dividend income engine
        - 🎯 Monte Carlo goal probability
        - 🧠 AI-style insights
        - 🚨 Risk & tax analysis
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
        if st.button("Upgrade (Demo)", use_container_width=True):
            mark_user_paid(st.session_state.current_user)
            st.session_state.is_paid = True
            st.success("Welcome to Pro 🎉")
            st.rerun()


# ------------------------------------------------------------
# AUTH + PAYMENT GATE
# ------------------------------------------------------------

def enforce_auth_and_payment():
    # SAFE access — never assume key exists
    current_user = st.session_state.get("current_user")
    is_paid = st.session_state.get("is_paid", False)

    if not current_user:
        render_auth()
        return False

    if not is_paid:
        render_payment()
        return False

    return True

# ============================================================
# PART 5 / 12 — DEMO SIDEBAR + DEMO ROUTER
# ============================================================

# ------------------------------------------------------------
# DEMO SIDEBAR (LEFT NAV)
# ------------------------------------------------------------

def render_sidebar():
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
        ]

        page = st.radio(
            "",
            pages,
            key="demo_sidebar_page",
        )

        st.markdown("---")

        if st.button("🚪 Log out", use_container_width=True):
            st.session_state.current_user = None
            st.session_state.is_paid = False
            st.session_state.app_mode = "about"
            st.rerun()

    return page


# ------------------------------------------------------------
# DEMO ROUTER (PAGE DISPATCHER)
# ------------------------------------------------------------

def demo_router(page: str):
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
        st.info("Select a demo page from the sidebar.")
# ============================================================
# PART 6 / 12 — PORTFOLIO ENGINE
# ============================================================

# ------------------------------------------------------------
# PORTFOLIO INPUT MODEL
# ------------------------------------------------------------

REQUIRED_COLUMNS = ["Ticker", "Shares", "Cost_Basis"]


def portfolio_template_csv():
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


# ------------------------------------------------------------
# PORTFOLIO CALCULATION (OFFLINE / DEMO SAFE)
# ------------------------------------------------------------

def compute_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in df.iterrows():
        ticker = str(r["Ticker"]).upper().strip()
        shares = float(r["Shares"])
        cost = float(r["Cost_Basis"])

        # Demo-safe pricing model (no live calls)
        live_price = cost
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
# ETF LOOK-THROUGH (STATIC EDUCATIONAL MODEL)
# ------------------------------------------------------------

ETF_LOOKTHROUGH = {
    "VOO": [("AAPL", 0.07), ("MSFT", 0.06), ("NVDA", 0.05)],
    "SPY": [("AAPL", 0.07), ("MSFT", 0.06), ("AMZN", 0.04)],
    "QQQ": [("AAPL", 0.11), ("MSFT", 0.09), ("NVDA", 0.08)],
}


def compute_lookthrough(portfolio: pd.DataFrame) -> pd.DataFrame:
    stocks = []
    total_value = portfolio.loc["TOTAL", "Market Value"]

    for _, r in portfolio.drop(index="TOTAL").iterrows():
        ticker = r["Ticker"]
        weight = r["Market Value"] / total_value

        if ticker in ETF_LOOKTHROUGH:
            for sym, w in ETF_LOOKTHROUGH[ticker]:
                stocks.append(
                    {"Ticker": sym, "Weight": weight * w}
                )
        else:
            stocks.append(
                {"Ticker": ticker, "Weight": weight}
            )

    df = pd.DataFrame(stocks)
    df = df.groupby("Ticker").sum().reset_index()
    df["Weight %"] = (df["Weight"] * 100).round(2)

    return df.sort_values("Weight %", ascending=False)


# ------------------------------------------------------------
# DIVIDEND ENGINE (SIMPLIFIED)
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
        y = DIVIDEND_YIELDS.get(r["Ticker"], 0)
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
        portfolio_template_csv(),
        file_name="portfolio_template.csv",
    )

    uploaded = st.file_uploader(
        "Upload Portfolio CSV",
        type="csv",
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

    st.subheader("Look-Through Stock Exposure")
    st.dataframe(exposure.head(15), use_container_width=True)
# ============================================================
# PART 7 / 12 — GOAL PROBABILITY + MONTE CARLO ENGINE
# ============================================================

# ------------------------------------------------------------
# RETURN & VOLATILITY ESTIMATION (EDUCATIONAL DEFAULTS)
# ------------------------------------------------------------

def estimate_return_and_volatility():
    """
    Conservative long-term assumptions.
    These are educational defaults, NOT predictions.
    """
    expected_return = 0.07   # 7% annual
    volatility = 0.15        # 15% annual
    return expected_return, volatility


# ------------------------------------------------------------
# MONTE CARLO SIMULATION ENGINE
# ------------------------------------------------------------

def monte_carlo_simulation(
    start_value: float,
    annual_contribution: float,
    years: int,
    expected_return: float,
    volatility: float,
    simulations: int = 3000,
):
    """
    Monte Carlo simulation of portfolio growth.
    Returns array shape (simulations, years).
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
    """
    Percentage of simulations ending above the goal.
    """
    final_values = simulations[:, -1]
    return round((final_values >= goal).mean() * 100, 2)


def simulation_summary(simulations: np.ndarray) -> Dict[str, float]:
    """
    Pessimistic / Median / Optimistic outcomes.
    """
    final_values = simulations[:, -1]
    return {
        "Pessimistic (10%)": round(np.percentile(final_values, 10), 0),
        "Median (50%)": round(np.percentile(final_values, 50), 0),
        "Optimistic (90%)": round(np.percentile(final_values, 90), 0),
    }


# ------------------------------------------------------------
# GOAL PROBABILITY PAGE
# ------------------------------------------------------------

def render_goal_probability():
    st.header("🎯 Goal Probability")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        st.info("Upload a portfolio first.")
        return

    total_value = float(portfolio.loc["TOTAL", "Market Value"])

    col1, col2, col3 = st.columns(3)

    with col1:
        goal = st.number_input(
            "Target Goal ($)",
            min_value=0.0,
            value=1_000_000.0,
            step=50_000.0,
        )

    with col2:
        annual_contribution = st.number_input(
            "Annual Contribution ($)",
            min_value=0.0,
            value=10_000.0,
            step=1_000.0,
        )

    with col3:
        years = st.slider(
            "Years",
            min_value=1,
            max_value=40,
            value=20,
        )

    st.markdown("---")

    expected_return, volatility = estimate_return_and_volatility()

    simulations = monte_carlo_simulation(
        start_value=total_value,
        annual_contribution=annual_contribution,
        years=years,
        expected_return=expected_return,
        volatility=volatility,
    )

    probability = goal_success_probability(simulations, goal)

    c1, c2, c3 = st.columns(3)
    c1.metric("Goal Success Probability", f"{probability}%")
    c2.metric("Expected Return", f"{round(expected_return * 100, 2)}%")
    c3.metric("Volatility", f"{round(volatility * 100, 2)}%")

    st.markdown("---")

    summary = simulation_summary(simulations)

    summary_df = pd.DataFrame(
        {
            "Scenario": list(summary.keys()),
            "Ending Value ($)": list(summary.values()),
        }
    )

    st.subheader("Outcome Distribution")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Monte Carlo Paths (Sample)")
    sample_paths = pd.DataFrame(simulations[:50].T)
    st.line_chart(sample_paths)

    # Persist results for AI / educational context
    st.session_state.portfolio_meta["goal_analysis"] = {
        "goal": goal,
        "years": years,
        "probability": probability,
        "expected_return": expected_return,
        "volatility": volatility,
    }
# ============================================================
# PART 8 / 12 — AI-STYLE INSIGHTS & EDUCATIONAL FEATURES
# ============================================================

# ------------------------------------------------------------
# PORTFOLIO HEALTH (RULE-BASED AI)
# ------------------------------------------------------------

def render_portfolio_health_ai():
    st.header("🧠 Portfolio Health")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        st.info("Upload a portfolio first.")
        return

    holdings = portfolio.drop(index="TOTAL")
    total_value = holdings["Market Value"].sum()
    weights = holdings["Market Value"] / total_value

    max_weight = weights.max()
    concentration_index = float((weights ** 2).sum())

    c1, c2 = st.columns(2)
    c1.metric("Largest Holding", f"{round(max_weight * 100, 2)}%")
    c2.metric("Concentration Index", round(concentration_index, 3))

    st.markdown("---")
    st.subheader("AI Assessment")

    if max_weight > 0.4:
        st.warning(
            "⚠️ Your portfolio is highly concentrated in one holding. "
            "This can increase volatility and downside risk."
        )
    elif concentration_index > 0.25:
        st.info(
            "ℹ️ Your portfolio shows moderate concentration. "
            "Additional diversification may reduce risk."
        )
    else:
        st.success(
            "✅ Your portfolio appears well diversified based on current holdings."
        )


# ------------------------------------------------------------
# AI REBALANCING SUGGESTIONS
# ------------------------------------------------------------

def render_ai_rebalancing():
    st.header("⚖️ AI Rebalancing Suggestions")

    portfolio = st.session_state.get("portfolio")
    if portfolio is None:
        st.info("Upload a portfolio first.")
        return

    holdings = portfolio.drop(index="TOTAL")
    total_value = holdings["Market Value"].sum()
    weights = holdings["Market Value"] / total_value

    suggestions = []

    for ticker, weight in zip(holdings["Ticker"], weights):
        if weight > 0.35:
            suggestions.append(
                f"Consider trimming **{ticker}** to reduce concentration risk."
            )

    if not suggestions:
        suggestions.append(
            "Your portfolio weights appear balanced. No immediate changes suggested."
        )

    st.subheader("Suggested Actions")
    for s in suggestions:
        st.markdown(f"- {s}")


# ------------------------------------------------------------
# INCOME FORECAST (DIVIDEND GROWTH MODEL)
# ------------------------------------------------------------

def render_income_forecast_ai():
    st.header("💵 Income Forecast")

    income_meta = st.session_state.get("portfolio_meta", {}).get("income")
    if not income_meta:
        st.info("Upload a dividend-paying portfolio first.")
        return

    df = pd.DataFrame(income_meta).T
    annual_income = float(df.loc["TOTAL", "Annual Income"])

    growth_rate = st.slider(
        "Assumed Annual Income Growth (%)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.5,
    )

    years = list(range(1, 11))
    projected_income = [
        round(annual_income * ((1 + growth_rate / 100) ** y), 2)
        for y in years
    ]

    forecast_df = pd.DataFrame(
        {
            "Year": years,
            "Projected Income ($)": projected_income,
        }
    )

    st.metric("Current Annual Income", f"${round(annual_income, 2)}")
    st.line_chart(forecast_df.set_index("Year"))


# ------------------------------------------------------------
# TEEN EXPLAINER (STUDENT FRIENDLY)
# ------------------------------------------------------------

def render_teen_explainer_ai():
    st.header("🎒 Teen Explainer")

    st.markdown(
        """
        ### What is a portfolio?
        A **portfolio** is a collection of investments like stocks and ETFs.

        ### Why diversify?
        Diversification spreads risk so one bad investment
        doesn’t ruin everything.

        ### What is risk?
        Risk is how much your investment value can go up **or down**.

        ### Long-term investing
        Time and consistency usually matter more than timing the market.
        """
    )


# ------------------------------------------------------------
# AI CHATBOT (RULE-BASED, SAFE)
# ------------------------------------------------------------

def render_ai_chatbot():
    st.header("🤖 AI Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question about investing or portfolios")

    if st.button("Ask"):
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

    return (
        "I can explain portfolios, diversification, risk, goals, and long-term investing."
    )
# ============================================================
# PART 9 / 12 — RISK, TAX, PERFORMANCE & EXPORTS
# ============================================================

# ------------------------------------------------------------
# RISK ALERTS (EDUCATIONAL)
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
                f"⚠️ **{ticker}** represents more than 40% of your portfolio."
            )

    if not alerts:
        st.success("✅ No major concentration risks detected.")
    else:
        for alert in alerts:
            st.warning(alert)

    st.caption(
        "Risk alerts are informational only and do not constitute investment advice."
    )


# ------------------------------------------------------------
# TAX OPTIMIZATION (EDUCATIONAL ONLY)
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

    st.subheader("Potential Tax-Loss Harvesting (Illustrative)")
    st.dataframe(
        losses[["Ticker", "PnL"]],
        use_container_width=True,
    )

    st.caption(
        "This section is educational only. "
        "Consult a qualified tax professional before making decisions."
    )


# ------------------------------------------------------------
# PERFORMANCE SUMMARY
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
    c1.metric("Total Portfolio Value", f"${round(total_value, 2)}")
    c2.metric("Total P&L", f"${round(total_pnl, 2)}")

    st.markdown("---")

    st.caption(
        "Benchmark comparisons (e.g., S&P 500) are not included. "
        "Performance figures shown are educational snapshots only."
    )


# ------------------------------------------------------------
# EXPORTS (REPORTING, NOT ADVICE)
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
    )

    st.caption(
        "Exports are provided for personal learning and record-keeping only."
    )
# ============================================================
# PART 10 / 12 — ALWAYS-VISIBLE LEGAL & ABOUT (MANDATORY)
# ============================================================

# ------------------------------------------------------------
# LEGAL SECTION HELPER
# ------------------------------------------------------------

def legal_section(title: str, body: str):
    st.markdown(
        f"""
        <div style="margin-top:1.6rem;">
            <h3>{title}</h3>
            <p style="font-size:0.95rem; color:#d1d5db; line-height:1.6;">
                {body}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# ABOUT + LEGAL PAGE
# ------------------------------------------------------------

def render_about_us_legal():
    st.markdown("## 💎 About Katta Wealth Insights")

    st.markdown(
        """
        **Katta Wealth Insights** is an education-first financial analytics platform
        designed to help users understand portfolios, diversification, risk,
        and long-term investing concepts using visual tools and simulations.
        """
    )

    legal_section(
        "Educational Purpose Only",
        """
        Katta Wealth Insights is provided strictly for **educational and informational purposes**.
        The platform is intended to help users learn general investing concepts,
        portfolio construction, diversification, and long-term planning.
        """
    )

    legal_section(
        "Not Investment Advice",
        """
        Nothing on this platform constitutes **investment advice, financial advice,
        legal advice, or tax advice**.
        Katta Wealth Insights is **not** a registered investment adviser,
        broker-dealer, or financial planner.
        """
    )

    legal_section(
        "No Fiduciary Relationship",
        """
        Use of this platform does **not** create a fiduciary, advisory,
        or client relationship of any kind.
        All decisions are made solely at the user’s discretion and risk.
        """
    )

    legal_section(
        "Risk Disclosure",
        """
        Investing involves risk, including possible loss of principal.
        Past performance, simulations, or hypothetical results
        do **not** guarantee future outcomes.
        """
    )

    legal_section(
        "Simulations & Hypothetical Results",
        """
        Monte Carlo simulations and forecasts shown are **hypothetical**.
        They rely on assumptions that may not reflect real-world conditions
        and should not be relied upon as predictions or guarantees.
        """
    )

    legal_section(
        "Data Accuracy",
        """
        Financial data may be derived from public, third-party,
        or user-provided sources.
        Accuracy, completeness, and timeliness are not guaranteed.
        """
    )

    legal_section(
        "No Solicitation or Endorsement",
        """
        The platform does not solicit investments,
        endorse securities, or promote financial products.
        Examples and tickers are illustrative only.
        """
    )

    legal_section(
        "User Responsibility",
        """
        Users are encouraged to consult qualified financial,
        tax, or legal professionals before making decisions.
        By using this platform, you assume full responsibility
        for any actions taken.
        """
    )

    st.markdown("---")
    st.caption(
        "© Katta Wealth Insights — Educational use only · No advice · No guarantees"
    )


# ------------------------------------------------------------
# ALWAYS-VISIBLE LEGAL EXPANDER
# ------------------------------------------------------------

def render_legal_expander():
    with st.expander(
        "ℹ️ About, Legal & Disclosures (Always Available)",
        expanded=False,
    ):
        render_about_us_legal()


# ------------------------------------------------------------
# ALWAYS-VISIBLE LEGAL BANNER (BOTTOM)
# ------------------------------------------------------------

def render_legal_banner():
    st.markdown(
        """
        <div style="
            background:#111827;
            border-top:1px solid #374151;
            padding:0.6rem 1rem;
            font-size:0.8rem;
            color:#9ca3af;
            text-align:center;
        ">
            Educational use only · No financial, tax, or legal advice ·
            No guarantees · Investing involves risk
        </div>
        """,
        unsafe_allow_html=True,
    )
# ============================================================
# PART 11 / 12 — ADVANCED EDUCATIONAL FEATURES (SAFE)
# ============================================================

# ------------------------------------------------------------
# LEARNING NOTES / JOURNAL
# ------------------------------------------------------------

def render_learning_notes():
    st.header("📝 Learning Notes")

    st.markdown(
        """
        Use this space to write **your own understanding** of investing concepts,
        portfolio behavior, or lessons learned.
        Notes are stored only for the current session.
        """
    )

    notes = st.text_area(
        "Your notes",
        value=st.session_state.get("learning_notes", ""),
        height=220,
    )

    if st.button("Save Notes"):
        st.session_state.learning_notes = notes
        st.success("Notes saved for this session.")

    if notes:
        st.markdown("---")
        st.subheader("Your Reflection")
        st.markdown(notes)


# ------------------------------------------------------------
# SCENARIO COMPARISON (WHAT-IF EDUCATION)
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
        impact long-term outcomes.  
        These are **not predictions**.
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        return_a = st.slider("Scenario A – Expected Return (%)", 0.0, 12.0, 7.0)
        vol_a = st.slider("Scenario A – Volatility (%)", 5.0, 30.0, 15.0)

    with c2:
        return_b = st.slider("Scenario B – Expected Return (%)", 0.0, 12.0, 5.0)
        vol_b = st.slider("Scenario B – Volatility (%)", 5.0, 30.0, 10.0)

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
        "Results vary each run. This comparison is for learning purposes only."
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

    if st.button("Save Snapshot"):
        st.session_state.snapshots.append(portfolio.copy())
        st.success(f"Snapshot #{len(st.session_state.snapshots)} saved.")

    if not st.session_state.snapshots:
        st.info("No snapshots saved yet.")
        return

    st.subheader("Saved Snapshots")
    for i, snap in enumerate(st.session_state.snapshots):
        with st.expander(f"Snapshot {i + 1}"):
            st.dataframe(snap, use_container_width=True)


# ------------------------------------------------------------
# USER SETTINGS / PREFERENCES
# ------------------------------------------------------------

def render_settings():
    st.header("⚙️ Preferences & Settings")

    show_tips = st.checkbox(
        "Show educational tips",
        value=st.session_state.get("show_tips", True),
    )

    theme = st.selectbox(
        "Preferred theme (visual only)",
        ["Default", "Dark", "Light"],
        index=0,
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
        "learning_checklist", default_checklist.copy()
    )

    for item in checklist:
        checklist[item] = st.checkbox(item, value=checklist[item])

    st.session_state.learning_checklist = checklist

    completed = sum(checklist.values())
    total = len(checklist)

    st.metric("Progress", f"{completed} / {total} topics completed")


# ------------------------------------------------------------
# EXTEND DEMO ROUTER WITH NEW FEATURES
# ------------------------------------------------------------

def demo_router_extended(page):
    # Existing routes
    demo_router(page)

    # New educational routes
    if page == "Learning Notes":
        render_learning_notes()

    elif page == "Scenario Comparison":
        render_scenario_comparison()

    elif page == "Snapshots":
        render_portfolio_snapshots()

    elif page == "Settings":
        render_settings()

    elif page == "Learning Checklist":
        render_learning_checklist()
# ============================================================
# PART 12 / 12 — FINAL ROUTER & ENTRYPOINT
# ============================================================

# ------------------------------------------------------------
# FINAL APPLICATION ROUTER (SINGLE SOURCE OF TRUTH)
# ------------------------------------------------------------

def route_app():
    """
    Authoritative router.
    - Top navigation always visible
    - Legal banner + expander ALWAYS visible
    - Marketing, demo, and auth handled safely
    """

    # ---------------------------
    # TOP NAV (ALWAYS)
    # ---------------------------
    render_top_nav()

    mode = st.session_state.get("app_mode", "about")

    # ---------------------------
    # MARKETING / INFO PAGES
    # ---------------------------
    if mode == "about":
        render_about()

    elif mode == "features":
        render_features()

    elif mode == "how":
        render_how()

    elif mode == "benefits":
        render_benefits()

    elif mode == "legal":
        render_about_us_legal()

    # ---------------------------
    # DEMO MODE (AUTH + PAYWALL)
    # ---------------------------
    elif mode == "demo":

        if not enforce_auth_and_payment():
            # Even when blocked, legal must still show
            render_legal_expander()
            render_legal_banner()
            return

        # Sidebar + demo routing
        page = render_sidebar()

        # Use EXTENDED router if available
        try:
            demo_router_extended(page)
        except Exception:
            demo_router(page)

    # ---------------------------
    # FALLBACK
    # ---------------------------
    else:
        render_about()

    # ---------------------------
    # ALWAYS-VISIBLE LEGAL (MANDATORY)
    # ---------------------------
    render_legal_expander()
    render_legal_banner()


# ------------------------------------------------------------
# SINGLE ENTRYPOINT (ONLY ONE)
# ------------------------------------------------------------

route_app()
