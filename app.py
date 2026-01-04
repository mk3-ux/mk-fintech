# ============================================================
# KATTA WEALTH INSIGHTS — SINGLE FILE STREAMLIT APP
# PART 1 / 15 — BOOTSTRAP & CONFIG
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import datetime as dt
import math
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# ============================================================
# STREAMLIT CONFIG (MUST BE FIRST STREAMLIT CALL)
# ============================================================

st.set_page_config(
    page_title="Katta Wealth Insights",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL CONSTANTS
# ============================================================

APP_NAME = "Katta Wealth Insights"
APP_VERSION = "1.0.0"
APP_MODE_DEFAULT = "dashboard"

DB_PATH = "kwi_app.db"

# Financial assumptions (educational defaults)
TRADING_DAYS = 252
DEFAULT_RETURN = 0.07
DEFAULT_VOLATILITY = 0.15

# LLM configuration flags
ENABLE_LLM = True
LLM_PROVIDER = "openai"  # "openai" or "groq"

# ============================================================
# UTILITY HELPERS
# ============================================================

def now_utc() -> str:
    return dt.datetime.utcnow().isoformat()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
# ============================================================
# PART 2 / 15 — SESSION STATE & APP SCHEMA
# ============================================================

def init_session_state() -> None:
    """
    Initialize ALL session keys exactly once.
    This function is idempotent and safe on reruns.
    """

    defaults: Dict[str, Any] = {
        # Navigation
        "current_page": APP_MODE_DEFAULT,

        # User / auth (future-safe)
        "current_user": None,
        "is_paid": False,

        # Portfolio
        "portfolio_raw": None,
        "portfolio_df": None,
        "portfolio_metrics": {},
        "portfolio_history": [],

        # Snapshots
        "snapshots": [],

        # Learning / notes
        "learning_notes": "",
        "learning_checklist": {},

        # Chatbot
        "chat_history": [],
        "use_llm": ENABLE_LLM,

        # UI preferences
        "show_tips": True,
        "theme": "Default",

        # System
        "alerts": [],
        "debug": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Initialize session state immediately
init_session_state()
# ============================================================
# PART 3 / 15 — FEATURE MODEL & REGISTRY
# ============================================================

@dataclass
class Feature:
    """
    Declarative feature definition.
    Used for navigation, capability control, and UI labeling.
    """
    key: str
    name: str
    description: str
    enabled: bool = True


# ------------------------------------------------------------
# FEATURE REGISTRY (SINGLE SOURCE OF TRUTH)
# ------------------------------------------------------------

FEATURES: List[Feature] = [
    Feature(
        key="dashboard",
        name="Dashboard",
        description="Overview of active features and system status",
    ),
    Feature(
        key="portfolio_overview",
        name="Portfolio Overview",
        description="Holdings, values, and allocation",
    ),
    Feature(
        key="portfolio_math",
        name="Portfolio Analytics",
        description="Returns, weights, and performance metrics",
    ),
    Feature(
        key="risk_metrics",
        name="Risk Metrics",
        description="Volatility, drawdown, and risk indicators",
    ),
    Feature(
        key="monte_carlo",
        name="Goal Probability",
        description="Monte Carlo simulations for long-term goals",
    ),
    Feature(
        key="income",
        name="Income Forecast",
        description="Dividend and income modeling",
    ),
    Feature(
        key="chatbot",
        name="AI Chatbot",
        description="Educational chatbot with guardrails",
    ),
    Feature(
        key="snapshots",
        name="Snapshots",
        description="Saved portfolio states over time",
    ),
    Feature(
        key="exports",
        name="Exports",
        description="Downloadable reports and CSVs",
    ),
]


# ------------------------------------------------------------
# FEATURE HELPERS
# ------------------------------------------------------------

def get_enabled_features() -> List[Feature]:
    return [f for f in FEATURES if f.enabled]


def get_feature_by_key(key: str) -> Optional[Feature]:
    for f in FEATURES:
        if f.key == key:
            return f
    return None
# ============================================================
# PART 4 / 15 — NAVIGATION + ROUTER
# ============================================================

# ------------------------------------------------------------
# SIDEBAR NAVIGATION (DERIVED FROM FEATURE REGISTRY)
# ------------------------------------------------------------

def render_sidebar() -> str:
    """
    Sidebar navigation driven by enabled FEATURES.
    Returns the selected feature key.
    """
    with st.sidebar:
        st.markdown(f"## 💎 {APP_NAME}")
        st.caption(f"Version {APP_VERSION}")

        page_labels = [f.name for f in get_enabled_features()]
        page_keys = [f.key for f in get_enabled_features()]

        selection = st.radio(
            "Navigate",
            options=page_labels,
            index=page_keys.index(
                st.session_state.current_page
                if st.session_state.current_page in page_keys
                else APP_MODE_DEFAULT
            ),
            key="sidebar_nav_radio",
        )

        # Map label back to key
        selected_key = page_keys[page_labels.index(selection)]

        st.session_state.current_page = selected_key

        st.markdown("---")
        st.caption("Educational platform only")

    return selected_key


# ------------------------------------------------------------
# DASHBOARD (FEATURE STATUS VIEW)
# ------------------------------------------------------------

def render_dashboard():
    st.header("📊 Dashboard")
    st.markdown(
        """
        This dashboard shows which educational features are currently active.
        """
    )

    for feature in get_enabled_features():
        st.success(f"✅ {feature.name}")
        st.caption(feature.description)


# ------------------------------------------------------------
# CENTRAL ROUTER (NO BUSINESS LOGIC)
# ------------------------------------------------------------

def route_feature(page_key: str):
    """
    Routes execution based on feature key.
    This function NEVER defines UI state.
    """

    if page_key == "dashboard":
        render_dashboard()

    elif page_key == "portfolio_overview":
        render_portfolio_overview()

    elif page_key == "portfolio_math":
        render_portfolio_math()

    elif page_key == "risk_metrics":
        render_risk_metrics()

    elif page_key == "monte_carlo":
        render_monte_carlo()

    elif page_key == "income":
        render_income_forecast()

    elif page_key == "chatbot":
        render_chatbot()

    elif page_key == "snapshots":
        render_snapshots()

    elif page_key == "exports":
        render_exports()

    else:
        st.warning("Feature not implemented.")
# ============================================================
# PART 5 / 15 — LEGAL & COMPLIANCE (ALWAYS VISIBLE)
# ============================================================

# ------------------------------------------------------------
# LEGAL CONTENT (CENTRALIZED)
# ------------------------------------------------------------

def render_legal_content():
    st.markdown("### ℹ️ About & Legal Disclosures")

    st.markdown(
        """
        **Katta Wealth Insights** is an education-first financial analytics platform.

        #### Educational Use Only
        This platform is provided strictly for **educational and informational purposes**.
        It does **not** provide financial, investment, tax, or legal advice.

        #### No Investment Advice
        Nothing presented constitutes a recommendation to buy, sell,
        or hold any security or financial product.

        #### Risk Disclosure
        Investing involves risk, including the possible loss of principal.
        Past performance, simulations, or hypothetical results
        do **not** guarantee future outcomes.

        #### Hypothetical & Simulated Results
        Monte Carlo simulations and forecasts are **hypothetical**
        and rely on assumptions that may not reflect real-world conditions.

        #### User Responsibility
        Users should consult qualified professionals before making financial decisions.
        """
    )


# ------------------------------------------------------------
# LEGAL EXPANDER (TOP-LEVEL, ALWAYS AVAILABLE)
# ------------------------------------------------------------

def render_legal_expander():
    with st.expander(
        "ℹ️ About, Legal & Disclosures (Always Available)",
        expanded=False,
    ):
        render_legal_content()


# ------------------------------------------------------------
# LEGAL BANNER (BOTTOM, ALWAYS SHOWN)
# ------------------------------------------------------------

def render_legal_banner():
    st.markdown(
        """
        <div style="
            margin-top:2rem;
            padding:0.6rem;
            font-size:0.8rem;
            color:#9ca3af;
            text-align:center;
            border-top:1px solid #374151;
        ">
            Educational use only · No financial, tax, or legal advice ·
            No guarantees · Investing involves risk
        </div>
        """,
        unsafe_allow_html=True,
    )
# ============================================================
# PART 6 / 15 — PORTFOLIO DATA MODEL & INPUT
# ============================================================

# ------------------------------------------------------------
# PORTFOLIO SCHEMA
# ------------------------------------------------------------

REQUIRED_PORTFOLIO_COLUMNS = [
    "Ticker",
    "Shares",
    "Price",
]


# ------------------------------------------------------------
# PORTFOLIO TEMPLATE
# ------------------------------------------------------------

def get_portfolio_template() -> bytes:
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "NVDA"],
            "Shares": [10, 5, 8],
            "Price": [180.0, 410.0, 950.0],
        }
    )
    return df.to_csv(index=False).encode("utf-8")


# ------------------------------------------------------------
# PORTFOLIO VALIDATION
# ------------------------------------------------------------

def validate_portfolio(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return "Portfolio is empty."

    missing = [c for c in REQUIRED_PORTFOLIO_COLUMNS if c not in df.columns]
    if missing:
        return f"Missing columns: {', '.join(missing)}"

    try:
        df["Shares"] = pd.to_numeric(df["Shares"])
        df["Price"] = pd.to_numeric(df["Price"])
    except Exception:
        return "Shares and Price must be numeric."

    if (df["Shares"] <= 0).any():
        return "Shares must be greater than zero."

    if (df["Price"] <= 0).any():
        return "Price must be greater than zero."

    return None


# ------------------------------------------------------------
# PORTFOLIO OVERVIEW PAGE
# ------------------------------------------------------------

def render_portfolio_overview():
    st.header("📈 Portfolio Overview")

    st.download_button(
        "Download Portfolio Template",
        data=get_portfolio_template(),
        file_name="portfolio_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader(
        "Upload Portfolio CSV",
        type=["csv"],
        key="portfolio_upload",
    )

    if uploaded is None:
        st.info("Upload a portfolio to begin.")
        return

    df = pd.read_csv(uploaded)
    error = validate_portfolio(df)

    if error:
        st.error(error)
        return

    st.session_state.portfolio_raw = df.copy()
    st.dataframe(df, use_container_width=True)

    st.success("Portfolio loaded successfully.")
# ============================================================
# PART 7 / 15 — REAL PORTFOLIO MATH
# ============================================================

# ------------------------------------------------------------
# CORE CALCULATIONS
# ------------------------------------------------------------

def compute_portfolio_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes:
    - Market Value
    - Weights
    - Position Value
    """
    out = df.copy()

    out["MarketValue"] = out["Shares"] * out["Price"]
    total_value = out["MarketValue"].sum()

    out["Weight"] = out["MarketValue"] / total_value

    return out


def compute_portfolio_return(
    weights: np.ndarray,
    returns: np.ndarray,
) -> float:
    """
    Weighted portfolio return.
    """
    return float(np.dot(weights, returns))


def compute_portfolio_volatility(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    Portfolio volatility = sqrt(w.T * Cov * w)
    """
    return float(np.sqrt(weights.T @ cov_matrix @ weights))


# ------------------------------------------------------------
# PORTFOLIO ANALYTICS PAGE
# ------------------------------------------------------------

def render_portfolio_math():
    st.header("📊 Portfolio Analytics")

    df = st.session_state.get("portfolio_raw")
    if df is None:
        st.info("Upload a portfolio first.")
        return

    metrics = compute_portfolio_metrics(df)
    st.session_state.portfolio_df = metrics

    st.subheader("Holdings & Weights")
    st.dataframe(
        metrics[
            ["Ticker", "Shares", "Price", "MarketValue", "Weight"]
        ],
        use_container_width=True,
    )

    st.metric(
        "Total Portfolio Value",
        f"${metrics['MarketValue'].sum():,.2f}",
    )

    st.caption(
        "Returns and volatility calculations are educational and use user-provided prices."
    )
# ============================================================
# PART 8 / 15 — RISK METRICS
# ============================================================

# ------------------------------------------------------------
# RISK HELPERS
# ------------------------------------------------------------

def compute_concentration_index(weights: np.ndarray) -> float:
    """
    Herfindahl-Hirschman Index (HHI)
    Higher = more concentrated
    """
    return float(np.sum(weights ** 2))


def compute_max_drawdown(values: np.ndarray) -> float:
    """
    Max drawdown from a value series
    """
    peak = values[0]
    max_dd = 0.0

    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    return round(max_dd * 100, 2)


def simulate_simple_path(
    start_value: float,
    years: int,
    exp_return: float,
    volatility: float,
) -> np.ndarray:
    """
    Single simulated growth path
    """
    values = [start_value]
    value = start_value

    for _ in range(years):
        shock = np.random.normal(exp_return, volatility)
        value = value * (1 + shock)
        values.append(value)

    return np.array(values)


# ------------------------------------------------------------
# RISK ANALYTICS PAGE
# ------------------------------------------------------------

def render_risk_metrics():
    st.header("⚠️ Risk Metrics")

    df = st.session_state.get("portfolio_df")
    if df is None:
        st.info("Run portfolio analytics first.")
        return

    weights = df["Weight"].values
    total_value = df["MarketValue"].sum()

    concentration = compute_concentration_index(weights)

    # Educational assumptions
    exp_return = 0.07
    volatility = 0.15
    years = 20

    path = simulate_simple_path(
        start_value=total_value,
        years=years,
        exp_return=exp_return,
        volatility=volatility,
    )

    max_dd = compute_max_drawdown(path)

    c1, c2, c3 = st.columns(3)
    c1.metric("Concentration Index (HHI)", round(concentration, 3))
    c2.metric("Assumed Volatility", "15%")
    c3.metric("Simulated Max Drawdown", f"{max_dd}%")

    st.subheader("Illustrative Value Path")
    st.line_chart(path)

    st.caption(
        "Risk metrics are educational illustrations based on assumptions, not predictions."
    )
# ============================================================
# PART 9 / 15 — MONTE CARLO GOAL ENGINE
# ============================================================

# ------------------------------------------------------------
# MONTE CARLO SIMULATION
# ------------------------------------------------------------

def monte_carlo_simulation(
    start_value: float,
    annual_contribution: float,
    years: int,
    exp_return: float,
    volatility: float,
    simulations: int = 5000,
) -> np.ndarray:
    """
    Monte Carlo simulation of portfolio growth
    """
    results = np.zeros((simulations, years))
    
    for i in range(simulations):
        value = start_value
        for y in range(years):
            shock = np.random.normal(exp_return, volatility)
            value = value * (1 + shock) + annual_contribution
            results[i, y] = value

    return results


def probability_of_goal(simulations: np.ndarray, goal: float) -> float:
    final_vals = simulations[:, -1]
    return round((final_vals >= goal).mean() * 100, 2)


# ------------------------------------------------------------
# GOAL PROBABILITY PAGE
# ------------------------------------------------------------

def render_goal_probability():
    st.header("🎯 Goal Probability")

    df = st.session_state.get("portfolio_df")
    if df is None:
        st.info("Run portfolio analytics first.")
        return

    total_value = df["MarketValue"].sum()

    c1, c2, c3 = st.columns(3)

    with c1:
        goal = st.number_input(
            "Target Amount ($)",
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

    exp_return = 0.07
    volatility = 0.15

    sims = monte_carlo_simulation(
        start_value=total_value,
        annual_contribution=annual,
        years=years,
        exp_return=exp_return,
        volatility=volatility,
    )

    prob = probability_of_goal(sims, goal)

    c1, c2, c3 = st.columns(3)
    c1.metric("Goal Success Probability", f"{prob}%")
    c2.metric("Expected Return", "7%")
    c3.metric("Volatility", "15%")

    st.subheader("Outcome Distribution")
    final_vals = sims[:, -1]

    dist_df = pd.DataFrame(
        {
            "Scenario": ["10th %ile", "Median", "90th %ile"],
            "Ending Value ($)": [
                round(np.percentile(final_vals, 10), 0),
                round(np.percentile(final_vals, 50), 0),
                round(np.percentile(final_vals, 90), 0),
            ],
        }
    )

    st.dataframe(dist_df, use_container_width=True)

    st.subheader("Sample Monte Carlo Paths")
    st.line_chart(pd.DataFrame(sims[:50].T))

    st.caption(
        "Monte Carlo simulations are hypothetical and for education only."
    )
# ============================================================
# PART 10 / 15 — PERSISTENT STORAGE (SQLITE)
# ============================================================

DB_PATH = "kwi_persistent.db"

# ------------------------------------------------------------
# DATABASE CONNECTION
# ------------------------------------------------------------

def db_connect_persistent():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def db_init_persistent():
    conn = db_connect_persistent()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            created_at TEXT,
            total_value REAL,
            data_json TEXT
        );
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            created_at TEXT,
            goal REAL,
            probability REAL,
            years INTEGER,
            result_json TEXT
        );
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            created_at TEXT,
            note_text TEXT
        );
    """)

    conn.commit()
    conn.close()


db_init_persistent()

# ------------------------------------------------------------
# SAVE / LOAD HELPERS
# ------------------------------------------------------------

def save_portfolio_to_db(df: pd.DataFrame):
    user = st.session_state.get("current_user")
    if not user:
        return

    conn = db_connect_persistent()
    conn.execute(
        """
        INSERT INTO portfolios (user_email, created_at, total_value, data_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            user,
            dt.datetime.utcnow().isoformat(),
            float(df["MarketValue"].sum()),
            df.to_json(),
        ),
    )
    conn.commit()
    conn.close()


def load_portfolios_from_db() -> List[pd.DataFrame]:
    user = st.session_state.get("current_user")
    if not user:
        return []

    conn = db_connect_persistent()
    rows = conn.execute(
        """
        SELECT data_json FROM portfolios
        WHERE user_email = ?
        ORDER BY created_at DESC
        """,
        (user,),
    ).fetchall()
    conn.close()

    return [pd.read_json(r[0]) for r in rows]


def save_simulation_result(goal, probability, years, result_dict):
    user = st.session_state.get("current_user")
    if not user:
        return

    conn = db_connect_persistent()
    conn.execute(
        """
        INSERT INTO simulations (user_email, created_at, goal, probability, years, result_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user,
            dt.datetime.utcnow().isoformat(),
            goal,
            probability,
            years,
            json.dumps(result_dict),
        ),
    )
    conn.commit()
    conn.close()
# ============================================================
# PART 11 / 15 — AI CHATBOT (LLM READY)
# ============================================================

import os

# ------------------------------------------------------------
# LLM CONFIG
# ------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------------------------------------------------
# SAFE SYSTEM PROMPT (LEGAL-FIRST)
# ------------------------------------------------------------

SYSTEM_PROMPT = """
You are an educational financial assistant.
You do NOT provide investment advice.
You explain concepts only.
You must always include a disclaimer if discussing risk or returns.
"""

# ------------------------------------------------------------
# CHAT RESPONSE ENGINE
# ------------------------------------------------------------

def generate_llm_response(question: str) -> str:
    """
    Uses OpenAI or Groq if available, otherwise fallback.
    """

    if OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
            )
            return response.choices[0].message.content
        except Exception:
            pass

    if GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)

            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
            )
            return completion.choices[0].message.content
        except Exception:
            pass

    # ----------------------------
    # FALLBACK (RULE-BASED)
    # ----------------------------
    q = question.lower()

    if "risk" in q:
        return (
            "Risk refers to how much investment values can fluctuate. "
            "Higher potential returns usually come with higher risk. "
            "This is educational, not advice."
        )
    if "divers" in q:
        return (
            "Diversification means spreading investments across assets "
            "to reduce the impact of any single investment."
        )
    if "etf" in q:
        return (
            "ETFs are funds that hold many assets, providing diversification "
            "in a single investment vehicle."
        )
    if "goal" in q:
        return (
            "Goal probability estimates how likely a portfolio might reach "
            "a target under assumptions. It is not a prediction."
        )

    return (
        "I can help explain portfolios, diversification, risk, goals, "
        "and long-term investing concepts."
    )


# ------------------------------------------------------------
# CHATBOT UI
# ------------------------------------------------------------

def render_chatbot():
    st.header("🤖 Educational AI Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Ask a question about investing concepts")

    if st.button("Ask", key="chatbot_ask"):
        answer = generate_llm_response(question)
        st.session_state.chat_history.append(
            {"q": question, "a": answer}
        )

    st.markdown("---")

    for msg in reversed(st.session_state.chat_history[-8:]):
        st.markdown(f"**You:** {msg['q']}")
        st.markdown(f"**AI:** {msg['a']}")
        st.markdown("---")

    st.caption(
        "Educational use only · No investment, tax, or legal advice."
    )
# ============================================================
# PART 12 / 15 — ADVANCED PORTFOLIO MATH
# ============================================================

# ------------------------------------------------------------
# RETURN SERIES (EDUCATIONAL SYNTHETIC)
# ------------------------------------------------------------

def generate_synthetic_returns(n_assets: int, periods: int = 252):
    """
    Generate synthetic correlated returns for education.
    """
    rng = np.random.default_rng(42)
    base = rng.normal(0.0004, 0.01, size=(periods, 1))
    noise = rng.normal(0, 0.008, size=(periods, n_assets))
    returns = base + noise
    return returns


def compute_correlation_matrix(returns: np.ndarray) -> pd.DataFrame:
    corr = np.corrcoef(returns.T)
    return pd.DataFrame(corr)


def compute_portfolio_beta(weights: np.ndarray, asset_betas: np.ndarray) -> float:
    return float(np.dot(weights, asset_betas))


def render_correlation_and_beta():
    st.header("📐 Correlation & Beta (Educational)")

    df = st.session_state.get("portfolio_df")
    if df is None:
        st.info("Run portfolio analytics first.")
        return

    tickers = df["Ticker"].tolist()
    weights = df["Weight"].values

    returns = generate_synthetic_returns(len(tickers))
    corr_df = compute_correlation_matrix(returns)
    corr_df.columns = tickers
    corr_df.index = tickers

    st.subheader("Correlation Matrix")
    st.dataframe(corr_df.round(2), use_container_width=True)

    # Synthetic betas (educational)
    asset_betas = np.clip(np.random.normal(1.0, 0.2, size=len(tickers)), 0.5, 1.8)
    beta = compute_portfolio_beta(weights, asset_betas)

    st.metric("Estimated Portfolio Beta", round(beta, 2))

    st.caption(
        "Correlation and beta shown are educational simulations, not real market estimates."
    )
# ============================================================
# PART 13 / 15 — DEPLOYMENT PREP (AWS / STREAMLIT CLOUD)
# ============================================================

# ------------------------------------------------------------
# ENVIRONMENT CHECKS
# ------------------------------------------------------------

def render_deployment_info():
    st.header("🚀 Deployment Readiness")

    st.markdown(
        """
        ### Streamlit Cloud
        - Add secrets in **Settings → Secrets**
        - Keys supported:
          - `OPENAI_API_KEY`
          - `GROQ_API_KEY`

        ### AWS (Optional)
        - EC2 / ECS / App Runner supported
        - SQLite works locally; switch to RDS for scale
        - Use IAM roles for secrets
        """
    )

    st.subheader("Environment Variables Detected")
    envs = {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
    }

    for k, v in envs.items():
        st.write(f"{k}: {'✅ Found' if v else '❌ Not set'}")

    st.caption(
        "Missing keys will automatically fall back to safe, rule-based behavior."
    )
# ============================================================
# PART 14 / 15 — LEGAL & COMPLIANCE (ALWAYS ON)
# ============================================================

LEGAL_BANNER_HTML = """
<div style="
  background:#0b1220;
  border-top:1px solid #374151;
  padding:0.6rem 1rem;
  font-size:0.8rem;
  color:#9ca3af;
  text-align:center;
">
  Educational use only · No financial, investment, tax, or legal advice ·
  Investing involves risk · No guarantees
</div>
"""

def render_legal_always_on():
    with st.expander("ℹ️ About, Legal & Disclosures (Always Available)", expanded=False):
        render_about_us_legal()

    st.markdown(LEGAL_BANNER_HTML, unsafe_allow_html=True)
# ============================================================
# PART 15 / 15 — FINAL ROUTER & ENTRYPOINT
# ============================================================

def render_marketing_router(mode: str):
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
    elif mode == "deploy":
        render_deployment_info()
    else:
        render_about()


def render_demo_router(page: str):
    if page == "Portfolio Overview":
        render_portfolio_overview()
    elif page == "Portfolio Insights":
        render_portfolio_insights()
    elif page == "Risk Metrics":
        render_risk_metrics()
    elif page == "Goal Probability":
        render_goal_probability()
    elif page == "Correlation & Beta":
        render_correlation_and_beta()
    elif page == "AI Chatbot":
        render_chatbot()
    else:
        st.info("Select a page from the sidebar.")


def route_app():
    # ---------------------------
    # INIT (SAFE)
    # ---------------------------
    init_session()

    # ---------------------------
    # TOP NAV (ALWAYS)
    # ---------------------------
    render_top_nav()

    mode = st.session_state.get("app_mode", "about")

    # ---------------------------
    # MARKETING / INFO
    # ---------------------------
    if mode in {"about", "features", "how", "benefits", "legal", "deploy"}:
        render_marketing_router(mode)
        render_legal_always_on()
        return

    # ---------------------------
    # DEMO (AUTH + PAYWALL)
    # ---------------------------
    if mode == "demo":
        if not enforce_auth_and_paywall():
            render_legal_always_on()
            return

        page = render_sidebar()
        render_demo_router(page)
        render_legal_always_on()
        return

    # ---------------------------
    # FALLBACK
    # ---------------------------
    render_about()
    render_legal_always_on()


# ------------------------------------------------------------
# SINGLE ENTRYPOINT (ONLY ONE)
# ------------------------------------------------------------
route_app()
