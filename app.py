# ============================================================
# KATTA WEALTH INSIGHTS — OPTION B CLEAN REBUILD
# PART 1 / 15 — CORE CONFIG, SESSION, LEGAL (DO NOT RUN YET)
# ============================================================

from __future__ import annotations

import os
import json
import sqlite3
import hashlib
import datetime as dt
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional deps
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from groq import Groq
except Exception:
    Groq = None

# ============================================================
# APP CONFIG
# ============================================================

APP_NAME = "Katta Wealth Insights"
APP_VERSION = "2.0.0"
DB_PATH = "kwi.db"
MODEL_NAME = "llama-3.1-8b-instant"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE (AUTHORITATIVE)
# ============================================================

def init_session():
    defaults = {
        "user": None,
        "tier": "Free",
        "portfolio_raw": None,
        "portfolio": None,
        "meta": {},
        "chat": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_session()

# ============================================================
# LEGAL — ALWAYS VISIBLE
# ============================================================

def render_legal_banner():
    st.markdown(
        """
        <div style="
            background:#111827;
            color:#9ca3af;
            font-size:0.75rem;
            padding:0.6rem;
            text-align:center;
            border-top:1px solid #374151;
            margin-top:2rem;">
            Educational use only · No financial, tax, or legal advice ·
            Investing involves risk · Hypothetical results
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_legal_expander():
    with st.expander("ℹ️ About & Legal Disclosures", expanded=False):
        st.markdown(
            """
            **Educational Use Only**

            Katta Wealth Insights is an educational analytics platform.
            It does not provide financial, investment, tax, or legal advice.

            **No Fiduciary Relationship**

            Use of this platform does not create any advisory or fiduciary relationship.

            **Risk Disclosure**

            Investing involves risk, including loss of principal.
            Past performance and simulations do not guarantee future results.

            **Hypothetical Results**

            Monte Carlo simulations and forecasts are hypothetical and illustrative only.
            """
        )

# ============================================================
# BASIC HELPERS
# ============================================================

def logged_in() -> bool:
    return st.session_state.user is not None

def is_pro() -> bool:
    return st.session_state.tier == "Pro"

def hash_pw(pw: str) -> str:
    return hashlib.sha256(("kwi_" + pw).encode()).hexdigest()
# ============================================================
# KATTA WEALTH INSIGHTS — OPTION B CLEAN REBUILD
# PART 1 / 15 — CORE CONFIG, SESSION, LEGAL (DO NOT RUN YET)
# ============================================================

from __future__ import annotations

import os
import json
import sqlite3
import hashlib
import datetime as dt
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional deps
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from groq import Groq
except Exception:
    Groq = None

# ============================================================
# APP CONFIG
# ============================================================

APP_NAME = "Katta Wealth Insights"
APP_VERSION = "2.0.0"
DB_PATH = "kwi.db"
MODEL_NAME = "llama-3.1-8b-instant"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE (AUTHORITATIVE)
# ============================================================

def init_session():
    defaults = {
        "user": None,
        "tier": "Free",
        "portfolio_raw": None,
        "portfolio": None,
        "meta": {},
        "chat": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_session()

# ============================================================
# LEGAL — ALWAYS VISIBLE
# ============================================================

def render_legal_banner():
    st.markdown(
        """
        <div style="
            background:#111827;
            color:#9ca3af;
            font-size:0.75rem;
            padding:0.6rem;
            text-align:center;
            border-top:1px solid #374151;
            margin-top:2rem;">
            Educational use only · No financial, tax, or legal advice ·
            Investing involves risk · Hypothetical results
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_legal_expander():
    with st.expander("ℹ️ About & Legal Disclosures", expanded=False):
        st.markdown(
            """
            **Educational Use Only**

            Katta Wealth Insights is an educational analytics platform.
            It does not provide financial, investment, tax, or legal advice.

            **No Fiduciary Relationship**

            Use of this platform does not create any advisory or fiduciary relationship.

            **Risk Disclosure**

            Investing involves risk, including loss of principal.
            Past performance and simulations do not guarantee future results.

            **Hypothetical Results**

            Monte Carlo simulations and forecasts are hypothetical and illustrative only.
            """
        )

# ============================================================
# BASIC HELPERS
# ============================================================

def logged_in() -> bool:
    return st.session_state.user is not None

def is_pro() -> bool:
    return st.session_state.tier == "Pro"

def hash_pw(pw: str) -> str:
    return hashlib.sha256(("kwi_" + pw).encode()).hexdigest()
# ============================================================
# PART 2 / 15 — DATABASE & AUTH (SAFE, ORDERED)
# ============================================================

def db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def db_init():
    conn = db_connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            pw_hash TEXT NOT NULL,
            tier TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

db_init()

def db_get_user(email: str) -> Optional[Dict[str, Any]]:
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
    try:
        conn = db_connect()
        conn.execute(
            "INSERT INTO users VALUES (?,?,?,?)",
            (email, pw_hash, "Free", dt.datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def db_set_tier(email: str, tier: str):
    conn = db_connect()
    conn.execute(
        "UPDATE users SET tier=? WHERE email=?",
        (tier, email),
    )
    conn.commit()
    conn.close()

def auth_ui():
    st.header("🔐 Sign In")

    login, signup = st.tabs(["Log In", "Sign Up"])

    with login:
        email = st.text_input("Email", key="login_email")
        pw = st.text_input("Password", type="password", key="login_pw")

        if st.button("Log In"):
            user = db_get_user(email)
            if not user or user["pw"] != hash_pw(pw):
                st.error("Invalid credentials")
                return
            st.session_state.user = email
            st.session_state.tier = user["tier"]
            st.rerun()

    with signup:
        email = st.text_input("New Email", key="signup_email")
        pw = st.text_input("New Password", type="password", key="signup_pw")

        if st.button("Create Account"):
            if not db_create_user(email, hash_pw(pw)):
                st.error("Account exists")
                return
            st.success("Account created. Please log in.")
# ============================================================
# PART 4 / 15 — PORTFOLIO UI
# ============================================================

def render_portfolio():
    st.header("📊 Portfolio Overview")

    st.download_button(
        "Download CSV Template",
        portfolio_template(),
        "portfolio_template.csv",
    )

    file = st.file_uploader("Upload Portfolio CSV", type="csv")
    if not file:
        return

    raw = pd.read_csv(file)
    ok, msg = validate_portfolio(raw)
    if not ok:
        st.error(msg)
        return

    portfolio = compute_portfolio(raw)
    st.session_state.portfolio = portfolio

    st.dataframe(portfolio, use_container_width=True)

    fig, ax = plt.subplots()
    portfolio.drop(index="TOTAL").set_index("Ticker")["Market Value"].plot.pie(
        autopct="%1.1f%%", ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)
# ============================================================
# PART 5 / 15 — MONTE CARLO ENGINE
# ============================================================

def monte_carlo(start, contrib, years, r=0.07, vol=0.15, sims=3000):
    results = np.zeros((sims, years))
    for i in range(sims):
        val = start
        for y in range(years):
            val = val * (1 + np.random.normal(r, vol)) + contrib
            results[i, y] = val
    return results

def render_goals():
    st.header("🎯 Goal Probability")

    p = st.session_state.get("portfolio")
    if p is None:
        return

    start = float(p.loc["TOTAL", "Market Value"])
    goal = st.number_input("Goal ($)", value=1_000_000.0)
    years = st.slider("Years", 5, 40, 20)

    sims = monte_carlo(start, 0, years)
    prob = (sims[:, -1] >= goal).mean() * 100

    st.metric("Success Probability", f"{prob:.1f}%")
# ============================================================
# PART 6 / 15 — AI CHATBOT (GROQ)
# ============================================================

_groq = None

def ai(prompt: str) -> str:
    global _groq
    if not is_pro():
        return "Upgrade to Pro to use AI."
    if Groq is None or not GROQ_API_KEY:
        return "Groq API key missing."
    if _groq is None:
        _groq = Groq(api_key=GROQ_API_KEY)

    resp = _groq.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Educational finance assistant. No advice."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600,
    )
    return resp.choices[0].message.content

def render_chatbot():
    st.header("💬 AI Chatbot")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for m in st.session_state.chat:
        st.markdown(f"**{m['role']}:** {m['content']}")

    q = st.text_input("Ask a question")
    if st.button("Send") and q:
        st.session_state.chat.append({"role": "You", "content": q})
        ans = ai(q)
        st.session_state.chat.append({"role": "AI", "content": ans})
        st.rerun()
# ============================================================
# PART 7 / 15 — ROUTER & ENTRYPOINT
# ============================================================

def sidebar():
    with st.sidebar:
        st.markdown("## 💎 Katta Wealth")

        if logged_in():
            st.caption(st.session_state.user)
            st.caption(st.session_state.tier)

        page = st.radio(
            "Navigate",
            ["Portfolio", "Goals", "AI Chatbot"],
        )

        if not is_pro():
            if st.button("Upgrade to Pro"):
                db_set_tier(st.session_state.user, "Pro")
                st.session_state.tier = "Pro"
                st.rerun()

        if logged_in() and st.button("Log out"):
            st.session_state.user = None
            st.session_state.tier = "Free"
            st.rerun()

        return page

def run_app():
    if not logged_in():
        auth_ui()
        render_legal_expander()
        render_legal_banner()
        return

    page = sidebar()

    if page == "Portfolio":
        render_portfolio()
    elif page == "Goals":
        render_goals()
    elif page == "AI Chatbot":
        render_chatbot()

    render_legal_expander()
    render_legal_banner()

if __name__ == "__main__":
    run_app()
# ============================================================
# PART 8 / 15 — ETF LOOK-THROUGH ENGINE
# ============================================================

KNOWN_ETFS = {"SPY", "VOO", "VTI", "QQQ", "IVV", "DIA"}

def is_etf(ticker: str) -> bool:
    if yf is None:
        return ticker in KNOWN_ETFS
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("quoteType") == "ETF" or ticker in KNOWN_ETFS
    except Exception:
        return ticker in KNOWN_ETFS


@st.cache_data(ttl=86400)
def get_etf_holdings(ticker: str, top_n: int = 10) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()

    try:
        t = yf.Ticker(ticker)
        holdings = getattr(t, "fund_holdings", None)
        if holdings is None or holdings.empty:
            return pd.DataFrame()

        df = holdings[["symbol", "holdingPercent"]].dropna()
        df.columns = ["Ticker", "Weight"]
        df["Weight"] = df["Weight"] / 100
        return df.head(top_n)
    except Exception:
        return pd.DataFrame()


def compute_lookthrough(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_value = portfolio.loc["TOTAL", "Market Value"]

    for _, r in portfolio.drop(index="TOTAL").iterrows():
        ticker = r["Ticker"]
        weight = r["Market Value"] / total_value

        if is_etf(ticker):
            h = get_etf_holdings(ticker)
            for _, row in h.iterrows():
                rows.append({
                    "Ticker": row["Ticker"],
                    "Weight": weight * row["Weight"],
                })
        else:
            rows.append({"Ticker": ticker, "Weight": weight})

    df = pd.DataFrame(rows)
    df = df.groupby("Ticker", as_index=False).sum()
    df["Weight %"] = (df["Weight"] * 100).round(2)
    return df.sort_values("Weight %", ascending=False)


def render_lookthrough():
    st.header("🧬 ETF Look-Through")

    p = st.session_state.get("portfolio")
    if p is None:
        st.info("Upload a portfolio first.")
        return

    df = compute_lookthrough(p)
    st.dataframe(df.head(15), use_container_width=True)
# ============================================================
# PART 9 / 15 — DIVIDEND & INCOME ENGINE
# ============================================================

@st.cache_data(ttl=3600)
def get_dividends(ticker: str) -> float:
    if yf is None:
        return 0.0
    try:
        divs = yf.Ticker(ticker).dividends
        if divs.empty:
            return 0.0
        cutoff = pd.Timestamp.today() - pd.DateOffset(years=1)
        return float(divs[divs.index >= cutoff].sum())
    except Exception:
        return 0.0


def compute_income(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in portfolio.drop(index="TOTAL").iterrows():
        dps = get_dividends(r["Ticker"])
        income = dps * r["Shares"]
        rows.append({
            "Ticker": r["Ticker"],
            "Annual Income": round(income, 2),
        })

    df = pd.DataFrame(rows)
    df.loc["TOTAL", "Annual Income"] = df["Annual Income"].sum()
    return df


def render_income():
    st.header("💵 Dividend Income")

    p = st.session_state.get("portfolio")
    if p is None:
        return

    df = compute_income(p)
    total = df.loc["TOTAL", "Annual Income"]

    st.metric("Annual Income", f"${total:,.2f}")
    st.dataframe(df, use_container_width=True)
# ============================================================
# PART 10 / 15 — RISK & DRAWDOWN
# ============================================================

def max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    dd = (series - peak) / peak
    return round(dd.min() * 100, 2)


def render_risk():
    st.header("🚨 Risk Analysis")

    p = st.session_state.get("portfolio")
    if p is None or yf is None:
        st.info("Portfolio or data unavailable.")
        return

    prices = []
    for _, r in p.drop(index="TOTAL").iterrows():
        try:
            s = yf.Ticker(r["Ticker"]).history(period="1y")["Close"]
            prices.append(s)
        except Exception:
            pass

    if not prices:
        st.warning("Not enough data.")
        return

    combined = pd.concat(prices, axis=1).mean(axis=1)
    dd = max_drawdown(combined)

    st.metric("Max Drawdown (1Y)", f"{dd}%")

    if is_pro():
        st.markdown(ai(f"Explain a {dd}% drawdown for a long-term investor."))
# ============================================================
# PART 11 / 15 — TAX AWARENESS (EDUCATIONAL)
# ============================================================

def render_tax():
    st.header("🧾 Tax Awareness")

    p = st.session_state.get("portfolio")
    if p is None:
        return

    rate = st.slider("Capital Gains Tax Rate", 0.0, 0.4, 0.15)
    gains = max(p.loc["TOTAL", "PnL"], 0)
    tax = gains * rate

    st.metric("Estimated Tax", f"${tax:,.2f}")

    if is_pro():
        st.markdown(ai("Explain tax-efficient investing at a high level."))
# ============================================================
# PART 12 / 15 — PERFORMANCE BENCHMARKING
# ============================================================

def render_performance():
    st.header("📈 Performance vs S&P 500")

    p = st.session_state.get("portfolio")
    if p is None or yf is None:
        return

    prices = []
    for _, r in p.drop(index="TOTAL").iterrows():
        try:
            prices.append(yf.Ticker(r["Ticker"]).history(period="1y")["Close"])
        except Exception:
            pass

    if not prices:
        return

    port = pd.concat(prices, axis=1).mean(axis=1)
    spy = yf.Ticker("SPY").history(period="1y")["Close"]

    df = pd.DataFrame({
        "Portfolio": port / port.iloc[0] - 1,
        "S&P 500": spy / spy.iloc[0] - 1,
    })

    st.line_chart(df)
# ============================================================
# PART 13 / 15 — EXPORTS & REPORTING
# ============================================================

def render_exports():
    st.header("📤 Exports & Reports")

    p = st.session_state.get("portfolio")
    if p is None:
        st.info("Upload a portfolio first.")
        return

    st.download_button(
        "⬇️ Download Portfolio CSV",
        p.to_csv().encode("utf-8"),
        file_name="portfolio_snapshot.csv",
        mime="text/csv",
    )

    snapshot = {
        "generated_at": dt.datetime.utcnow().isoformat(),
        "user": st.session_state.user,
        "tier": st.session_state.tier,
        "portfolio": p.to_dict(),
    }

    st.download_button(
        "⬇️ Download Portfolio JSON",
        json.dumps(snapshot, indent=2).encode("utf-8"),
        file_name="portfolio_snapshot.json",
        mime="application/json",
    )

    st.success("Exports generated successfully.")


def persist_portfolio():
    """
    Optional persistence hook (can later write to SQLite artifacts table).
    Currently stored in session only for safety.
    """
    if "portfolio" in st.session_state:
        st.session_state.portfolio_meta["last_saved"] = (
            dt.datetime.utcnow().isoformat()
        )
# ============================================================
# PART 14 / 15 — LEGAL & COMPLIANCE (ALWAYS ON)
# ============================================================

LEGAL_TEXT = """
**Important Legal Notice**

Katta Wealth Insights is an **educational and research platform only**.

• This application does **NOT** provide financial, investment, tax, or legal advice  
• All analytics, simulations, and AI-generated content are **informational only**  
• Past performance does **not** guarantee future results  
• You are solely responsible for any financial decisions you make  

Consult a qualified financial advisor before making investment decisions.
"""

def render_legal_banner():
    st.markdown(
        """
        <div style="
            background:#fef3c7;
            padding:0.6rem 1rem;
            border-radius:0.5rem;
            margin-top:1rem;
            font-size:0.85rem;
            color:#92400e;
        ">
        ⚠️ Educational use only — no investment advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_legal_expander():
    with st.expander("⚖️ Legal & Disclosures", expanded=False):
        st.markdown(LEGAL_TEXT)
# ============================================================
# PART 15 / 15 — FINAL NAV EXTENSION (NO DUPLICATION)
# ============================================================

def sidebar():
    with st.sidebar:
        st.markdown("## 💎 Katta Wealth Insights")

        if logged_in():
            st.caption(f"👤 {st.session_state.user}")
            st.caption(f"Plan: {st.session_state.tier}")
        else:
            st.caption("Not logged in")

        st.divider()

        pages = [
            "Portfolio",
            "ETF Look-Through",
            "Income",
            "Goals",
            "Risk",
            "Performance",
            "Tax",
            "AI Chatbot",
            "Exports",
        ]

        page = st.radio("Navigate", pages)

        if logged_in() and not is_pro():
            st.info("Upgrade to unlock AI features")
            if st.button("Upgrade to Pro"):
                db_set_tier(st.session_state.user, "Pro")
                st.session_state.tier = "Pro"
                st.rerun()

        if logged_in() and st.button("Log out"):
            st.session_state.user = None
            st.session_state.tier = "Free"
            st.rerun()

        return page


def run_app():
    if not logged_in():
        auth_ui()
        render_legal_expander()
        render_legal_banner()
        return

    page = sidebar()

    if page == "Portfolio":
        render_portfolio()
    elif page == "ETF Look-Through":
        render_lookthrough()
    elif page == "Income":
        render_income()
    elif page == "Goals":
        render_goals()
    elif page == "Risk":
        render_risk()
    elif page == "Performance":
        render_performance()
    elif page == "Tax":
        render_tax()
    elif page == "AI Chatbot":
        render_chatbot()
    elif page == "Exports":
        render_exports()

    persist_portfolio()
    render_legal_expander()
    render_legal_banner()


# ============================================================
# EXECUTION GUARD — FINAL
# ============================================================

if __name__ == "__main__":
    run_app()
