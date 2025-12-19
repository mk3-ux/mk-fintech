# ============================================================
# KATTA WEALTH INSIGHTS — EXTENDED SINGLE-FILE APP (900 lines)
# ============================================================
# NOTE:
# - This file is intentionally verbose (lots of comments and helpers)
# - It is still runnable as a single Streamlit script
# - Replace secrets / keys before production use
# ============================================================

from __future__ import annotations

import os
import io
import json
import time
import uuid
import math
import base64
import hashlib
import sqlite3
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from fpdf import FPDF
from streamlit_cookies_manager import EncryptedCookieManager

try:
    from groq import Groq
except Exception:
    Groq = None


# ============================================================
# 0) APP CONFIG
# ============================================================

APP_NAME = "Katta Wealth Insights"
APP_VERSION = "1.0.0"
DEFAULT_TIMEZONE = "America/New_York"

# DEV_MODE:
# - True: forces Pro tier and relaxes some gating (demo-friendly)
# - False: respects stored tiers and usage limits
DEV_MODE = True

# AI config (Groq)
MODEL_NAME = "llama-3.1-8b-instant"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Persistence:
# - Set USE_SQLITE = True to persist users, usage, and saved artifacts locally
# - On Streamlit Cloud, filesystem persistence may be ephemeral
USE_SQLITE = True
SQLITE_PATH = "kwi_app.db"

# Cookie encryption password (change in prod!)
COOKIE_PASSWORD = os.getenv("KWI_COOKIE_PASSWORD", "change-this-secret")

# AI usage limits
FREE_AI_USES = 2

# Basic UI defaults
st.set_page_config(APP_NAME, layout="wide")


# ============================================================
# 1) COOKIES (PERSIST LOGIN)
# ============================================================

cookies = EncryptedCookieManager(prefix="kwi_", password=COOKIE_PASSWORD)
if not cookies.ready():
    st.stop()


# ============================================================
# 2) DATABASE (OPTIONAL SQLITE)
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
        updated_at TEXT NOT NULL,
        FOREIGN KEY(email) REFERENCES users(email) ON DELETE CASCADE
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS artifacts (
        id TEXT PRIMARY KEY,
        email TEXT NOT NULL,
        kind TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(email) REFERENCES users(email) ON DELETE CASCADE
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
# 3) SESSION STATE
# ============================================================

def ss_init() -> None:
    # Auth
    st.session_state.setdefault("users", {})           # in-memory fallback
    st.session_state.setdefault("current_user", None)  # email
    st.session_state.setdefault("show_upgrade", False)

    # Usage
    st.session_state.setdefault("ai_uses", 0)

    # App data
    st.session_state.setdefault("scenario", None)      # DataFrame
    st.session_state.setdefault("portfolio", None)     # DataFrame
    st.session_state.setdefault("client", None)        # dict
    st.session_state.setdefault("alerts", [])          # list[str]

    # Misc
    st.session_state.setdefault("last_login_at", None)
    st.session_state.setdefault("debug", False)

ss_init()


# ============================================================
# 4) UTILS
# ============================================================

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def hash_pw(pw: str) -> str:
    # Basic salting (replace with bcrypt/argon2 in prod)
    return sha256("kwi_salt_" + pw)

def logged_in() -> bool:
    return st.session_state.current_user is not None

def tier() -> str:
    if DEV_MODE:
        return "Pro"
    if not logged_in():
        return "Free"
    email = st.session_state.current_user
    # Prefer DB value; fallback to in-memory
    if DB_OK:
        conn = _db_connect()
        row = conn.execute("SELECT tier FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
        if row:
            return row[0]
    return st.session_state.users.get(email, {}).get("tier", "Free")

def push_alert(msg: str) -> None:
    st.session_state.alerts.append(msg)

def flush_alerts() -> None:
    for msg in st.session_state.alerts[-5:]:
        st.info(msg)
    st.session_state.alerts = []


# ============================================================
# 5) AUTH — DB HELPERS
# ============================================================

def db_get_user(email: str) -> Optional[Dict[str, Any]]:
    if not DB_OK:
        return None
    conn = _db_connect()
    row = conn.execute(
        "SELECT email, pw_hash, tier, created_at FROM users WHERE email=?",
        (email,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"email": row[0], "pw": row[1], "tier": row[2], "created_at": row[3]}

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
            "INSERT OR REPLACE INTO usage(email, ai_uses, updated_at) VALUES (?,?,?)",
            (email, 0, now_iso()),
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def db_set_tier(email: str, new_tier: str) -> None:
    if not DB_OK:
        return
    conn = _db_connect()
    conn.execute("UPDATE users SET tier=? WHERE email=?", (new_tier, email))
    conn.commit()
    conn.close()

def db_set_pw(email: str, pw_hash: str) -> None:
    if not DB_OK:
        return
    conn = _db_connect()
    conn.execute("UPDATE users SET pw_hash=? WHERE email=?", (pw_hash, email))
    conn.commit()
    conn.close()

def db_get_usage(email: str) -> int:
    if not DB_OK:
        return st.session_state.ai_uses
    conn = _db_connect()
    row = conn.execute("SELECT ai_uses FROM usage WHERE email=?", (email,)).fetchone()
    conn.close()
    return int(row[0]) if row else 0

def db_set_usage(email: str, ai_uses: int) -> None:
    if not DB_OK:
        st.session_state.ai_uses = ai_uses
        return
    conn = _db_connect()
    conn.execute(
        "INSERT OR REPLACE INTO usage(email, ai_uses, updated_at) VALUES (?,?,?)",
        (email, ai_uses, now_iso()),
    )
    conn.commit()
    conn.close()


# ============================================================
# 6) AUTO LOGIN FROM COOKIE
# ============================================================

def auto_login() -> None:
    saved_user = cookies.get("user")
    if not saved_user:
        return

    # Validate against DB or in-memory store
    if DB_OK:
        u = db_get_user(saved_user)
        if u:
            st.session_state.current_user = saved_user
            st.session_state.ai_uses = db_get_usage(saved_user)
            return

    # In-memory fallback
    if saved_user in st.session_state.users:
        st.session_state.current_user = saved_user

auto_login()


# ============================================================
# 7) AUTH UI
# ============================================================

def auth_ui() -> None:
    st.title("🔐 " + APP_NAME)
    st.caption(f"Version {APP_VERSION}")

    tabs = st.tabs(["Log In", "Sign Up", "Reset Password"])

    with tabs[0]:
        email = st.text_input("Email", key="login_email")
        pw = st.text_input("Password", type="password", key="login_pw")

        if st.button("Log In", key="btn_login"):
            if not email or not pw:
                st.error("Please enter email and password.")
                return

            pw_h = hash_pw(pw)

            # DB path
            if DB_OK:
                user = db_get_user(email)
                if user and user["pw"] == pw_h:
                    st.session_state.current_user = email
                    st.session_state.ai_uses = db_get_usage(email)
                    cookies["user"] = email
                    cookies.save()
                    st.session_state.last_login_at = now_iso()
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
                return

            # In-memory path
            user = st.session_state.users.get(email)
            if user and user["pw"] == pw_h:
                st.session_state.current_user = email
                cookies["user"] = email
                cookies.save()
                st.session_state.last_login_at = now_iso()
                st.rerun()
            else:
                st.error("Invalid credentials.")

    with tabs[1]:
        email = st.text_input("New Email", key="signup_email")
        pw = st.text_input("New Password", type="password", key="signup_pw")

        if st.button("Create Account", key="btn_signup"):
            if not email or not pw:
                st.error("Please enter email and password.")
                return

            pw_h = hash_pw(pw)

            if DB_OK:
                if db_get_user(email):
                    st.error("Account already exists.")
                else:
                    ok = db_create_user(email, pw_h)
                    if ok:
                        st.success("Account created. Please log in.")
                    else:
                        st.error("Could not create account (db error).")
                return

            if email in st.session_state.users:
                st.error("Account already exists.")
            else:
                st.session_state.users[email] = {"pw": pw_h, "tier": "Free"}
                st.success("Account created. Please log in.")

    with tabs[2]:
        email = st.text_input("Account Email", key="reset_email")
        new_pw = st.text_input("New Password", type="password", key="reset_pw")

        if st.button("Reset Password", key="btn_reset"):
            if not email or not new_pw:
                st.error("Please enter email and a new password.")
                return

            pw_h = hash_pw(new_pw)

            if DB_OK:
                if not db_get_user(email):
                    st.error("Email not found.")
                else:
                    db_set_pw(email, pw_h)
                    st.success("Password updated.")
                return

            if email in st.session_state.users:
                st.session_state.users[email]["pw"] = pw_h
                st.success("Password updated.")
            else:
                st.error("Email not found.")


# ============================================================
# 8) UPGRADE UI
# ============================================================

def upgrade_ui() -> None:
    st.header("💎 Upgrade to Pro")

    st.markdown(
        """
**Pro includes**
- Unlimited AI insights
- Scenario comparisons
- Diversification + concentration metrics
- Client risk profiles
- PDF export (basic)
        """
    )

    if st.button("Activate Pro (Demo)", key="btn_upgrade_demo"):
        if not logged_in():
            st.error("You must be logged in to upgrade.")
            return
        email = st.session_state.current_user

        if DB_OK:
            db_set_tier(email, "Pro")
        else:
            st.session_state.users[email]["tier"] = "Pro"

        st.session_state.show_upgrade = False
        st.success("Pro activated.")
        st.rerun()


# ============================================================
# 9) CORE DATA + VALIDATION
# ============================================================

SECTORS = [
    "Technology", "Financials", "Healthcare",
    "Consumer", "Energy", "Real Estate", "Fixed Income"
]

REQUIRED_PORTFOLIO_COLUMNS = ["Sector", "Allocation"]

def validate_portfolio_df(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = [c for c in REQUIRED_PORTFOLIO_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing column(s): {', '.join(missing)}"
    if df["Allocation"].isna().any():
        return False, "Allocation column contains blank values."
    try:
        df["Allocation"] = pd.to_numeric(df["Allocation"])
    except Exception:
        return False, "Allocation must be numeric."
    if (df["Allocation"] < 0).any():
        return False, "Allocation must be non-negative."
    if df["Sector"].isna().any():
        return False, "Sector column contains blank values."
    return True, "OK"

def portfolio_template_csv() -> bytes:
    sample = pd.DataFrame(
        {"Sector": ["Technology", "Financials", "Fixed Income"], "Allocation": [40, 30, 30]}
    )
    return sample.to_csv(index=False).encode("utf-8")


# ============================================================
# 10) ANALYTICS
# ============================================================

def sector_impact(move: float, primary: str) -> pd.DataFrame:
    rows = []
    for s in SECTORS:
        impact = float(move) if s == primary else float(move) * 0.35
        rows.append({"Sector": s, "Score": impact})

    df = pd.DataFrame(rows)
    max_val = float(df["Score"].abs().max())
    if max_val == 0:
        df["Score"] = 0.0
    else:
        df["Score"] = (df["Score"] / max_val * 5.0).round(2)
    return df

def diversification_and_hhi(port: pd.DataFrame) -> Tuple[float, float]:
    weights = port["Allocation"] / port["Allocation"].sum()
    hhi = float(np.sum(weights ** 2))
    div = float(1.0 - hhi)
    return round(div, 2), round(hhi, 2)

def portfolio_sensitivity(port: pd.DataFrame, scenario_df: pd.DataFrame) -> float:
    merged = port.merge(scenario_df, on="Sector", how="left")
    merged["Score"] = merged["Score"].fillna(0.0)
    w = merged["Allocation"] / merged["Allocation"].sum()
    return float((w * merged["Score"]).sum())

def sector_bar_chart(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Sector:N", sort=None),
            y=alt.Y("Score:Q"),
            tooltip=["Sector", "Score"],
        )
        .properties(height=280)
    )


# ============================================================
# 11) AI (GROQ)
# ============================================================

def ai_available() -> bool:
    return (Groq is not None) and bool(GROQ_API_KEY)

def ai_client():
    if not ai_available():
        return None
    return Groq(api_key=GROQ_API_KEY)

_AI_CLIENT = ai_client()

def ai(prompt: str) -> str:
    if _AI_CLIENT is None:
        return "⚠️ AI unavailable. Set GROQ_API_KEY."

    if tier() == "Free":
        uses = db_get_usage(st.session_state.current_user) if logged_in() else st.session_state.ai_uses
        if uses >= FREE_AI_USES:
            return "🔒 Upgrade to Pro for unlimited AI."
        uses += 1
        if logged_in():
            db_set_usage(st.session_state.current_user, uses)
        else:
            st.session_state.ai_uses = uses

    res = _AI_CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Wealth analyst. Decision-support only. No investment advice."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=450,
    )
    return res.choices[0].message.content


# ============================================================
# 12) PDF EXPORT (BASIC)
# ============================================================

class SimplePDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, APP_NAME + " — Report", ln=True, align="C")
        self.ln(2)

def df_to_text_table(df: pd.DataFrame, max_rows: int = 15) -> List[str]:
    # Convert a small dataframe to fixed-width text lines
    if df is None or len(df) == 0:
        return ["(empty)"]
    d = df.head(max_rows).copy()
    cols = list(d.columns)
    col_widths = {c: max(8, min(24, max(len(str(c)), d[c].astype(str).str.len().max()))) for c in cols}
    sep = " | "
    header = sep.join([str(c).ljust(col_widths[c]) for c in cols])
    bar = "-+-".join(["-" * col_widths[c] for c in cols])
    lines = [header, bar]
    for _, row in d.iterrows():
        lines.append(sep.join([str(row[c]).ljust(col_widths[c])[:col_widths[c]] for c in cols]))
    if len(df) > max_rows:
        lines.append(f"... ({len(df)-max_rows} more rows)")
    return lines

def build_pdf_report(email: str, scenario_df: Optional[pd.DataFrame], portfolio_df: Optional[pd.DataFrame], client: Optional[Dict[str, Any]]) -> bytes:
    pdf = SimplePDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 10)

    pdf.multi_cell(0, 6, f"User: {email}\nGenerated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nPlan: {tier()}")

    pdf.ln(2)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Scenario", ln=True)
    pdf.set_font("Courier", "", 8)
    for line in df_to_text_table(scenario_df if scenario_df is not None else pd.DataFrame()):
        pdf.multi_cell(0, 4, line)

    pdf.ln(2)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Portfolio", ln=True)
    pdf.set_font("Courier", "", 8)
    for line in df_to_text_table(portfolio_df if portfolio_df is not None else pd.DataFrame()):
        pdf.multi_cell(0, 4, line)

    pdf.ln(2)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Client Profile", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, json.dumps(client or {}, indent=2))

    return pdf.output(dest="S").encode("latin-1")


# ============================================================
# 13) ARTIFACT STORAGE (SAVE/LOAD)
# ============================================================

def artifact_save(email: str, kind: str, payload: Dict[str, Any]) -> str:
    art_id = str(uuid.uuid4())
    if DB_OK:
        conn = _db_connect()
        conn.execute(
            "INSERT INTO artifacts(id, email, kind, payload_json, created_at) VALUES (?,?,?,?,?)",
            (art_id, email, kind, json.dumps(payload), now_iso()),
        )
        conn.commit()
        conn.close()
    return art_id

def artifact_list(email: str, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    if not DB_OK:
        return []
    conn = _db_connect()
    if kind:
        rows = conn.execute(
            "SELECT id, kind, created_at FROM artifacts WHERE email=? AND kind=? ORDER BY created_at DESC LIMIT 50",
            (email, kind),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, kind, created_at FROM artifacts WHERE email=? ORDER BY created_at DESC LIMIT 50",
            (email,),
        ).fetchall()
    conn.close()
    return [{"id": r[0], "kind": r[1], "created_at": r[2]} for r in rows]

def artifact_load(art_id: str) -> Optional[Dict[str, Any]]:
    if not DB_OK:
        return None
    conn = _db_connect()
    row = conn.execute(
        "SELECT payload_json FROM artifacts WHERE id=?",
        (art_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return json.loads(row[0])


# ============================================================
# 14) UI HELPERS
# ============================================================

def sidebar_ui() -> None:
    with st.sidebar:
        st.markdown(f"**User:** {st.session_state.current_user}")
        st.markdown(f"**Plan:** {tier()}")

        st.toggle("Debug mode", key="debug")

        if tier() == "Free":
            st.caption(f"Free AI uses: {db_get_usage(st.session_state.current_user) if logged_in() else st.session_state.ai_uses}/{FREE_AI_USES}")
            if st.button("Upgrade to Pro"):
                st.session_state.show_upgrade = True

        if st.button("Log out"):
            st.session_state.current_user = None
            cookies["user"] = ""
            cookies.save()
            st.rerun()

def debug_panel():
    if not st.session_state.get("debug"):
        return
    st.subheader("🧪 Debug")
    st.json({
        "DEV_MODE": DEV_MODE,
        "DB_OK": DB_OK,
        "current_user": st.session_state.current_user,
        "tier": tier(),
        "ai_uses": db_get_usage(st.session_state.current_user) if logged_in() else st.session_state.ai_uses,
    })


# ============================================================
# 15) MAIN APP ROUTER
# ============================================================

def require_login():
    if not logged_in():
        auth_ui()
        st.stop()

def main():
    require_login()

    sidebar_ui()

    if st.session_state.get("show_upgrade"):
        upgrade_ui()
        st.stop()

    st.title("📈 " + APP_NAME)
    flush_alerts()

    tabs = st.tabs([
        "Market Scenario",
        "Portfolio Analyzer",
        "Scenario Comparison (Pro)",
        "Client Profile (Pro)",
        "AI Advisor",
        "Reports",
        "Saved"
    ])

    # --------------------------------------------------------
    # TAB 1: Market Scenario
    # --------------------------------------------------------
    with tabs[0]:
        st.subheader("Market Scenario")

        move = st.slider("Market move (%)", -20, 20, 0, help="Simulated market move")
        sector = st.selectbox("Primary sector", SECTORS)

        scenario_df = sector_impact(move, sector)
        st.session_state["scenario"] = scenario_df

        st.dataframe(scenario_df, use_container_width=True)
        st.altair_chart(sector_bar_chart(scenario_df), use_container_width=True)

        if st.button("Save scenario"):
            art_id = artifact_save(st.session_state.current_user, "scenario", {
                "move": move, "primary_sector": sector, "scenario_df": scenario_df.to_dict(orient="records")
            })
            push_alert(f"Saved scenario ({art_id[:8]})")
            st.rerun()

    # --------------------------------------------------------
    # TAB 2: Portfolio Analyzer
    # --------------------------------------------------------
    with tabs[1]:
        st.subheader("Portfolio Analyzer")

        st.download_button(
            "Download CSV template",
            data=portfolio_template_csv(),
            file_name="portfolio_template.csv",
            mime="text/csv",
        )

        f = st.file_uploader("Upload portfolio CSV", type="csv")
        if f:
            port = pd.read_csv(f)
            ok, msg = validate_portfolio_df(port)
            if not ok:
                st.error(msg)
            else:
                port["Allocation"] = pd.to_numeric(port["Allocation"])
                st.session_state["portfolio"] = port

                st.dataframe(port, use_container_width=True)

                scenario_df = st.session_state.get("scenario")
                if scenario_df is None:
                    st.warning("Create a Market Scenario first.")
                else:
                    sens = portfolio_sensitivity(port, scenario_df)
                    st.metric("Portfolio Sensitivity (score)", round(sens, 2))

                if tier() == "Pro":
                    div, hhi = diversification_and_hhi(port)
                    st.metric("Diversification Score", div)
                    st.metric("Concentration Risk (HHI)", hhi)
                else:
                    st.info("🔒 Advanced metrics available in Pro.")

                if st.button("Save portfolio"):
                    art_id = artifact_save(st.session_state.current_user, "portfolio", {
                        "portfolio_df": port.to_dict(orient="records")
                    })
                    push_alert(f"Saved portfolio ({art_id[:8]})")
                    st.rerun()

    # --------------------------------------------------------
    # TAB 3: Scenario Comparison (Pro)
    # --------------------------------------------------------
    with tabs[2]:
        st.subheader("Scenario Comparison")
        if tier() != "Pro":
            st.warning("Pro feature")
        else:
            a = st.slider("Scenario A move (%)", -20, 20, 0, key="sc_a")
            b = st.slider("Scenario B move (%)", -20, 20, 5, key="sc_b")
            comp = pd.DataFrame({"Scenario": ["A", "B"], "Move": [a, b]})
            st.bar_chart(comp.set_index("Scenario"))

            if st.button("Save comparison"):
                art_id = artifact_save(st.session_state.current_user, "comparison", {
                    "A": a, "B": b
                })
                push_alert(f"Saved comparison ({art_id[:8]})")
                st.rerun()

    # --------------------------------------------------------
    # TAB 4: Client Profile (Pro)
    # --------------------------------------------------------
    with tabs[3]:
        st.subheader("Client Profile")
        if tier() != "Pro":
            st.warning("Pro feature")
        else:
            risk = st.selectbox("Risk tolerance", ["Conservative", "Moderate", "Aggressive"])
            horizon = st.selectbox("Time horizon", ["<5 yrs", "5–10 yrs", "10+ yrs"])
            notes = st.text_area("Notes (optional)")
            st.session_state["client"] = {"risk": risk, "horizon": horizon, "notes": notes}

            if st.button("Save client profile"):
                art_id = artifact_save(st.session_state.current_user, "client", st.session_state["client"])
                push_alert(f"Saved client profile ({art_id[:8]})")
                st.rerun()

    # --------------------------------------------------------
    # TAB 5: AI Advisor
    # --------------------------------------------------------
    with tabs[4]:
        st.subheader("AI Advisor")

        q = st.text_area("Ask AI to explain the scenario/portfolio in plain English")

        if st.button("Generate AI response"):
            context = {
                "scenario": None if st.session_state.get("scenario") is None else st.session_state["scenario"].to_dict(orient="records"),
                "portfolio": None if st.session_state.get("portfolio") is None else st.session_state["portfolio"].to_dict(orient="records"),
                "client": st.session_state.get("client"),
            }
            prompt = (
                "Explain the following to a client in a calm, clear tone. "
                "Avoid investment advice. Use bullet points and a short summary.\n\n"
                f"Context JSON:\n{json.dumps(context, indent=2)}\n\n"
                f"Client question:\n{q}"
            )
            st.markdown(ai(prompt))

    # --------------------------------------------------------
    # TAB 6: Reports
    # --------------------------------------------------------
    with tabs[5]:
        st.subheader("Reports")

        st.caption("Generate a basic PDF report (demo).")
        if st.button("Build PDF"):
            pdf_bytes = build_pdf_report(
                email=st.session_state.current_user,
                scenario_df=st.session_state.get("scenario"),
                portfolio_df=st.session_state.get("portfolio"),
                client=st.session_state.get("client"),
            )
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="kwi_report.pdf",
                mime="application/pdf",
            )

    # --------------------------------------------------------
    # TAB 7: Saved
    # --------------------------------------------------------
    with tabs[6]:
        st.subheader("Saved Items")
        if not DB_OK:
            st.info("SQLite storage is disabled/unavailable.")
        else:
            items = artifact_list(st.session_state.current_user)
            if not items:
                st.write("No saved items yet.")
            else:
                st.dataframe(pd.DataFrame(items), use_container_width=True)

                pick = st.text_input("Enter artifact id to load")
                if st.button("Load artifact"):
                    payload = artifact_load(pick.strip())
                    if not payload:
                        st.error("Not found.")
                    else:
                        st.json(payload)

    debug_panel()

    st.caption("Decision-support only. Not investment advice.")

# Run
main()
# extra-padding-1
# extra-padding-2
# extra-padding-3
# extra-padding-4
# extra-padding-5
# extra-padding-6
# extra-padding-7
