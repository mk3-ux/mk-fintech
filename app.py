# ============================================================
# KATTA WEALTH INSIGHTS — FULL APP WITH EMAIL OTP AUTH
# ============================================================

from __future__ import annotations
import os, time, json, uuid, hashlib, sqlite3, random
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from fpdf import FPDF

# Optional imports
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from streamlit_cookies_manager import EncryptedCookieManager
except Exception:
    EncryptedCookieManager = None

# ============================================================
# CONFIG
# ============================================================
APP_NAME = "Katta Wealth Insights"
PRO_PRICE = 24
FREE_AI_LIMIT = 2
OTP_TTL_SECONDS = 300
DEV_MODE = False

MODEL_NAME = "llama-3.1-8b-instant"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

USE_SQLITE = True
SQLITE_PATH = "kwi.db"

USE_COOKIES = True
COOKIE_PASSWORD = os.getenv("KWI_COOKIE_PASSWORD", "change-this-secret")

st.set_page_config(APP_NAME, layout="wide")

# ============================================================
# SESSION STATE
# ============================================================
def ss(key, default):
    st.session_state.setdefault(key, default)

ss("users", {})
ss("current_user", None)
ss("ai_uses", 0)
ss("show_upgrade", False)
ss("scenario", None)
ss("portfolio", None)
ss("client", None)
ss("alerts", [])

# OTP
ss("otp_email", None)
ss("otp_hash", None)
ss("otp_expiry", None)

# ============================================================
# UTILITIES
# ============================================================
def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def hash_pw(pw: str) -> str:
    return sha256("pw_salt_" + pw)

def logged_in() -> bool:
    return st.session_state.current_user is not None

def tier() -> str:
    if DEV_MODE:
        return "Pro"
    if not logged_in():
        return "Free"
    if USE_SQLITE:
        u = db_get_user(st.session_state.current_user)
        if u:
            return u["tier"]
    return st.session_state.users.get(st.session_state.current_user, {}).get("tier", "Free")

def is_pro() -> bool:
    return tier() == "Pro"

def alert(msg: str):
    st.session_state.alerts.append(msg)

def flush_alerts():
    for a in st.session_state.alerts:
        st.info(a)
    st.session_state.alerts = []

# ============================================================
# SQLITE
# ============================================================
def db():
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def db_init():
    c = db()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        pw_hash TEXT,
        tier TEXT,
        created_at TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS usage (
        email TEXT PRIMARY KEY,
        ai_uses INTEGER
    )
    """)
    c.commit()
    c.close()

def db_get_user(email):
    c = db()
    r = c.execute("SELECT email,pw_hash,tier FROM users WHERE email=?", (email,)).fetchone()
    c.close()
    return {"email": r[0], "pw": r[1], "tier": r[2]} if r else None

def db_create_user(email, pw):
    c = db()
    c.execute("INSERT INTO users VALUES (?,?,?,?)",
              (email, pw, "Free", dt.datetime.utcnow().isoformat()))
    c.execute("INSERT INTO usage VALUES (?,?)", (email, 0))
    c.commit()
    c.close()

def db_set_pw(email, pw):
    c = db()
    c.execute("UPDATE users SET pw_hash=? WHERE email=?", (pw, email))
    c.commit()
    c.close()

def db_set_tier(email, tier):
    c = db()
    c.execute("UPDATE users SET tier=? WHERE email=?", (tier, email))
    c.commit()
    c.close()

def db_get_usage(email):
    c = db()
    r = c.execute("SELECT ai_uses FROM usage WHERE email=?", (email,)).fetchone()
    c.close()
    return r[0] if r else 0

def db_set_usage(email, n):
    c = db()
    c.execute("UPDATE usage SET ai_uses=? WHERE email=?", (n, email))
    c.commit()
    c.close()

db_init()

# ============================================================
# OTP
# ============================================================
def generate_otp():
    return f"{random.randint(100000, 999999)}"

def start_otp(email):
    otp = generate_otp()
    st.session_state.otp_email = email
    st.session_state.otp_hash = sha256("otp_" + otp)
    st.session_state.otp_expiry = time.time() + OTP_TTL_SECONDS
    st.info(f"📧 DEMO OTP for {email}: **{otp}**")

def verify_otp(otp):
    if time.time() > st.session_state.otp_expiry:
        return False, "OTP expired"
    if sha256("otp_" + otp) != st.session_state.otp_hash:
        return False, "Invalid OTP"
    return True, None

# ============================================================
# AUTH UI
# ============================================================
def auth_ui():
    st.title("🔐 " + APP_NAME)

    tabs = st.tabs(["Log In", "Sign Up"])

    # ---------------- LOGIN ----------------
    with tabs[0]:
        st.subheader("Log In")

        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")

        if st.button("Log In"):
            pw_h = hash_pw(pw)
            u = db_get_user(email)
            if u and u["pw"] == pw_h:
                st.session_state.current_user = email
                st.session_state.ai_uses = db_get_usage(email)
                st.success("Logged in")
                st.rerun()
            else:
                st.error("Invalid email or password")

        st.markdown("### 🔐 Log in with Email OTP")
        otp_email = st.text_input("Email for OTP login")
        if st.button("Send OTP"):
            if not db_get_user(otp_email):
                st.error("No account found")
            else:
                start_otp(otp_email)

        if st.session_state.otp_email:
            otp = st.text_input("Enter 6-digit OTP")
            if st.button("Verify OTP"):
                ok, err = verify_otp(otp)
                if not ok:
                    st.error(err)
                else:
                    st.session_state.current_user = st.session_state.otp_email
                    st.success("Logged in via OTP")
                    st.rerun()

        st.markdown("### Need help?")
        if st.button("Forgot email (username)?"):
            st.info("Your username is the email you signed up with.")

        if st.button("Forgot password?"):
            r_email = st.text_input("Account email", key="reset_email")
            new_pw = st.text_input("New password", type="password", key="reset_pw")
            if st.button("Reset password"):
                if not db_get_user(r_email):
                    st.error("Email not found")
                else:
                    db_set_pw(r_email, hash_pw(new_pw))
                    st.success("Password reset")

    # ---------------- SIGN UP ----------------
    with tabs[1]:
        st.subheader("Create Account")
        email = st.text_input("Email", key="signup_email")
        pw = st.text_input("Password", type="password", key="signup_pw")

        if st.button("Create Account"):
            if db_get_user(email):
                st.error("Account exists")
            else:
                db_create_user(email, hash_pw(pw))
                st.success("Account created")

# ============================================================
# APP START
# ============================================================
if not logged_in():
    auth_ui()
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"**User:** {st.session_state.current_user}")
    st.markdown(f"**Plan:** {tier()}")

    if not is_pro():
        st.caption(f"AI preview uses: {db_get_usage(st.session_state.current_user)}/{FREE_AI_LIMIT}")
        if st.button("Upgrade to Pro"):
            st.session_state.show_upgrade = True

    if st.button("Log out"):
        st.session_state.current_user = None
        st.rerun()

# ============================================================
# UPGRADE
# ============================================================
if st.session_state.show_upgrade:
    st.header("💎 Upgrade to Pro")
    st.markdown(f"""
**Pro – ${PRO_PRICE}/month**

✔ Unlimited AI  
✔ Scenario comparison  
✔ Client profiles  
✔ PDF reports
""")
    if st.button("Activate Pro (Demo)"):
        db_set_tier(st.session_state.current_user, "Pro")
        st.success("Pro activated")
        st.rerun()
    st.stop()

# ============================================================
# MAIN APP (SHORTENED FOR CLARITY)
# ============================================================
st.title("📈 " + APP_NAME)
flush_alerts()
st.caption("Decision-support only. Not investment advice.")

st.success("🎉 App loaded successfully with Email OTP authentication")
