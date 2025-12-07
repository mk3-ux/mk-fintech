# app.py

import os
import requests
import streamlit as st
import pandas as pd
import altair as alt
from groq import Groq
from fpdf import FPDF

# ---------------------------
# HARD-CODED API KEYS (your keys)
# ---------------------------
# WARNING: Do NOT commit this file to GitHub with real keys.
NEWSAPI_KEY = "4f0f0589094c414a8ef178ee05c9226d"
MASSIVE_API_KEY = "Q1pmrfqv0vV6caqxpJUjwcyEsSEvvSJU"

# Groq key is optional – use env var so you don’t leak it
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ---------------------------
# Streamlit config
# ---------------------------
st.set_page_config(
    page_title="Katta Finsight",
    layout="wide",
    page_icon="📊",
)

COLORS = {
    "bg": "#F7FAFF",
    "text": "#0F172A",
    "card": "#FFFFFF",
    "accent": "#0EA5E9",
    "subtle": "#E6EEF8",
}

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}
        .block-container {{ max-width: 1200px; padding-top: 1.2rem; padding-bottom: 2rem; }}
        button[data-baseweb="tab"] {{ border-radius: 999px !important; padding: 0.3rem 1rem !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Domain model
# ---------------------------
SECTORS = [
    "Tech",
    "Real Estate",
    "Luxury / Discretionary",
    "Bonds",
    "Energy",
    "Consumer Staples",
    "Banks",
]

# Simple “stock → sector” sensitivity
def compute_stock_sector_impacts(stock_move_pct: float, primary_sector: str) -> pd.DataFrame:
    """
    stock_move_pct: % move of stock (e.g., +3, -5)
    primary_sector: main sector of this stock
    Primary sector gets sensitivity 1.0; all others 0.4
    """
    rows = []
    for sec in SECTORS:
        sensitivity = 1.0 if sec == primary_sector else 0.4
        raw_score = sensitivity * stock_move_pct
        rows.append(
            {
                "Sector": sec,
                "Sensitivity": sensitivity,
                "Raw Score": raw_score,
            }
        )

    df = pd.DataFrame(rows)
    max_abs = df["Raw Score"].abs().max()
    if max_abs and max_abs > 0:
        df["Impact Score"] = df["Raw Score"] / max_abs * 5
    else:
        df["Impact Score"] = 0.0

    df["Impact Score"] = df["Impact Score"].round(2)

    def label(score):
        if score <= -3.5:
            return "Strong Negative"
        if score <= -1.5:
            return "Mild Negative"
        if score < 1.5:
            return "Neutral"
        if score < 3.5:
            return "Mild Positive"
        return "Strong Positive"

    df["Impact Label"] = df["Impact Score"].apply(label)
    return df[["Sector", "Impact Score", "Impact Label"]]

# ---------------------------
# Massive helpers (live stock data)
# ---------------------------

MASSIVE_BASE = "https://api.massive.com"

def fetch_live_snapshot(ticker: str):
    """
    Use Massive single-ticker snapshot:
    GET /v2/snapshot/locale/us/markets/stocks/tickers/{stocksTicker}?apiKey=...
    Returns dict with price, change, change_pct and raw snapshot.
    """
    if not ticker:
        return None

    url = f"{MASSIVE_BASE}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}"
    params = {"apiKey": MASSIVE_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        t = data.get("ticker")
        if not t:
            return None

        last_trade = t.get("lastTrade") or {}
        day_bar = t.get("day") or {}

        price = last_trade.get("p") or day_bar.get("c")
        change = t.get("todaysChange")
        change_pct = t.get("todaysChangePerc")

        return {
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "raw": t,
        }
    except Exception as e:
        print("Massive error:", e)
        return None


def fetch_bulk_snapshots(tickers):
    """
    Simple loop over tickers for hedge fund dashboard.
    """
    rows = []
    for tk in tickers:
        tk = tk.strip().upper()
        if not tk:
            continue
        snap = fetch_live_snapshot(tk)
        if snap is None:
            rows.append(
                {
                    "Ticker": tk,
                    "Price": None,
                    "Change": None,
                    "Change %": None,
                    "Status": "Error / No data",
                }
            )
        else:
            rows.append(
                {
                    "Ticker": tk,
                    "Price": snap["price"],
                    "Change": snap["change"],
                    "Change %": snap["change_pct"],
                    "Status": "OK",
                }
            )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["Ticker", "Price", "Change", "Change %", "Status"])

# ---------------------------
# NewsAPI helper
# ---------------------------

def fetch_news(keyword="finance", page_size=5):
    """
    Use NewsAPI 'everything' endpoint for a keyword.
    """
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        if resp.status_code == 200 and "articles" in data:
            return data["articles"]
        return []
    except Exception as e:
        print("NewsAPI error:", e)
        return []

# ---------------------------
# Portfolio + AI + PDF helpers
# ---------------------------

def compute_portfolio_exposure(portfolio_df: pd.DataFrame, sector_scores: pd.DataFrame):
    df = portfolio_df.copy()
    if "Allocation" not in df.columns:
        raise ValueError("Uploaded portfolio must contain 'Allocation' column")

    total = df["Allocation"].sum()
    if total == 0:
        df["Weight"] = 0.0
    else:
        df["Weight"] = df["Allocation"] / total

    merged = df.merge(sector_scores, on="Sector", how="left")
    merged["Impact Score"].fillna(0.0, inplace=True)
    merged["Weighted Impact"] = merged["Weight"] * merged["Impact Score"]
    portfolio_score = merged["Weighted Impact"].sum()
    return portfolio_score, merged[
        ["Sector", "Allocation", "Weight", "Impact Score", "Weighted Impact"]
    ]

def call_ai_research(history, user_text: str, level: str = "Professional") -> str:
    if client is None:
        return "AI Research Analyst not configured. Set GROQ_API_KEY in your environment."

    level_line = {
        "Professional": "Write as a corporate research analyst for internal use, concise and structured.",
        "Executive": "Write a polished executive summary for C-suite consumption, ~3 paragraphs.",
        "Technical": "Write a detailed technical memo with supporting reasoning and caveats.",
    }.get(level, "Write in clear professional language.")

    system_prompt = (
        "You are an internal AI research analyst for an institutional client. "
        "Your task: convert stock scenarios, sector impacts, and live data into concise internal research narratives, "
        "scenario summaries, risk notes and suggested topics for internal follow-up. "
        "You MUST NOT provide direct buy/sell recommendations, price targets or personalized investment advice. "
        + level_line
    )

    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_text})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=700,
            temperature=0.2,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"AI service error: {e}"

def create_pdf_report(
    title: str,
    scenario_name: str,
    scenario_meta: dict,
    sector_df: pd.DataFrame,
    portfolio_table: pd.DataFrame = None,
    ai_summary: str = "",
):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="L")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Scenario: {scenario_name}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Scenario Inputs:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for k, v in scenario_meta.items():
        pdf.cell(0, 7, f"{k}: {v}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Sector Impacts:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for _, r in sector_df.iterrows():
        pdf.cell(
            0,
            7,
            f"{r['Sector']}: {r['Impact Score']} ({r['Impact Label']})",
            ln=True,
        )
    pdf.ln(4)

    if portfolio_table is not None and not portfolio_table.empty:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Portfolio Exposure (sample):", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for _, r in portfolio_table.iterrows():
            pdf.cell(
                0,
                6,
                f"{r['Sector']}: Allocation {r['Allocation']}, "
                f"WeightedImpact {r['Weighted Impact']:.3f}",
                ln=True,
            )
        pdf.ln(4)

    if ai_summary:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "AI Research Summary:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for line in ai_summary.split("\n"):
            pdf.multi_cell(0, 6, line)

    return pdf.output(dest="S").encode("latin-1")

# ---------------------------
# Header
# ---------------------------
def render_header():
    st.markdown(
        f"""
        <div style="background:linear-gradient(90deg,#ECFEFF,#F0F9FF);padding:12px;border-radius:14px;border:1px solid {COLORS['subtle']};margin-bottom:14px;">
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:56px;height:56px;border-radius:12px;background:{COLORS['accent']};color:white;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:20px;">
              KM
            </div>
            <div>
              <div style="font-size:18px;font-weight:800;color:{COLORS['text']};">Katta MacroSuite — Stock Intelligence</div>
              <div style="font-size:12px;color:#475569;">Live Massive data • Sector sensitivity • Hedge fund dashboard • AI research • PDF reports</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_header()

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.title("Katta MacroSuite")

    ai_level = st.radio(
        "AI output style:",
        ["Professional", "Executive", "Technical"],
        index=0,
    )

    st.markdown("---")

    if client is None:
        st.warning(
            "Groq API not configured — AI Research Analyst disabled until GROQ_API_KEY is set."
        )

    st.markdown(
        "Upload a CSV portfolio (columns: Sector, Allocation). "
        "Allocation can be percent or units."
    )

    st.markdown("---")

    st.caption(
        "This platform is a decision-support tool. "
        "It does NOT provide buy/sell advice."
    )

# ---------------------------
# Session state init
# ---------------------------
for key, default in [
    ("sector_df", None),
    ("scenario_name", None),
    ("scenario_meta", {}),
    ("portfolio_df", None),
    ("ai_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------
# Tabs
# ---------------------------
tab_live, tab_portfolio, tab_dashboard, tab_ai, tab_reports = st.tabs(
    [
        "Live Stock Impact",
        "Portfolio Analyzer",
        "Hedge Fund Dashboard",
        "AI Research Analyst",
        "Generate Report",
    ]
)

# ---------- Live Stock Impact ----------
with tab_live:
    st.subheader("Live Stock Impact — Stock → Sector Sensitivity")

    stock_name = st.text_input("Stock ticker", "AAPL").upper()
    primary_sector = st.selectbox(
        "Primary sector for this stock",
        SECTORS,
        index=0,
        help="Which sector best represents this stock?",
    )

    # Live snapshot from Massive
    snapshot = fetch_live_snapshot(stock_name)

    col_live_1, col_live_2, col_live_3 = st.columns(3)
    if snapshot and snapshot["price"] is not None:
        price = snapshot["price"]
        change = snapshot["change"]
        change_pct = snapshot["change_pct"]

        with col_live_1:
            st.metric("Live Price", f"${price:,.2f}")
        with col_live_2:
            st.metric(
                "Today’s Change",
                f"{change:+.2f}" if change is not None else "N/A",
            )
        with col_live_3:
            st.metric(
                "Change (%)",
                f"{change_pct:+.2f}%" if change_pct is not None else "N/A",
            )

        default_move = int(round(change_pct)) if isinstance(change_pct, (int, float)) else 0
    else:
        st.info("Could not load live data from Massive (check key/plan/internet).")
        default_move = 0

    st.markdown("### Scenario: extra move in this stock")
    stock_move = st.slider(
        "Assumed additional stock move (%)",
        -20,
        20,
        int(default_move),
        help="Negative = stock down, Positive = stock up. Uses Massive live data as context.",
    )

    sector_df = compute_stock_sector_impacts(stock_move, primary_sector)

    scenario_name = f"{stock_name} extra move {stock_move:+.1f}%"
    scenario_meta = {
        "Stock": stock_name,
        "Extra Move (%)": stock_move,
        "Primary Sector": primary_sector,
        "Live price (if available)": snapshot["price"] if snapshot else None,
        "Live change % (if available)": snapshot["change_pct"] if snapshot else None,
    }

    st.session_state["sector_df"] = sector_df
    st.session_state["scenario_name"] = scenario_name
    st.session_state["scenario_meta"] = scenario_meta

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("#### Sector Impact from this Stock Scenario")
        st.dataframe(
            sector_df.style.format({"Impact Score": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.markdown("#### Visual Overview")
        chart = (
            alt.Chart(sector_df)
            .mark_bar()
            .encode(
                x=alt.X("Sector:N", sort=None),
                y=alt.Y("Impact Score:Q"),
                tooltip=["Sector", "Impact Score", "Impact Label"],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Quick take")
    sorted_df = sector_df.sort_values("Impact Score", ascending=False)
    winners = sorted_df.head(2)
    losers = sorted_df.tail(2)
    winner_text = ", ".join(
        f"{row.Sector} ({row['Impact Label']})" for _, row in winners.iterrows()
    )
    loser_text = ", ".join(
        f"{row.Sector} ({row['Impact Label']})" for _, row in losers.iterrows()
    )
    st.markdown(
        f"- **Most positively exposed sectors:** {winner_text}  \n"
        f"- **Most negatively exposed / least helped:** {loser_text}"
    )
    st.caption(
        "Simplified teaching model — not a real-world risk engine or investment advice."
    )

    # Embedded news for this stock
    st.markdown("### Recent Headlines for this Ticker")
    articles = fetch_news(stock_name, page_size=5)
    if articles:
        for art in articles:
            title = art.get("title") or "Untitled"
            desc = art.get("description") or ""
            url = art.get("url")
            st.markdown(f"**📰 {title}**")
            if desc:
                st.write(desc)
            if url:
                st.write(f"[Read more]({url})")
            st.write("---")
    else:
        st.write("No news found or NewsAPI issue.")

# ---------- Portfolio Analyzer ----------
with tab_portfolio:
    st.subheader("Portfolio / Revenue Exposure Analyzer")
    st.markdown("Upload a CSV with columns: `Sector`, `Allocation` (percent or units).")

    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])

    if uploaded is not None:
        try:
            portfolio_df = pd.read_csv(uploaded)
            st.session_state["portfolio_df"] = portfolio_df
            st.write("Uploaded portfolio preview:")
            st.dataframe(portfolio_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    if st.button("Analyze exposure to current stock scenario"):
        sector_df_current = st.session_state.get("sector_df")
        portfolio_df_current = st.session_state.get("portfolio_df")

        if sector_df_current is None:
            st.error(
                "No scenario found. Configure a stock scenario on the 'Live Stock Impact' tab first."
            )
        elif portfolio_df_current is None:
            st.error("Please upload a portfolio CSV first.")
        else:
            try:
                score, breakdown = compute_portfolio_exposure(
                    portfolio_df_current, sector_df_current
                )
                st.metric("Portfolio Weighted Impact Score", f"{score:.3f}")
                st.markdown("Breakdown:")
                st.dataframe(breakdown, use_container_width=True)

                if score > 1.5:
                    st.success(
                        "Portfolio tilt: Mild to strong positive sensitivity to the current stock scenario."
                    )
                elif score < -1.5:
                    st.warning(
                        "Portfolio tilt: Mild to strong negative sensitivity to the current stock scenario."
                    )
                else:
                    st.info(
                        "Portfolio tilt: Largely neutral under the chosen stock scenario."
                    )
            except Exception as e:
                st.error(f"Analysis error: {e}")

# ---------- Hedge Fund Dashboard ----------
with tab_dashboard:
    st.subheader("Hedge Fund Dashboard — Live Watchlist (Massive)")

    default_watchlist = "AAPL, MSFT, NVDA, AMZN, GOOGL"
    watchlist_str = st.text_input(
        "Watchlist tickers (comma separated)", default_watchlist
    )
    tickers = [t.strip().upper() for t in watchlist_str.split(",") if t.strip()]

    if st.button("Refresh live data"):
        if not tickers:
            st.error("Please enter at least one ticker.")
        else:
            df_watch = fetch_bulk_snapshots(tickers)
            if df_watch.empty:
                st.error("No data returned from Massive.")
            else:
                st.dataframe(df_watch, use_container_width=True)

                # Simple bar chart of % moves
                df_chart = df_watch.dropna(subset=["Change %"])
                if not df_chart.empty:
                    st.markdown("#### % Change Heat Map")
                    chart = (
                        alt.Chart(df_chart)
                        .mark_bar()
                        .encode(
                            x=alt.X("Ticker:N"),
                            y=alt.Y("Change %:Q"),
                            tooltip=["Ticker", "Price", "Change %"],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No valid Change % values to chart.")

# ---------- AI Research Analyst ----------
with tab_ai:
    st.subheader("AI Research Analyst — Generate internal narratives")
    st.markdown(
        "Draft internal memos, scenario summaries, and risk notes. (AI requires GROQ_API_KEY set in your environment.)"
    )

    user_q = st.text_area(
        "Ask the AI Research Analyst "
        "(e.g., 'Write an executive summary of this stock scenario for a hedge fund PM')",
        height=140,
    )

    if st.button("Run AI"):
        if client is None:
            st.error("AI not configured. Set GROQ_API_KEY as an environment variable.")
        else:
            sector_df_current = st.session_state.get("sector_df")
            scenario_name = st.session_state.get("scenario_name", "Current scenario")
            scenario_meta = st.session_state.get("scenario_meta", {})

            if sector_df_current is None:
                st.error(
                    "No scenario found. Please configure a stock on the 'Live Stock Impact' tab first."
                )
            else:
                context = (
                    f"Scenario: {scenario_name}\n"
                    f"Scenario inputs: {scenario_meta}\n"
                    f"Top sector impacts:\n"
                )
                top_n = sector_df_current.sort_values(
                    "Impact Score", ascending=False
                ).head(3)
                for _, r in top_n.iterrows():
                    context += (
                        f"- {r['Sector']}: {r['Impact Score']} ({r['Impact Label']})\n"
                    )
                prompt = context + "\nUser request:\n" + user_q
                st.session_state["ai_history"].append(
                    {"role": "user", "content": prompt}
                )
                with st.spinner("Generating AI summary..."):
                    out = call_ai_research(
                        st.session_state["ai_history"], prompt, ai_level
                    )
                st.markdown("**AI output**")
                st.markdown(out)
                st.session_state["ai_history"].append(
                    {"role": "assistant", "content": out}
                )
                st.session_state["ai_history"] = st.session_state["ai_history"][-20:]

# ---------- Report Generation ----------
with tab_reports:
    st.subheader("Generate Downloadable Report")
    report_title = st.text_input("Report title", "Stock Scenario & Portfolio Insight")
    include_portfolio = st.checkbox(
        "Include uploaded portfolio exposure (if available)", value=True
    )
    ai_summary_for_report = st.text_area(
        "Optional: paste AI summary to include in report", height=120
    )

    if st.button("Create PDF Report"):
        sector_df_current = st.session_state.get("sector_df")
        scenario_name = st.session_state.get("scenario_name", "Current scenario")
        scenario_meta = st.session_state.get("scenario_meta", {})
        portfolio_df_current = st.session_state.get("portfolio_df")

        if sector_df_current is None:
            st.error(
                "No scenario found. Configure a stock on the 'Live Stock Impact' tab first."
            )
        else:
            portfolio_table = None
            if include_portfolio and portfolio_df_current is not None:
                try:
                    _, portfolio_table = compute_portfolio_exposure(
                        portfolio_df_current, sector_df_current
                    )
                except Exception as e:
                    st.error(f"Cannot include portfolio: {e}")
                    portfolio_table = None

            pdf_bytes = create_pdf_report(
                report_title,
                scenario_name,
                scenario_meta,
                sector_df_current,
                portfolio_table,
                ai_summary_for_report,
            )
            st.download_button(
                "Download PDF report",
                data=pdf_bytes,
                file_name="katta_report.pdf",
                mime="application/pdf",
            )

# Footer
st.markdown("---")
st.caption(
    "Katta MacroSuite — decision-support analytics with Massive live data. Not investment advice."
)
