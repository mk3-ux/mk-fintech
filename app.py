import os
import requests
import streamlit as st
import pandas as pd
import altair as alt
from groq import Groq
from fpdf import FPDF

# ---------------------------
# Config / API Keys
# ---------------------------

# 🔴 IMPORTANT: paste your real keys here on your machine
MASSIVE_API_KEY = "Q1pmrfqv0vV6caqxpJUjwcyEsSEvvSJU"   # e.g. Q1pmrfqv0vV6caqxpJUj...
NEWSAPI_KEY = "4f0f0589094c414a8ef178ee05c9226d"

# Optional: you can also hard-code Groq key instead of env if you want
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "llama-3.1-8b-instant"
MASSIVE_BASE_URL = "https://api.massive.com"
NEWSAPI_URL = "https://newsapi.org/v2/everything"

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

st.set_page_config(
    page_title="Katta Finsight",
    layout="wide",
    page_icon="📊",
)

# Light theme colors
COLORS = {
    "bg": "#F7FAFF",
    "text": "#0F172A",
    "card": "#FFFFFF",
    "accent": "#0EA5E9",
    "subtle": "#E6EEF8",
}

# Small CSS polish
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
# Static data
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

# very simple "how sensitive is this sector to the stock move" weights
SECTOR_SENSITIVITY = {
    "Tech": 1.0,
    "Real Estate": 0.6,
    "Luxury / Discretionary": 0.8,
    "Bonds": -0.4,
    "Energy": 0.5,
    "Consumer Staples": 0.3,
    "Banks": 0.7,
}

# ---------------------------
# Helpers: Massive + NewsAPI
# ---------------------------

def fetch_massive_snapshot(ticker: str):
    """
    Call Massive single-ticker snapshot endpoint.

    GET /v2/snapshot/locale/us/markets/stocks/tickers/{stocksTicker}

    Returns (snapshot_dict, None) on success
    or (None, error_message) on failure.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        return None, "No ticker provided."

    if not MASSIVE_API_KEY or MASSIVE_API_KEY == "YOUR_MASSIVE_API_KEY_HERE":
        return None, "MASSIVE_API_KEY is missing. Paste your key in the code."

    url = f"{MASSIVE_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
    headers = {"Authorization": f"Bearer {MASSIVE_API_KEY}"}

    try:
        resp = requests.get(url, headers=headers, timeout=8)
        data = resp.json()
    except Exception as e:
        return None, f"Network error calling Massive: {e}"

    if resp.status_code != 200:
        status = data.get("status") if isinstance(data, dict) else str(data)
        return None, f"Massive API error {resp.status_code}: {status}"

    ticker_obj = data.get("ticker") or {}
    last_trade = ticker_obj.get("lastTrade") or {}

    price = last_trade.get("p")
    change = ticker_obj.get("todaysChange")
    change_pct = ticker_obj.get("todaysChangePerc")
    updated = ticker_obj.get("updated")

    if price is None:
        return None, "Snapshot returned no lastTrade price (plan may not include trades)."

    snapshot = {
        "symbol": ticker_obj.get("ticker", ticker),
        "price": price,
        "change": change,
        "change_pct": change_pct,
        "updated": updated,
        "raw": data,
    }
    return snapshot, None


def fetch_news(keyword: str, page_size: int = 6):
    """
    Fetch finance/news headlines for the given keyword using NewsAPI.
    """
    if not NEWSAPI_KEY or NEWSAPI_KEY == "YOUR_NEWSAPI_KEY_HERE":
        return [], "NEWSAPI_KEY is missing. Paste your key in the code."

    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY,
    }
    try:
        resp = requests.get(NEWSAPI_URL, params=params, timeout=8)
        data = resp.json()
    except Exception as e:
        return [], f"Network error calling NewsAPI: {e}"

    if resp.status_code != 200:
        status = data.get("message") if isinstance(data, dict) else str(data)
        return [], f"NewsAPI error {resp.status_code}: {status}"

    return data.get("articles", []), None

# ---------------------------
# Helpers: scoring, AI, report
# ---------------------------

def compute_stock_sector_impacts(stock_move_pct: float, primary_sector: str) -> pd.DataFrame:
    """
    Very simple model:
    - The primary sector has sensitivity 1.0
    - Other sectors use SECTOR_SENSITIVITY
    Impact Score is scaled to +/-5 so it's easy to read.
    """
    rows = []
    for sec in SECTORS:
        base_sens = SECTOR_SENSITIVITY.get(sec, 0.4)
        sensitivity = 1.0 if sec == primary_sector else base_sens
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


def compute_portfolio_exposure(portfolio_df: pd.DataFrame, sector_scores: pd.DataFrame):
    """
    Expects portfolio_df with columns: Sector, Allocation (as percent or fraction)
    Returns weighted exposure and breakdown.
    """
    df = portfolio_df.copy()
    if "Allocation" not in df.columns:
        raise ValueError("Uploaded portfolio must contain 'Allocation' column")

    total = df["Allocation"].sum()
    df["Weight"] = df["Allocation"] / total if total else 0.0

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
        "Your task: convert stock scenarios and sector outputs into concise internal research narratives, "
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
# UI helpers
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
              <div style="font-size:18px;font-weight:800;color:{COLORS['text']};">Katta MacroSuite — Live Markets Intelligence</div>
              <div style="font-size:12px;color:#475569;">
                Live stock snapshot • Sector sensitivity • Portfolio exposure • AI research • PDF reports
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_header()

# ---------------------------
# Session state
# ---------------------------

if "sector_df" not in st.session_state:
    st.session_state["sector_df"] = None
if "scenario_name" not in st.session_state:
    st.session_state["scenario_name"] = None
if "scenario_meta" not in st.session_state:
    st.session_state["scenario_meta"] = {}
if "portfolio_df" not in st.session_state:
    st.session_state["portfolio_df"] = None
if "ai_history" not in st.session_state:
    st.session_state["ai_history"] = []

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

    if not MASSIVE_API_KEY or MASSIVE_API_KEY == "YOUR_MASSIVE_API_KEY_HERE":
        st.info("Add your Massive API key in the code to enable live prices.")

    if not NEWSAPI_KEY or NEWSAPI_KEY == "YOUR_NEWSAPI_KEY_HERE":
        st.info("Add your NewsAPI key in the code to enable headlines.")

    if client is None:
        st.warning("Groq API not configured — AI Research Analyst disabled until GROQ_API_KEY is set.")

    st.markdown(
        "Upload a CSV portfolio (columns: `Sector`, `Allocation`). "
        "Allocation can be percent or units."
    )

    st.markdown("---")

    st.caption(
        "Decision-support only. This dashboard does NOT provide investment advice."
    )

# ---------------------------
# Tabs
# ---------------------------

tab_live, tab_portfolio, tab_ai, tab_reports = st.tabs(
    [
        "Live Stock Dashboard",
        "Portfolio Analyzer",
        "AI Research Analyst",
        "Generate Report",
    ]
)

# ---------- Live Stock Dashboard ----------
with tab_live:
    st.subheader("Live Stock Dashboard — Massive + NewsAPI")

    col_input1, col_input2 = st.columns([2, 1])
    with col_input1:
        ticker = st.text_input("Stock ticker", "AAPL").upper().strip()
    with col_input2:
        primary_sector = st.selectbox(
            "Primary sector for this stock",
            SECTORS,
            index=0,
            help="Which sector best represents this stock?",
        )

    use_live = st.checkbox("Use Massive live snapshot (if available)", value=True)

    snapshot = None
    live_error = None
    if use_live and ticker:
        snapshot, live_error = fetch_massive_snapshot(ticker)

    if snapshot is None:
        if live_error:
            st.warning(f"Could not load live data from Massive: {live_error}")
        st.info(
            "You can still use the sliders below for a manual 'what-if' scenario even without live data."
        )

    # Metrics row
    col_m1, col_m2, col_m3 = st.columns(3)
    default_move = 0.0

    if snapshot is not None:
        price = snapshot["price"]
        change = snapshot["change"]
        change_pct = snapshot["change_pct"]

        with col_m1:
            st.metric("Last Price", f"${price:,.2f}")
        with col_m2:
            if change is not None:
                st.metric("Today's Change", f"{change:+.2f}")
            else:
                st.metric("Today's Change", "n/a")
        with col_m3:
            if change_pct is not None:
                st.metric("Change %", f"{change_pct:+.2f}%")
                default_move = float(change_pct)
            else:
                st.metric("Change %", "n/a")

    st.markdown("### What-if: sector sensitivity to this stock move")

    col_sl, col_txt = st.columns([2, 3])
    with col_sl:
        stock_move = st.slider(
            "Assumed stock move (%)",
            -20.0,
            20.0,
            float(round(default_move, 1)) if snapshot is not None else 0.0,
            step=0.5,
            help="Negative = stock down, Positive = stock up.",
        )
    with col_txt:
        st.write(
            "This is a toy model for how much each sector **could** feel this stock move. "
            "Primary sector gets the strongest effect; others get smaller spillovers."
        )

    sector_df = compute_stock_sector_impacts(stock_move, primary_sector)
    st.session_state["sector_df"] = sector_df
    scenario_name = f"{ticker} move {stock_move:+.1f}%"
    scenario_meta = {
        "Stock": ticker,
        "Move (%)": stock_move,
        "Primary Sector": primary_sector,
        "Live price used": bool(snapshot),
    }
    st.session_state["scenario_name"] = scenario_name
    st.session_state["scenario_meta"] = scenario_meta

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("#### Sector Impact Table")
        st.dataframe(
            sector_df.style.format({"Impact Score": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        st.markdown("#### Sector Impact Bars")
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
        f"- **Most positively exposed sectors (toy model):** {winner_text}  \n"
        f"- **Most negatively exposed / least helped:** {loser_text}"
    )

    # Integrated news panel
    st.markdown("### Live Headlines for this stock / theme")
    if ticker:
        articles, news_err = fetch_news(ticker)
        if news_err:
            st.warning(f"Could not load news: {news_err}")
        elif not articles:
            st.info("No news articles found for this keyword right now.")
        else:
            for art in articles:
                title = art.get("title", "No title")
                source = (art.get("source") or {}).get("name")
                url = art.get("url")
                desc = art.get("description")

                st.markdown(f"**📰 {title}**")
                if source:
                    st.caption(f"Source: {source}")
                if desc:
                    st.write(desc)
                if url:
                    st.markdown(f"[Read more]({url})")
                st.markdown("---")
    else:
        st.info("Enter a ticker symbol above to see news headlines.")

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
                "No scenario found. Configure a stock move on the 'Live Stock Dashboard' tab first."
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
                        "Portfolio tilt: mild to strong positive sensitivity to the current stock scenario."
                    )
                elif score < -1.5:
                    st.warning(
                        "Portfolio tilt: mild to strong negative sensitivity to the current stock scenario."
                    )
                else:
                    st.info(
                        "Portfolio tilt: largely neutral under the chosen stock scenario."
                    )
            except Exception as e:
                st.error(f"Analysis error: {e}")

# ---------- AI Research Analyst ----------
with tab_ai:
    st.subheader("AI Research Analyst — Internal Narratives")
    st.markdown(
        "Draft internal memos, scenario summaries, and risk notes. "
        "AI must be configured with `GROQ_API_KEY`."
    )

    user_q = st.text_area(
        "Ask the AI Research Analyst "
        "(e.g., 'Write an executive summary of this stock scenario for PMs')",
        height=140,
    )
    ai_style = st.selectbox(
        "AI style", ["Professional", "Executive", "Technical"], index=0
    )

    if st.button("Run AI"):
        if client is None:
            st.error(
                "AI not configured. Set GROQ_API_KEY as an environment variable or in your secrets."
            )
        else:
            sector_df_current = st.session_state.get("sector_df")
            scenario_name = st.session_state.get("scenario_name", "Current scenario")
            scenario_meta = st.session_state.get("scenario_meta", {})

            if sector_df_current is None:
                st.error(
                    "No scenario found. Configure a stock move on the 'Live Stock Dashboard' tab first."
                )
            else:
                context = (
                    f"Scenario: {scenario_name}\nScenario inputs: {scenario_meta}\nTop sector impacts:\n"
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
                        st.session_state["ai_history"], prompt, ai_style
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
                "No scenario found. Configure a stock move on the 'Live Stock Dashboard' tab first."
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

# Footer / notes
st.markdown("---")
st.caption(
    "Katta MacroSuite — decision-support analytics only. Not investment advice."
)
