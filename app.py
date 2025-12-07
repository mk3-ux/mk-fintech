import os
import streamlit as st
import pandas as pd
import altair as alt
from groq import Groq
from fpdf import FPDF
import requests

# ---------------------------
# Config / Clients
# ---------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

st.set_page_config(
    page_title="Katta Finsight",
    layout="wide",
    page_icon="📊",
)

COLORS = {
    "bg": "#F7FAFF",
    "text": "#0F172A",
    "accent": "#0EA5E9",
    "subtle": "#E6EEF8",
}

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}
        .block-container {{ max-width: 1200px; padding-top: 1rem; padding-bottom: 2rem; }}
        button[data-baseweb="tab"] {{ border-radius: 999px !important; padding: 0.3rem 1rem !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# News API
# ---------------------------

NEWSAPI_KEY = "4f0f0589094c414a8ef178ee05c9226d"
NEWSAPI_URL = "https://newsapi.org/v2/everything"


def fetch_news(keyword: str = "finance", page_size: int = 5):
    """Fetch latest news articles for a given keyword using NewsAPI."""
    if not NEWSAPI_KEY:
        return []
    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY,
    }
    try:
        resp = requests.get(NEWSAPI_URL, params=params, timeout=10)
        data = resp.json()
        if resp.status_code == 200 and "articles" in data:
            return data["articles"]
        else:
            return []
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []


# ---------------------------
# Model / Data definitions
# ---------------------------

sectors = [
    "Tech",
    "Real Estate",
    "Luxury / Discretionary",
    "Bonds",
    "Energy",
    "Consumer Staples",
    "Banks",
]

DEFAULT_SECTOR_STOCKS = {
    "Tech": ["AAPL", "MSFT", "NVDA"],
    "Real Estate": ["SPG", "PSA"],
    "Luxury / Discretionary": ["TSLA", "NKE", "SBUX"],
    "Bonds": ["TLT", "IEF"],
    "Energy": ["XOM", "CVX"],
    "Consumer Staples": ["KO", "WMT", "COST"],
    "Banks": ["JPM", "BAC", "WFC"],
}


def label_from_score(score: float) -> str:
    if score <= -3.5:
        return "Strong Negative"
    if score <= -1.5:
        return "Mild Negative"
    if score < 1.5:
        return "Neutral"
    if score < 3.5:
        return "Mild Positive"
    return "Strong Positive"


def compute_sector_impacts(sector_move: float, shocked_sector: str) -> pd.DataFrame:
    """
    Sector → Sector sensitivity.
    Shocked sector gets sensitivity 1.0, others 0.4.
    """
    rows = []
    for sec in sectors:
        sensitivity = 1.0 if sec == shocked_sector else 0.4
        raw_score = sensitivity * sector_move
        rows.append({"Sector": sec, "Raw Score": raw_score})

    df = pd.DataFrame(rows)
    max_abs = df["Raw Score"].abs().max()
    if not max_abs:
        max_abs = 1.0
    df["Impact Score"] = (df["Raw Score"] / max_abs * 5).round(2)
    df["Impact Label"] = df["Impact Score"].apply(label_from_score)
    return df[["Sector", "Impact Score", "Impact Label"]]


def compute_portfolio_exposure(portfolio_df: pd.DataFrame, sector_scores: pd.DataFrame):
    """
    Expects portfolio_df with columns: Sector, Allocation (as percent or fraction).
    Returns weighted exposure and breakdown.
    """
    df = portfolio_df.copy()
    if "Sector" not in df.columns or "Allocation" not in df.columns:
        raise ValueError("Uploaded portfolio must contain 'Sector' and 'Allocation' columns")
    total = df["Allocation"].sum()
    if total == 0:
        df["Weight"] = 0.0
    else:
        df["Weight"] = df["Allocation"] / total

    merged = df.merge(sector_scores, on="Sector", how="left")
    merged["Impact Score"].fillna(0.0, inplace=True)
    merged["Weighted Impact"] = merged["Weight"] * merged["Impact Score"]
    portfolio_score = merged["Weighted Impact"].sum()
    return portfolio_score, merged[["Sector", "Allocation", "Weight", "Impact Score", "Weighted Impact"]]


def call_ai_research(history, user_text: str, level: str = "Professional") -> str:
    """Call Groq Llama as a corporate research analyst (concepts, narratives)."""
    if client is None:
        return "AI Research Analyst not configured. Set GROQ_API_KEY in environment or Streamlit Secrets."
    level_line = {
        "Professional": "Write as a corporate research analyst for internal use, concise and structured.",
        "Executive": "Write a polished executive summary for C-suite consumption, ~3 paragraphs.",
        "Technical": "Write a detailed technical memo with supporting reasoning and caveats.",
    }.get(level, "Write in clear professional language.")
    system_prompt = (
        "You are an internal AI research analyst for an institutional client. "
        "Your task: convert sector shock inputs and sector outputs into concise internal research narratives, "
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
    stock_df: pd.DataFrame,
    sector_df: pd.DataFrame = None,
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

    # Sector impacts (optional but nice)
    if sector_df is not None and not sector_df.empty:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Sector Impacts:", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for _, r in sector_df.iterrows():
            pdf.cell(0, 7, f"{r['Sector']}: {r['Impact Score']} ({r['Impact Label']})", ln=True)
        pdf.ln(4)

    # Stock impacts
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Stock Impacts:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for _, r in stock_df.iterrows():
        pdf.cell(
            0,
            6,
            f"{r['Stock']}: {r['Impact Score']} ({r['Impact Label']})",
            ln=True,
        )
    pdf.ln(4)

    # Portfolio
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

    # AI summary
    if ai_summary:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "AI Research Summary:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for line in ai_summary.split("\n"):
            pdf.multi_cell(0, 6, line)

    return pdf.output(dest="S").encode("latin-1")


# ---------------------------
# UI Rendering
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
              <div style="font-size:18px;font-weight:800;color:{COLORS['text']};">Katta MacroSuite — Sector → Stock Intelligence</div>
              <div style="font-size:12px;color:#475569;">Sector shocks • Stock impact • Portfolio exposure • Live news • Internal research automation</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_header()

# Sidebar
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

    if not NEWSAPI_KEY:
        st.info("Set NEWSAPI_KEY for live sector news.")
    else:
        st.caption("News headlines powered by NewsAPI.org")

    st.caption(
        "This platform is a decision-support tool. "
        "It does NOT provide buy/sell advice."
    )

# Init session state
if "sector_df" not in st.session_state:
    st.session_state["sector_df"] = None
if "stock_df" not in st.session_state:
    st.session_state["stock_df"] = None
if "scenario_name" not in st.session_state:
    st.session_state["scenario_name"] = None
if "scenario_meta" not in st.session_state:
    st.session_state["scenario_meta"] = {}
if "portfolio_df" not in st.session_state:
    st.session_state["portfolio_df"] = None
if "ai_history" not in st.session_state:
    st.session_state["ai_history"] = []

# Main tabs
tab_explorer, tab_portfolio, tab_ai, tab_reports = st.tabs(
    [
        "Sector → Stock Impact (with News)",
        "Portfolio Analyzer",
        "AI Research Analyst",
        "Generate Report",
    ]
)

# ---------- Sector → Stock Explorer ----------
with tab_explorer:
    st.subheader("Sector Shock → Stocks & Sectors Impact")

    shocked_sector = st.selectbox(
        "Sector to shock",
        sectors,
        index=0,
        help="Choose which sector is experiencing the shock (move up or down).",
    )
    sector_move = st.slider(
        "Assumed sector move (%)",
        -20,
        20,
        0,
        help="Negative = sector underperforms, Positive = sector outperforms",
    )

    default_stocks = DEFAULT_SECTOR_STOCKS.get(shocked_sector, [])
    default_stock_str = ", ".join(default_stocks)
    stock_str = st.text_input(
        f"Stocks in {shocked_sector} (comma-separated tickers)",
        value=default_stock_str,
    )
    stocks = [s.strip().upper() for s in stock_str.split(",") if s.strip()]

    st.caption(
        "This section shows how a move in a **sector** flows through to all sectors, "
        "and then maps that sector's impact to the stocks you listed."
    )

    # Sector-level impacts
    sector_df = compute_sector_impacts(sector_move, shocked_sector)

    scenario_name = f"{shocked_sector} sector move {sector_move:+.1f}%"
    scenario_meta = {
        "Shocked Sector": shocked_sector,
        "Sector Move (%)": sector_move,
        "Stocks in Sector": ", ".join(stocks),
    }

    # Stock-level impact: all listed stocks inherit the shocked sector's Impact Score
    if stocks:
        shocked_row = sector_df[sector_df["Sector"] == shocked_sector].iloc[0]
        stock_score = shocked_row["Impact Score"]
        stock_label = shocked_row["Impact Label"]
        stock_rows = [
            {
                "Stock": s,
                "Sector": shocked_sector,
                "Impact Score": stock_score,
                "Impact Label": stock_label,
            }
            for s in stocks
        ]
        stock_df = pd.DataFrame(stock_rows)
    else:
        stock_df = pd.DataFrame(columns=["Stock", "Sector", "Impact Score", "Impact Label"])

    # Save in session for other tabs
    st.session_state["sector_df"] = sector_df
    st.session_state["stock_df"] = stock_df
    st.session_state["scenario_name"] = scenario_name
    st.session_state["scenario_meta"] = scenario_meta

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("#### Sector Impact Overview")
        st.dataframe(
            sector_df.style.format({"Impact Score": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.markdown("#### Visual Overview (Sectors)")
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

    st.markdown("#### Quick take (by sector)")
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
        f"- **Most positively affected sectors:** {winner_text}  \n"
        f"- **Most negatively affected sectors:** {loser_text}"
    )

    # Stock-level impact for the shocked sector
    st.markdown("### Stock Impact in the Shocked Sector")
    if not stock_df.empty:
        col3, col4 = st.columns([2, 3])
        with col3:
            st.markdown("#### Stocks in Shocked Sector")
            st.dataframe(stock_df, use_container_width=True)
        with col4:
            st.markdown("#### Visual Overview (Stocks)")
            stock_chart = (
                alt.Chart(stock_df)
                .mark_bar()
                .encode(
                    x=alt.X("Stock:N", sort=None),
                    y=alt.Y("Impact Score:Q"),
                    tooltip=["Stock", "Sector", "Impact Score", "Impact Label"],
                )
                .properties(height=320)
            )
            st.altair_chart(stock_chart, use_container_width=True)

        st.markdown("#### Quick take (by stock)")
        st.write(
            f"- **Shocked sector:** {shocked_sector} ({stock_label})\n"
            f"- **All listed stocks in this sector inherit that impact in this simplified model.**"
        )
    else:
        st.info("Add at least one stock ticker above to see stock-level impact.")

    # Integrated news in this section
    st.markdown("### 📰 Latest News for This Sector")
    news_keyword = shocked_sector.split("/")[0]  # use main word like "Tech"
    articles = fetch_news(news_keyword, page_size=5)

    if articles:
        for article in articles:
            title = article.get("title", "No title")
            desc = article.get("description", "")
            url = article.get("url", "#")
            st.markdown(
                f"<div style='padding:10px;border-left:4px solid #0EA5E9;margin-bottom:10px;'>"
                f"<div style='font-weight:700;font-size:15px;'>📰 {title}</div>"
                f"<div style='font-size:13px;margin-top:4px;'>{desc}</div>"
                f"<div style='margin-top:4px;'><a href='{url}' target='_blank'>Read more →</a></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No news available or NewsAPI returned no articles for this sector keyword.")

    st.caption(
        "This is a simplified, educational scenario engine — not a real-world risk model or investment advice."
    )

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

    if st.button("Analyze current sector scenario exposure"):
        sector_df_current = st.session_state.get("sector_df")
        portfolio_df_current = st.session_state.get("portfolio_df")

        if sector_df_current is None:
            st.error(
                "No scenario found. Please configure a sector shock on the 'Sector → Stock Impact (with News)' tab first."
            )
        elif portfolio_df_current is None:
            st.error("Please upload a portfolio CSV first.")
        else:
            try:
                score, breakdown = compute_portfolio_exposure(
                    portfolio_df_current, sector_df_current
                )
                st.metric("Portfolio Weighted Impact Score", f"{score:.3f}")
                st.markdown("#### Breakdown:")
                st.dataframe(breakdown, use_container_width=True)

                if score > 1.5:
                    st.success(
                        "Portfolio tilt: Mild to strong positive sensitivity to the current sector scenario."
                    )
                elif score < -1.5:
                    st.warning(
                        "Portfolio tilt: Mild to strong negative sensitivity to the current sector scenario."
                    )
                else:
                    st.info(
                        "Portfolio tilt: Largely neutral under the chosen sector scenario."
                    )
            except Exception as e:
                st.error(f"Analysis error: {e}")

# ---------- AI Research Analyst ----------
with tab_ai:
    st.subheader("AI Research Analyst — Generate internal narratives")
    st.markdown(
        "Draft internal memos, scenario summaries, and risk notes about this sector shock, "
        "its stock impact, and portfolio implications. (AI must be configured with GROQ_API_KEY.)"
    )

    user_q = st.text_area(
        "Ask the AI Research Analyst (e.g., 'Write an executive summary of this Tech sector shock and its impact on listed stocks')",
        height=120,
    )
    ai_style = st.selectbox(
        "AI style", ["Professional", "Executive", "Technical"], index=0
    )

    if st.button("Run AI"):
        if client is None:
            st.error("AI not configured. Set GROQ_API_KEY as an environment variable.")
        else:
            sector_df_current = st.session_state.get("sector_df")
            scenario_name = st.session_state.get("scenario_name", "Current scenario")
            scenario_meta = st.session_state.get("scenario_meta", {})
            stock_df_current = st.session_state.get("stock_df")

            if sector_df_current is None or stock_df_current is None:
                st.error(
                    "No scenario found. Please configure a sector shock on the 'Sector → Stock Impact (with News)' tab first."
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
                context += "\nStocks in shocked sector:\n"
                for _, r in stock_df_current.iterrows():
                    context += (
                        f"- {r['Stock']}: {r['Impact Score']} ({r['Impact Label']})\n"
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
    report_title = st.text_input("Report title", "Sector Scenario & Portfolio Insight")
    include_portfolio = st.checkbox(
        "Include uploaded portfolio exposure (if available)", value=True
    )
    ai_summary_for_report = st.text_area(
        "Optional: paste AI summary to include in report", height=120
    )

    if st.button("Create PDF Report"):
        sector_df_current = st.session_state.get("sector_df")
        stock_df_current = st.session_state.get("stock_df")
        scenario_name = st.session_state.get("scenario_name", "Current scenario")
        scenario_meta = st.session_state.get("scenario_meta", {})
        portfolio_df_current = st.session_state.get("portfolio_df")

        if sector_df_current is None or stock_df_current is None or stock_df_current.empty:
            st.error(
                "No scenario found. Please configure a sector shock and stocks on the 'Sector → Stock Impact (with News)' tab first."
            )
        else:
            portfolio_table = None
            if include_portfolio and portfolio_df_current is not None:
                try:
                    score, portfolio_table = compute_portfolio_exposure(
                        portfolio_df_current, sector_df_current
                    )
                except Exception as e:
                    st.error(f"Cannot include portfolio: {e}")
                    portfolio_table = None

            pdf_bytes = create_pdf_report(
                report_title,
                scenario_name,
                scenario_meta,
                stock_df_current,
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
    "Katta MacroSuite — decision-support analytics. Not investment advice. For internal corporate / educational use."
)
