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
# News API (already configured)
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
# Real-time stock data via Yahoo Finance (no extra installs)
# ---------------------------

def fetch_stock_quote(symbol: str):
    """
    Fetch current quote for a symbol from Yahoo Finance public API.
    This is lightweight and free; no extra libraries needed.
    """
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    params = {"symbols": symbol}
    try:
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        result = data.get("quoteResponse", {}).get("result", [])
        if not result:
            return None
        q = result[0]
        return {
            "symbol": q.get("symbol"),
            "name": q.get("shortName") or q.get("longName") or q.get("symbol"),
            "price": q.get("regularMarketPrice"),
            "change": q.get("regularMarketChange"),
            "change_pct": q.get("regularMarketChangePercent"),
            "currency": q.get("currency"),
        }
    except Exception as e:
        print("quote error", e)
        return None


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

# Simple multi-factor “stock driver” model (by sector)
# Think of these as drivers of STOCK returns, not macro GDP.
factors = ["Market", "Rates", "Growth Stocks Tilt", "Oil", "FX"]

factor_loadings = {
    "Tech": {"Market": 1.2, "Rates": -0.8, "Growth Stocks Tilt": 1.1, "Oil": -0.3, "FX": 0.3},
    "Real Estate": {"Market": 1.0, "Rates": -1.2, "Growth Stocks Tilt": 0.7, "Oil": -0.2, "FX": 0.1},
    "Luxury / Discretionary": {"Market": 1.3, "Rates": -0.7, "Growth Stocks Tilt": 1.2, "Oil": -0.4, "FX": 0.2},
    "Bonds": {"Market": -0.3, "Rates": -1.5, "Growth Stocks Tilt": -0.4, "Oil": -0.1, "FX": 0.1},
    "Energy": {"Market": 1.0, "Rates": -0.3, "Growth Stocks Tilt": 0.5, "Oil": 1.6, "FX": -0.1},
    "Consumer Staples": {"Market": 0.6, "Rates": -0.2, "Growth Stocks Tilt": 0.3, "Oil": -0.1, "FX": 0.1},
    "Banks": {"Market": 1.1, "Rates": 0.8, "Growth Stocks Tilt": 0.6, "Oil": 0.1, "FX": 0.0},
}


# ---------------------------
# Helper functions
# ---------------------------

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


def compute_factor_exposures_for_portfolio(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute factor (stock driver) exposures for a portfolio given sector-level factor loadings.
    """
    df = portfolio_df.copy()
    if "Sector" not in df.columns or "Allocation" not in df.columns:
        raise ValueError("Portfolio must have 'Sector' and 'Allocation' columns")
    total = df["Allocation"].sum()
    if total == 0:
        df["Weight"] = 0.0
    else:
        df["Weight"] = df["Allocation"] / total

    rows = []
    for _, row in df.iterrows():
        sec = row["Sector"]
        w = row["Weight"]
        fl = factor_loadings.get(sec, {})
        for fac in factors:
            rows.append(
                {
                    "Factor": fac,
                    "Contribution": w * fl.get(fac, 0.0),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["Factor", "Exposure"])

    temp = pd.DataFrame(rows)
    exposures = temp.groupby("Factor")["Contribution"].sum().reset_index()
    exposures.rename(columns={"Contribution": "Exposure"}, inplace=True)
    return exposures


def compute_factor_exposures_for_sector(shocked_sector: str) -> pd.DataFrame:
    """
    Show factor loadings for a single sector (normalized view).
    """
    fl = factor_loadings.get(shocked_sector, {})
    rows = []
    for fac in factors:
        rows.append({"Factor": fac, "Exposure": fl.get(fac, 0.0)})
    return pd.DataFrame(rows)


def estimate_impact_from_scenario(exposures_df: pd.DataFrame, scenario: dict) -> float:
    """
    exposures_df: columns Factor, Exposure
    scenario: dict of {Factor: ShockSize}
    Returns scalar impact score.
    """
    if exposures_df is None or exposures_df.empty:
        return 0.0
    merged = exposures_df.copy()
    merged["Shock"] = merged["Factor"].apply(lambda f: scenario.get(f, 0.0))
    merged["Impact"] = merged["Exposure"] * merged["Shock"]
    return merged["Impact"].sum()


def simple_sentiment(text: str) -> str:
    """
    Tiny lexicon-based sentiment scorer for news headlines/description.
    Just to categorize as Positive / Negative / Neutral.
    """
    if not text:
        return "Neutral"
    t = text.lower()
    positive_words = ["gain", "growth", "beat", "surge", "rally", "strong", "up", "record"]
    negative_words = ["loss", "fall", "drop", "plunge", "fear", "crisis", "down", "weak"]

    pos = sum(w in t for w in positive_words)
    neg = sum(w in t for w in negative_words)

    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"


def call_ai_research(history, user_text: str, level: str = "Professional") -> str:
    """Call Groq Llama as a corporate research analyst (stock/sector narratives)."""
    if client is None:
        return "AI Research Analyst not configured. Set GROQ_API_KEY in environment or Streamlit Secrets."
    level_line = {
        "Professional": "Write as a corporate research analyst for internal use, concise and structured.",
        "Executive": "Write a polished executive summary for C-suite or PMs, ~3 paragraphs.",
        "Technical": "Write a detailed technical memo with supporting reasoning and caveats.",
    }.get(level, "Write in clear professional language.")
    system_prompt = (
        "You are an internal AI research analyst for an institutional trading desk. "
        "Your task: convert sector shocks, stock reactions, factor exposures, portfolio tilts, "
        "and news sentiment into concise internal research narratives, "
        "scenario summaries, risk notes and suggested topics for follow-up. "
        "You MUST NOT provide direct buy/sell recommendations, price targets or personalized investment advice. "
        + level_line
    )
    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})
    
    # ✅ FIXED LINE (removed extra ')')
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

    if sector_df is not None and not sector_df.empty:
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

    if stock_df is not None and not stock_df.empty:
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
              <div style="font-size:18px;font-weight:800;color:{COLORS['text']};">Katta MacroSuite — Stock & Sector Intelligence</div>
              <div style="font-size:12px;color:#475569;">Sector shocks • Stock impact • Factor risk • Market scenarios • News sentiment • Internal research automation</div>
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
if "news_cache" not in st.session_state:
    st.session_state["news_cache"] = []

# Tabs (added Hedge Fund Dashboard tab)
tab_dash, tab_explorer, tab_portfolio, tab_risk, tab_ai, tab_reports = st.tabs(
    [
        "Hedge Fund Dashboard",
        "Sector → Stock Impact (with News)",
        "Portfolio Analyzer",
        "Risk & Market Scenarios",
        "AI Research Analyst",
        "Generate Report",
    ]
)

# ---------- Hedge Fund Dashboard ----------
with tab_dash:
    st.subheader("Hedge Fund Dashboard — Live Watchlist & Scenario Snapshot")

    default_watchlist = "AAPL, MSFT, NVDA, SPY, QQQ, XOM, JPM"
    watchlist_str = st.text_input(
        "Watchlist tickers (comma-separated)",
        value=default_watchlist,
    )
    symbols = [s.strip().upper() for s in watchlist_str.split(",") if s.strip()]

    quotes = []
    for sym in symbols:
        q = fetch_stock_quote(sym)
        if q:
            quotes.append(q)

    if quotes:
        quote_df = pd.DataFrame(quotes)
        st.markdown("### Live Watchlist")
        st.dataframe(
            quote_df[["symbol", "name", "price", "change", "change_pct", "currency"]],
            use_container_width=True,
        )

        st.markdown("#### Intraday Moves (%)")
        move_chart = (
            alt.Chart(quote_df)
            .mark_bar()
            .encode(
                x=alt.X("symbol:N", title="Ticker"),
                y=alt.Y("change_pct:Q", title="Change %"),
                tooltip=["symbol", "name", "price", "change_pct"],
            )
            .properties(height=260)
        )
        st.altair_chart(move_chart, use_container_width=True)
    else:
        st.info("No live quotes available (possibly no internet or Yahoo structure changed).")

    st.markdown("### Current Scenario Snapshot")
    sector_df_current = st.session_state.get("sector_df")
    stock_df_current = st.session_state.get("stock_df")
    scenario_name = st.session_state.get("scenario_name", "No scenario configured")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.metric("Active Scenario", scenario_name)
        if sector_df_current is not None and not sector_df_current.empty:
            best = sector_df_current.sort_values("Impact Score", ascending=False).iloc[0]
            worst = sector_df_current.sort_values("Impact Score", ascending=True).iloc[0]
            st.metric("Top Sector", f"{best['Sector']} ({best['Impact Label']})")
            st.metric("Bottom Sector", f"{worst['Sector']} ({worst['Impact Label']})")
        else:
            st.caption("Configure a sector shock in the next tab to see sector snapshot.")

    with col_d2:
        if stock_df_current is not None and not stock_df_current.empty:
            st.markdown("**Stocks in Shocked Sector**")
            st.dataframe(stock_df_current, use_container_width=True)
        else:
            st.caption("Add stocks in the Sector → Stock tab to see stock impact.")

    st.markdown("### Recent Sector News Sentiment")
    news_articles = st.session_state.get("news_cache", [])
    if news_articles:
        sentiments = []
        for art in news_articles[:5]:
            title = art.get("title", "")
            desc = art.get("description", "")
            sent = simple_sentiment((title or "") + " " + (desc or ""))
            sentiments.append(sent)
        pos = sentiments.count("Positive")
        neg = sentiments.count("Negative")
        neu = sentiments.count("Neutral")
        st.write(f"**Headline sentiment:** {pos} positive • {neg} negative • {neu} neutral")
    else:
        st.caption("No news cached yet — visit the Sector → Stock tab to load sector news.")

    st.caption("Use this dashboard as a PM-style snapshot: watchlist + scenario + sentiment.")


# ---------- Sector → Stock Explorer ----------
with tab_explorer:
    st.subheader("Sector Shock → Stocks & Sectors Impact")

    shocked_sector = st.selectbox(
        "Sector to shock",
        sectors,
        index=0,
        help="Choose which sector is experiencing the move (up or down).",
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

    sector_df = compute_sector_impacts(sector_move, shocked_sector)

    scenario_name = f"{shocked_sector} sector move {sector_move:+.1f}%"
    scenario_meta = {
        "Shocked Sector": shocked_sector,
        "Sector Move (%)": sector_move,
        "Stocks in Sector": ", ".join(stocks),
    }

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
        stock_label = "Neutral"

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

    st.markdown("### 📰 Latest News & Sentiment for This Sector")
    news_keyword = shocked_sector.split("/")[0]
    articles = fetch_news(news_keyword, page_size=5)
    st.session_state["news_cache"] = articles

    sentiments = []
    if articles:
        for article in articles:
            title = article.get("title", "No title")
            desc = article.get("description", "")
            url = article.get("url", "#")
            sent = simple_sentiment((title or "") + " " + (desc or ""))
            sentiments.append(sent)

            badge = {
                "Positive": "background:#16a34a;color:white;",
                "Negative": "background:#dc2626;color:white;",
                "Neutral": "background:#6b7280;color:white;",
            }.get(sent, "background:#6b7280;color:white;")

            st.markdown(
                f"<div style='padding:10px;border-left:4px solid #0EA5E9;margin-bottom:10px;'>"
                f"<div style='font-weight:700;font-size:15px;'>📰 {title}</div>"
                f"<div style='font-size:12px;margin:4px 0;'>{desc}</div>"
                f"<span style='font-size:11px;padding:2px 6px;border-radius:999px;{badge}'>Sentiment: {sent}</span>"
                f"<div style='margin-top:4px;'><a href='{url}' target='_blank'>Read more →</a></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No news available or NewsAPI returned no articles for this sector keyword.")

    if sentiments:
        pos_count = sentiments.count("Positive")
        neg_count = sentiments.count("Negative")
        neu_count = sentiments.count("Neutral")
        st.markdown(
            f"**Headline sentiment summary:** {pos_count} positive • {neg_count} negative • {neu_count} neutral"
        )

    st.caption(
        "This is a simplified, educational stock/sector scenario engine — not investment advice."
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

# ---------- Risk & Market Scenarios ----------
with tab_risk:
    st.subheader("Risk & Market Scenario Engine (Stock Drivers)")

    st.markdown("#### 1) Sector Factor (Stock Driver) Profile")
    if st.session_state.get("scenario_meta"):
        shocked_sector_current = st.session_state["scenario_meta"].get("Shocked Sector", sectors[0])
    else:
        shocked_sector_current = sectors[0]

    chosen_sector_for_factors = st.selectbox(
        "View factor profile for sector",
        sectors,
        index=sectors.index(shocked_sector_current) if shocked_sector_current in sectors else 0,
    )
    sector_factor_df = compute_factor_exposures_for_sector(chosen_sector_for_factors)
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        st.dataframe(sector_factor_df, use_container_width=True)
    with col_f2:
        factor_chart = (
            alt.Chart(sector_factor_df)
            .mark_bar()
            .encode(
                x=alt.X("Factor:N", sort=None),
                y=alt.Y("Exposure:Q"),
                tooltip=["Factor", "Exposure"],
            )
            .properties(height=260)
        )
        st.altair_chart(factor_chart, use_container_width=True)

    st.markdown("#### 2) Portfolio Factor Exposures (if portfolio uploaded)")
    portfolio_df_current = st.session_state.get("portfolio_df")
    if portfolio_df_current is not None:
        try:
            exposures_df = compute_factor_exposures_for_portfolio(portfolio_df_current)
            col_pf1, col_pf2 = st.columns([1, 2])
            with col_pf1:
                st.dataframe(exposures_df, use_container_width=True)
            with col_pf2:
                exp_chart = (
                    alt.Chart(exposures_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Factor:N", sort=None),
                        y=alt.Y("Exposure:Q"),
                        tooltip=["Factor", "Exposure"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(exp_chart, use_container_width=True)
        except Exception as e:
            st.error(f"Could not compute factor exposures: {e}")
            exposures_df = None
    else:
        st.info("Upload a portfolio on the 'Portfolio Analyzer' tab to see portfolio factor exposures.")
        exposures_df = None

    st.markdown("#### 3) Market Scenario Builder (stock-focused)")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        rates_shock = st.slider(
            "Rates shock (bps, + = higher yields, - = lower)",
            -200,
            200,
            0,
            step=25,
        )
        growth_tilt = st.slider(
            "Style tilt (toward growth vs defensives)",
            -5,
            5,
            0,
            help="-5 = defensive/low beta, +5 = high-growth/high-beta tilt",
        )
    with col_m2:
        oil_shock = st.slider(
            "Oil price shock (%)",
            -30,
            30,
            0,
        )
        risk_aversion = st.slider(
            "Risk sentiment (risk-off ↔ risk-on)",
            -5,
            5,
            0,
            help="-5 = strong risk-off, +5 = strong risk-on",
        )

    # Map to factor shocks (simple heuristic model)
    market_scenario = {
        "Market": risk_aversion / 5.0,
        "Rates": rates_shock / 100.0,
        "Growth Stocks Tilt": growth_tilt / 2.0,
        "Oil": oil_shock / 10.0,
        "FX": 0.0,
    }

    st.markdown("**Market scenario shock vector (model units):**")
    st.write(market_scenario)

    if exposures_df is not None and not exposures_df.empty:
        est_impact = estimate_impact_from_scenario(exposures_df, market_scenario)
        st.metric("Estimated Portfolio Impact (model units)", f"{est_impact:+.3f}")
        if est_impact > 0.5:
            st.success("Scenario suggests the portfolio is positively exposed to this market environment.")
        elif est_impact < -0.5:
            st.warning("Scenario suggests the portfolio is negatively exposed to this market environment.")
        else:
            st.info("Scenario impact on the portfolio appears modest / mixed.")
    else:
        st.caption("No portfolio factor exposures available yet. Upload a portfolio to see scenario impact.")

# ---------- AI Research Analyst ----------
with tab_ai:
    st.subheader("AI Research Analyst — Generate internal stock/sector narratives")
    st.markdown(
        "Draft internal memos, scenario summaries, and risk notes about this sector shock, "
        "its factor/portfolio impact, live watchlist, and news sentiment. (AI must be configured with GROQ_API_KEY.)"
    )

    user_q = st.text_area(
        "Ask the AI Research Analyst (e.g., 'Write an executive summary of this scenario for PMs')",
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
            portfolio_df_current = st.session_state.get("portfolio_df")
            news_articles = st.session_state.get("news_cache", [])

            context = f"Scenario: {scenario_name}\nInputs: {scenario_meta}\n\n"

            if sector_df_current is not None and not sector_df_current.empty:
                context += "Top sector impacts:\n"
                top_n = sector_df_current.sort_values("Impact Score", ascending=False).head(3)
                for _, r in top_n.iterrows():
                    context += f"- {r['Sector']}: {r['Impact Score']} ({r['Impact Label']})\n"

            if stock_df_current is not None and not stock_df_current.empty:
                context += "\nStocks in shocked sector:\n"
                for _, r in stock_df_current.iterrows():
                    context += f"- {r['Stock']}: {r['Impact Score']} ({r['Impact Label']})\n"

            if portfolio_df_current is not None:
                try:
                    exp_df = compute_factor_exposures_for_portfolio(portfolio_df_current)
                    context += "\nPortfolio factor exposures:\n"
                    for _, r in exp_df.iterrows():
                        context += f"- {r['Factor']}: {r['Exposure']:.3f}\n"
                except Exception:
                    pass

            if news_articles:
                context += "\nRecent sector news headlines and sentiment:\n"
                for art in news_articles[:5]:
                    title = art.get("title", "")
                    desc = art.get("description", "")
                    sent = simple_sentiment((title or "") + " " + (desc or ""))
                    context += f"- {title} (Sentiment: {sent})\n"

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
    "Katta MacroSuite — decision-support analytics. Not investment advice. For internal trading desk / educational use."
)
