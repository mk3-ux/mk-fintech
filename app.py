import os
import requests
import streamlit as st
import pandas as pd
import altair as alt
from fpdf import FPDF

# Groq client is optional – app still runs without it
try:
    from groq import Groq
except ImportError:
    Groq = None

# ---------------------------
# API KEYS (HARD-CODED HERE)
# ---------------------------
# 🔴 IMPORTANT: replace these three with your real keys:
GROQ_API_KEY = "Ygsk_18WeDlIgcHuC3C4FcyQnWGdyb3FYrF2m2CUVLYghyvJtTFcFlLRq"
MASSIVE_API_KEY = "Q1pmrfqv0vV6caqxpJUjwcyEsSEvvSJU"   # (Polygon/Massive key)
NEWSAPI_KEY = "4f0f0589094c414a8ef178ee05c9226d"

MODEL_NAME = "llama-3.1-8b-instant"
client = (
    Groq(api_key=GROQ_API_KEY)
    if (Groq and GROQ_API_KEY and "YOUR_GROQ_API_KEY" not in GROQ_API_KEY)
    else None
)

# Massive (Polygon) base URL
MASSIVE_BASE_URL = "https://api.polygon.io"
NEWSAPI_URL = "https://newsapi.org/v2/everything"

st.set_page_config(
    page_title="Katta Finsight – Stock Hedge Fund Lab",
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
# Static model definitions
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

FACTORS = [
    "Market Move",
    "Sector Move",
    "Earnings Surprise",
    "News Sentiment",
]

# Toy hedge-fund style factor loadings by sector
SECTOR_FACTOR_WEIGHTS = {
    "Tech": {
        "Market Move": 1.2,
        "Sector Move": 1.0,
        "Earnings Surprise": 0.9,
        "News Sentiment": 0.8,
    },
    "Real Estate": {
        "Market Move": 0.8,
        "Sector Move": 0.9,
        "Earnings Surprise": 0.5,
        "News Sentiment": 0.4,
    },
    "Luxury / Discretionary": {
        "Market Move": 1.1,
        "Sector Move": 0.9,
        "Earnings Surprise": 1.0,
        "News Sentiment": 0.9,
    },
    "Bonds": {
        "Market Move": -0.6,
        "Sector Move": -0.3,
        "Earnings Surprise": 0.1,
        "News Sentiment": 0.2,
    },
    "Energy": {
        "Market Move": 0.7,
        "Sector Move": 1.1,
        "Earnings Surprise": 0.6,
        "News Sentiment": 0.7,
    },
    "Consumer Staples": {
        "Market Move": 0.5,
        "Sector Move": 0.4,
        "Earnings Surprise": 0.4,
        "News Sentiment": 0.3,
    },
    "Banks": {
        "Market Move": 1.0,
        "Sector Move": 0.7,
        "Earnings Surprise": 0.8,
        "News Sentiment": 0.6,
    },
}

# For simpler single-factor effects
SECTOR_BASE_SENSITIVITY = {
    "Tech": 1.0,
    "Real Estate": 0.6,
    "Luxury / Discretionary": 0.8,
    "Bonds": -0.4,
    "Energy": 0.5,
    "Consumer Staples": 0.3,
    "Banks": 0.7,
}

# ---------------------------
# External API helpers (Massive + NewsAPI)
# ---------------------------

def fetch_massive_snapshot(ticker: str):
    """Get live snapshot from Massive (Polygon) for a single ticker."""
    if not MASSIVE_API_KEY or "YOUR_MASSIVE_API_KEY" in MASSIVE_API_KEY:
        return None, "Massive API key is not configured. Edit app.py and set MASSIVE_API_KEY."

    url = f"{MASSIVE_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
    params = {"apiKey": MASSIVE_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        if resp.status_code != 200:
            return None, f"Massive error: HTTP {resp.status_code} {data.get('error', '')}"
        ticker_data = data.get("ticker")
        if not ticker_data:
            return None, "No ticker data returned from Massive."
        last_trade = ticker_data.get("lastTrade") or {}
        prev_day = ticker_data.get("prevDay") or {}
        todays_change_pct = ticker_data.get("todaysChangePerc")
        price = last_trade.get("p") or prev_day.get("c")
        return {
            "last_price": price,
            "todays_change_pct": todays_change_pct,
            "raw": ticker_data,
        }, None
    except Exception as e:
        return None, f"Massive request failed: {e}"


def fetch_news_for_ticker(ticker: str, page_size: int = 5):
    """Get latest headlines for a ticker from NewsAPI."""
    if not NEWSAPI_KEY or "YOUR_NEWSAPI_KEY" in NEWSAPI_KEY:
        return [], "NewsAPI key is not configured. Edit app.py and set NEWSAPI_KEY."

    params = {
        "q": ticker,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY,
    }
    try:
        resp = requests.get(NEWSAPI_URL, params=params, timeout=5)
        data = resp.json()
        if resp.status_code != 200:
            return [], f"NewsAPI error: HTTP {resp.status_code} {data.get('message', '')}"
        return data.get("articles", []), None
    except Exception as e:
        return [], f"NewsAPI request failed: {e}"

# ---------------------------
# Local helpers: sentiment, model, portfolio
# ---------------------------

def simple_news_sentiment_from_text(text: str):
    """Tiny keyword-based sentiment using headline/title text."""
    text = (text or "").lower()
    if not text.strip():
        return 0.0, "Neutral"

    pos_words = [
        "beat", "beats", "growth", "rises", "surge", "record",
        "strong", "bullish", "upgrade", "optimistic", "profit",
        "profits", "gain", "gains", "improve", "improves",
    ]
    neg_words = [
        "miss", "misses", "loss", "losses", "falls", "fall",
        "plunge", "plunges", "downgrade", "weak", "bearish",
        "lawsuit", "fraud", "scandal", "warning", "cuts", "cut",
        "recession", "slowdown",
    ]

    pos = 0
    neg = 0
    for w in pos_words:
        pos += text.count(w)
    for w in neg_words:
        neg += text.count(w)

    total = max(1, pos + neg)
    raw = (pos - neg) / total  # roughly -1..+1
    score = max(-5.0, min(5.0, raw * 5.0))

    if score >= 1.5:
        label = "Positive"
    elif score <= -1.5:
        label = "Negative"
    else:
        label = "Neutral"

    return score, label


def compute_sector_impacts_from_factors(factor_shocks: dict):
    """Multi-factor model: factors drive sector impact scores."""
    rows = []
    for sec in SECTORS:
        weights = SECTOR_FACTOR_WEIGHTS.get(sec, {})
        raw = 0.0
        for f in FACTORS:
            raw += weights.get(f, 0.0) * factor_shocks.get(f, 0.0)
        rows.append({"Sector": sec, "Raw Score": raw})

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
    """Portfolio stress test under current sector scenario."""
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


def compute_portfolio_factor_exposures(portfolio_df: pd.DataFrame):
    """Hedge-fund style: factor exposures from sector weights."""
    df = portfolio_df.copy()
    if "Allocation" not in df.columns:
        raise ValueError("Uploaded portfolio must contain 'Allocation' column")

    total = df["Allocation"].sum()
    if total == 0:
        return pd.DataFrame({"Factor": FACTORS, "Exposure": [0.0] * len(FACTORS)})

    df["Weight"] = df["Allocation"] / total

    exposures = {f: 0.0 for f in FACTORS}
    for _, row in df.iterrows():
        sec = row["Sector"]
        w = row["Weight"]
        sec_weights = SECTOR_FACTOR_WEIGHTS.get(sec, {})
        for f in FACTORS:
            exposures[f] += w * sec_weights.get(f, 0.0)

    rows = [{"Factor": f, "Exposure": round(exposures[f], 3)} for f in FACTORS]
    return pd.DataFrame(rows)

# ---------------------------
# AI + PDF helpers (Groq)
# ---------------------------

def call_ai_research(history, user_text: str, level: str = "Professional") -> str:
    if client is None:
        return "AI Research Analyst not configured. Set GROQ_API_KEY in app.py if you want AI text."

    level_line = {
        "Professional": "Write as a hedge-fund style stock analyst, concise and structured.",
        "Executive": "Write a polished PM/IC summary (~3 paragraphs) with key risks and drivers.",
        "Technical": "Write a detailed risk memo with factor language, scenarios, and caveats.",
    }.get(level, "Write in clear professional language.")

    system_prompt = (
        "You are an internal AI research analyst for a hedge fund. "
        "You focus on single-stock analysis, factor drivers, sector linkages and positioning. "
        "You may reference market moves, sector trends, earnings, and news sentiment, "
        "but you MUST NOT give explicit buy/sell/hold recommendations, price targets, "
        "or personalized investment advice. "
        "Frame everything as scenario analysis, risk notes, and talking points. "
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
            max_tokens=900,
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
# UI header
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
              <div style="font-size:18px;font-weight:800;color:{COLORS['text']};">Katta MacroSuite — Stock Hedge Fund Lab</div>
              <div style="font-size:12px;color:#475569;">
                Massive snapshot • Factor shocks • Portfolio risk • News + AI research • PDF export
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_header()

# ---------------------------
# Session state defaults
# ---------------------------

defaults = {
    "sector_df": None,
    "scenario_name": None,
    "scenario_meta": {},
    "portfolio_df": None,
    "ai_history": [],
    "factor_shocks": {f: 0.0 for f in FACTORS},
    "news_sentiment_score": 0.0,
    "news_sentiment_label": "Neutral",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
        st.info(
            "AI tab will still show up, but you need a valid GROQ_API_KEY in app.py if you want AI text."
        )

    st.markdown(
        "Upload a CSV portfolio (columns: `Sector`, `Allocation`). "
        "Allocation can be percent or units."
    )

    st.markdown("---")
    st.caption("Decision-support only. Not investment advice.")

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
    st.subheader("Live Stock Dashboard — Massive + NewsAPI + Factor Model")

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

    # Live snapshot from Massive
    st.markdown("### Live Snapshot (from Massive, if key is set)")

    snapshot_data, snapshot_error = (None, None)
    if ticker:
        snapshot_data, snapshot_error = fetch_massive_snapshot(ticker)

    col_price1, col_price2, col_price3 = st.columns(3)
    if snapshot_error:
        st.warning(snapshot_error)

    if snapshot_data:
        last_price = snapshot_data.get("last_price") or 0.0
        todays_change_pct = snapshot_data.get("todays_change_pct") or 0.0
        with col_price1:
            st.metric("Last Price (Massive)", f"${last_price:,.2f}")
        with col_price2:
            st.metric("Today's Change %", f"{todays_change_pct:+.2f}%")
        with col_price3:
            if todays_change_pct:
                prev_price = last_price / (1 + todays_change_pct / 100.0)
            else:
                prev_price = last_price
            st.metric("Prev Close (implied)", f"${prev_price:,.2f}")
    else:
        with col_price1:
            last_price = st.number_input(
                "Last Price (manual)", min_value=0.0, value=180.0
            )
        with col_price2:
            todays_change_pct = st.slider(
                "Today's % move (manual)",
                -20.0,
                20.0,
                0.0,
                step=0.5,
            )
        with col_price3:
            new_price = last_price * (1 + todays_change_pct / 100.0)
            st.metric(
                "Implied End-of-day Price",
                f"${new_price:,.2f}",
                f"{todays_change_pct:+.1f}%",
            )

    # News + sentiment
    st.markdown("### Latest News Headlines (NewsAPI)")

    articles, news_error = fetch_news_for_ticker(ticker) if ticker else ([], None)
    news_text_blob = ""
    if news_error:
        st.warning(news_error)
    elif articles:
        for art in articles:
            title = art.get("title", "No title")
            source = (art.get("source") or {}).get("name", "")
            url = art.get("url", "")
            desc = art.get("description", "") or ""
            news_text_blob += f"{title}. {desc}\n"
            st.markdown(
                f"**{title}**  \n<small>{source}</small>",
                unsafe_allow_html=True,
            )
            if url:
                st.markdown(f"[Read more]({url})")
            st.markdown("---")

    sentiment_score, sentiment_label = simple_news_sentiment_from_text(news_text_blob)
    st.session_state["news_sentiment_score"] = sentiment_score
    st.session_state["news_sentiment_label"] = sentiment_label
    st.markdown(
        f"**News sentiment (from headlines, toy):** {sentiment_label}  (score {sentiment_score:.1f}/5)"
    )

    # Multi-factor scenario (stock-focused)
    st.markdown("### Multi-factor Stock Scenario")
    factor_shocks = st.session_state["factor_shocks"].copy()

    cF1, cF2 = st.columns(2)
    with cF1:
        factor_shocks["Market Move"] = st.slider(
            "Market Move (index)  [-5 bearish, +5 bullish]",
            -5.0,
            5.0,
            factor_shocks.get("Market Move", 0.0),
            step=0.5,
        )
        factor_shocks["Earnings Surprise"] = st.slider(
            "Earnings Surprise [-5 big miss, +5 big beat]",
            -5.0,
            5.0,
            factor_shocks.get("Earnings Surprise", 0.0),
            step=0.5,
        )
    with cF2:
        factor_shocks["Sector Move"] = st.slider(
            f"{primary_sector} Sector Trend [-5 weak, +5 strong]",
            -5.0,
            5.0,
            factor_shocks.get("Sector Move", 0.0),
            step=0.5,
        )
        factor_shocks["News Sentiment"] = st.slider(
            "News Sentiment factor",
            -5.0,
            5.0,
            float(round(sentiment_score, 1)),
            step=0.5,
            help=f"Starts from headline sentiment ({sentiment_label}), but you can override.",
        )

    st.session_state["factor_shocks"] = factor_shocks

    # Turn factor shocks into a toy "implied stock move"
    weights_for_move = {
        "Market Move": 0.4,
        "Sector Move": 0.3,
        "Earnings Surprise": 0.2,
        "News Sentiment": 0.1,
    }
    effective_stock_move = (
        factor_shocks["Market Move"] * weights_for_move["Market Move"]
        + factor_shocks["Sector Move"] * weights_for_move["Sector Move"]
        + factor_shocks["Earnings Surprise"] * weights_for_move["Earnings Surprise"]
        + factor_shocks["News Sentiment"] * weights_for_move["News Sentiment"]
    ) * 2.0

    st.markdown(
        f"**Implied stock move from factors (toy): ~{effective_stock_move:+.1f}%**"
    )

    sector_df = compute_sector_impacts_from_factors(factor_shocks)
    scenario_name = f"{ticker} factor scenario"
    scenario_meta = {
        "Stock": ticker,
        "Primary Sector": primary_sector,
        "Implied Move (%)": round(effective_stock_move, 1),
        "Factors": factor_shocks,
        "Headline Sentiment": f"{sentiment_label} ({sentiment_score:.1f}/5)",
    }

    st.session_state["sector_df"] = sector_df
    st.session_state["scenario_name"] = scenario_name
    st.session_state["scenario_meta"] = scenario_meta

    # Sector visuals
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

    st.markdown("#### Quick Take — Sectors Under This Stock Scenario")
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
        f"- **Sectors most helped:** {winner_text}  \n"
        f"- **Sectors facing headwinds:** {loser_text}"
    )
    st.caption("Toy factor model for learning only, not a live risk engine.")

# ---------- Portfolio Analyzer ----------
with tab_portfolio:
    st.subheader("Portfolio Analyzer — Factor Exposures & Scenario Stress Test")

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

    if st.button("Run portfolio analysis (current scenario)"):
        sector_df_current = st.session_state.get("sector_df")
        portfolio_df_current = st.session_state.get("portfolio_df")

        if sector_df_current is None:
            st.error(
                "No scenario found. Build a scenario on the 'Live Stock Dashboard' tab first."
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
                        "Portfolio tilt: Mild to strong positive sensitivity in this scenario."
                    )
                elif score < -1.5:
                    st.warning(
                        "Portfolio tilt: Mild to strong negative sensitivity in this scenario."
                    )
                else:
                    st.info("Portfolio tilt: Largely neutral under this scenario.")
            except Exception as e:
                st.error(f"Analysis error: {e}")

    st.markdown("---")
    if st.button("Show hedge-fund style factor exposures"):
        portfolio_df_current = st.session_state.get("portfolio_df")
        if portfolio_df_current is None:
            st.error("Upload a portfolio CSV first.")
        else:
            try:
                factor_exposures = compute_portfolio_factor_exposures(
                    portfolio_df_current
                )
                st.markdown("### Portfolio Factor Exposures (toy)")
                st.dataframe(
                    factor_exposures, use_container_width=True, hide_index=True
                )

                chart_f = (
                    alt.Chart(factor_exposures)
                    .mark_bar()
                    .encode(
                        x=alt.X("Factor:N", sort=None),
                        y=alt.Y("Exposure:Q"),
                        tooltip=["Factor", "Exposure"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart_f, use_container_width=True)
            except Exception as e:
                st.error(f"Exposure calc error: {e}")

# ---------- AI Research Analyst ----------
with tab_ai:
    st.subheader("AI Research Analyst — Stock-Focused Narratives (Groq)")

    st.markdown(
        "Write internal hedge-fund style notes, risk summaries, and scenario explanations. "
        "Requires a valid GROQ_API_KEY set in app.py."
    )

    user_q = st.text_area(
        "Ask the AI Research Analyst (e.g., 'Write a PM summary of this scenario for the IC call')",
        height=140,
    )
    ai_style = st.selectbox(
        "AI style", ["Professional", "Executive", "Technical"], index=0
    )

    if st.button("Run AI"):
        if client is None:
            st.error(
                "AI not configured. Edit app.py and set GROQ_API_KEY to use this."
            )
        else:
            sector_df_current = st.session_state.get("sector_df")
            scenario_name = st.session_state.get("scenario_name", "Current scenario")
            scenario_meta = st.session_state.get("scenario_meta", {})

            context = f"Scenario: {scenario_name}\nScenario inputs: {scenario_meta}\n"
            if sector_df_current is not None:
                context += "Top sector impacts:\n"
                top_n = sector_df_current.sort_values(
                    "Impact Score", ascending=False
                ).head(4)
                for _, r in top_n.iterrows():
                    context += (
                        f"- {r['Sector']}: {r['Impact Score']} ({r['Impact Label']})\n"
                    )

            full_prompt = context + "\nUser request:\n" + user_q
            st.session_state["ai_history"].append(
                {"role": "user", "content": full_prompt}
            )

            with st.spinner("Generating AI summary..."):
                out = call_ai_research(
                    st.session_state["ai_history"], full_prompt, ai_style
                )

            st.markdown("**AI output**")
            st.markdown(out)
            st.session_state["ai_history"].append(
                {"role": "assistant", "content": out}
            )
            st.session_state["ai_history"] = st.session_state["ai_history"][-20:]

# ---------- Report Generation ----------
with tab_reports:
    st.subheader("Generate Downloadable Scenario Report (PDF)")

    report_title = st.text_input(
        "Report title", "Stock Scenario & Portfolio Insight"
    )
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
                "No scenario found. Build a scenario on the 'Live Stock Dashboard' tab first."
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
                file_name="katta_stock_report.pdf",
                mime="application/pdf",
            )

st.markdown("---")
st.caption(
    "Katta MacroSuite — hedge-fund style stock factor lab. For education and internal research only. Not investment advice."
)
