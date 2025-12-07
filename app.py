import os
import streamlit as st
import pandas as pd
import altair as alt
from groq import Groq
from fpdf import FPDF
import requests

# optional market data dependency
try:
    import yfinance as yf
except ImportError:
    yf = None

# ---------------------------
# Config / Client
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

st.set_page_config(
    page_title="Katta Finsight",
    layout="wide",
    page_icon="📊",
)

# Light theme colors (simple)
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

# ---------------------------
# Helpers
# ---------------------------
def get_realtime_stock_data(ticker: str):
    """
    Fetch live-ish stock price info using yfinance.
    Returns dict with keys:
      price, open, previous_close, change, change_pct, currency
    or None / {'error': ...} if unavailable.
    """
    if not ticker:
        return None
    if yf is None:
        return {"error": "yfinance is not installed. Add 'yfinance' to requirements.txt."}
    try:
        t = yf.Ticker(ticker)
        info = getattr(t, "fast_info", None)
        if info:
            last = float(info.get("lastPrice") or info.get("last_price") or 0)
            prev_close = float(info.get("previousClose") or info.get("previous_close") or 0)
        else:
            hist = t.history(period="2d")
            if hist.empty:
                return {"error": "No market data returned for this ticker."}
            last = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last

        if last == 0:
            return {"error": "Invalid last price from data source."}

        change = last - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0
        currency = getattr(info, "currency", None) if info else None
        if not currency:
            currency = "USD"

        return {
            "price": round(last, 2),
            "previous_close": round(prev_close, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "currency": currency,
        }
    except Exception as e:
        return {"error": f"Market data error: {e}"}


def fetch_stock_news(query: str, limit: int = 5):
    """
    Fetch recent news for the stock using NewsAPI.
    Requires NEWS_API_KEY in environment / secrets.
    Returns list of dicts with: title, source, url, published_at, description.
    """
    if not query:
        return []
    if not NEWS_API_KEY:
        return []

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "apiKey": NEWS_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code != 200:
            return []
        data = resp.json()
        articles = data.get("articles", [])[:limit]
        cleaned = []
        for a in articles:
            cleaned.append(
                {
                    "title": a.get("title"),
                    "source": (a.get("source") or {}).get("name"),
                    "url": a.get("url"),
                    "published_at": a.get("publishedAt"),
                    "description": a.get("description"),
                }
            )
        return cleaned
    except Exception:
        return []


def compute_stock_sector_impacts(stock_move: float, primary_sector: str) -> pd.DataFrame:
    """
    Simple stock→sector sensitivity model, with a bit of hedge-fund-style framing.

    - Primary sector gets sensitivity 1.0
    - All other sectors get spillover sensitivity 0.4
    - Impact Score normalized to +/-5 and labeled.
    """
    rows = []
    for sec in sectors:
        if sec == primary_sector:
            sensitivity = 1.0
        else:
            sensitivity = 0.4  # simple spillover assumption

        raw_score = sensitivity * stock_move
        rows.append(
            {
                "Sector": sec,
                "Sensitivity (to stock move)": sensitivity,
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

    # Light hedge-fund style risk buckets for flavor
    bucket_map = {
        "Tech": "Cyclical / Growth",
        "Luxury / Discretionary": "Cyclical / Consumer",
        "Energy": "Commodity / Cyclical",
        "Banks": "Rate / Financials",
        "Real Estate": "Rate / Real Assets",
        "Consumer Staples": "Defensive",
        "Bonds": "Rates / Duration",
    }
    df["HF Risk Bucket"] = df["Sector"].map(bucket_map).fillna("General Equity")

    return df[["Sector", "HF Risk Bucket", "Impact Score", "Impact Label"]]


def compute_portfolio_exposure(portfolio_df: pd.DataFrame, sector_scores: pd.DataFrame):
    """
    Expects portfolio_df with columns: Sector, Allocation (as percent or fraction)
    Returns weighted exposure and breakdown.
    """
    df = portfolio_df.copy()
    if "Allocation" not in df.columns:
        raise ValueError("Uploaded portfolio must contain 'Allocation' column")
    total = df["Allocation"].sum()
    if total == 0:
        df["Weight"] = 0.0
    else:
        df["Weight"] = df["Allocation"] / total

    merged = df.merge(sector_scores[["Sector", "Impact Score"]], on="Sector", how="left")
    merged["Impact Score"].fillna(0.0, inplace=True)
    merged["Weighted Impact"] = merged["Weight"] * merged["Impact Score"]
    portfolio_score = merged["Weighted Impact"].sum()
    return portfolio_score, merged[
        ["Sector", "Allocation", "Weight", "Impact Score", "Weighted Impact"]
    ]


def call_ai_research(history, user_text: str, level: str = "Professional") -> str:
    """Call Groq Llama as a corporate research analyst (concepts, narratives)."""
    if client is None:
        return (
            "AI Research Analyst not configured. Set GROQ_API_KEY in environment or Streamlit Secrets."
        )
    level_line = {
        "Professional": "Write as a corporate research analyst for internal use, concise and structured.",
        "Executive": "Write a polished executive summary for C-suite consumption, ~3 paragraphs.",
        "Technical": "Write a detailed technical memo with supporting reasoning and caveats.",
    }.get(level, "Write in clear professional language.")

    system_prompt = (
        "You are an internal AI research analyst for an institutional client. "
        "Your task: convert scenario inputs and sector outputs into concise internal research narratives, "
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

    # Scenario inputs
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Scenario Inputs:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for k, v in scenario_meta.items():
        pdf.cell(0, 7, f"{k}: {v}", ln=True)
    pdf.ln(4)

    # Sector impacts
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
              <div style="font-size:18px;font-weight:800;color:{COLORS['text']};">Katta MacroSuite — Markets Intelligence</div>
              <div style="font-size:12px;color:#475569;">Single-stock impact • Sector sensitivity • Portfolio exposure • Internal research automation • Live prices & headlines</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_header()

with st.sidebar:
    st.title("Katta MacroSuite")
    ai_level = st.radio(
        "AI output style:", ["Professional", "Executive", "Technical"], index=0
    )
    st.markdown("---")
    if client is None:
        st.warning(
            "Groq API not configured — AI Research Analyst disabled until GROQ_API_KEY is set."
        )
    if not NEWS_API_KEY:
        st.info("Optional: set NEWS_API_KEY to enable live stock headlines.")
    if yf is None:
        st.info("Optional: add 'yfinance' to requirements.txt to enable live prices.")
    st.markdown(
        "Upload a CSV portfolio (columns: Sector, Allocation). Allocation can be percent or units."
    )
    st.markdown("---")
    st.caption(
        "This platform is a decision-support tool. It does NOT provide buy/sell advice."
    )

# Init scenario library
if "scenario_library" not in st.session_state:
    st.session_state["scenario_library"] = []

# Main tabs
tab_explorer, tab_portfolio, tab_whatif, tab_scenarios, tab_ai, tab_reports = st.tabs(
    [
        "Stock Impact Explorer",
        "Portfolio Analyzer",
        "What-if Builder",
        "Scenario Library",
        "AI Research Analyst",
        "Generate Report",
    ]
)

# ---------- Stock Impact Explorer (First Page) ----------
with tab_explorer:
    st.subheader("Single-Stock Impact Explorer — Stock → Sector Sensitivity")

    col_left, col_right = st.columns([2, 2])

    with col_left:
        raw_stock_input = st.text_input(
            "Stock ticker / name", "AAPL", help="Use an exchange ticker like AAPL, MSFT, JPM, etc."
        )
        ticker = raw_stock_input.strip().upper()

        primary_sector = st.selectbox(
            "Primary sector for this stock",
            sectors,
            index=0,
            help="Which sector best represents this stock?",
        )

        move_mode = st.radio(
            "Stock move source",
            ["Manual % move", "Use live market move"],
            index=0,
            help="Hedge-fund style: you can either hard-code a stress move or link it to the live market move.",
        )

        manual_stock_move = st.slider(
            "Assumed stock price move (%) (manual)",
            -20,
            20,
            0,
            help="Negative = stock down, Positive = stock up",
        )

        live_data = None
        stock_move = manual_stock_move

        if move_mode == "Use live market move":
            live_data = get_realtime_stock_data(ticker)
            if live_data and not live_data.get("error"):
                stock_move = live_data.get("change_pct", 0.0) or 0.0
                st.caption(
                    f"Using live % move from previous close: {stock_move:+.2f}%"
                )
            else:
                stock_move = manual_stock_move
                if live_data and live_data.get("error"):
                    st.warning(
                        f"Live data not available for {ticker}: {live_data.get('error')}. Falling back to manual slider."
                    )

        # save latest live data for other tabs if needed
        st.session_state["live_stock_data"] = live_data

    with col_right:
        st.markdown("#### Live market snapshot")
        live_data = st.session_state.get("live_stock_data")
        if live_data and not live_data.get("error"):
            price = live_data["price"]
            change = live_data["change"]
            change_pct = live_data["change_pct"]
            currency = live_data["currency"]
            delta_str = f"{change:+.2f} ({change_pct:+.2f}%)"
            st.metric(
                label=f"{ticker} price ({currency})",
                value=f"{price:.2f}",
                delta=delta_str,
            )
        elif live_data and live_data.get("error"):
            st.error(live_data.get("error"))
        else:
            if yf is None:
                st.caption("Install yfinance to see live price snapshot.")
            else:
                st.caption("Select 'Use live market move' to fetch a live snapshot.")

        st.markdown("#### Latest headlines")
        if NEWS_API_KEY:
            news_items = fetch_stock_news(ticker or raw_stock_input, limit=5)
            if not news_items:
                st.caption("No recent headlines found for this name (or API limit reached).")
            else:
                for n in news_items:
                    title = n.get("title") or "No title"
                    src = n.get("source") or "Unknown"
                    url = n.get("url") or "#"
                    st.markdown(
                        f"- [{title}]({url})  \n"
                        f"  <span style='font-size:11px;color:#6b7280;'>Source: {src}</span>",
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("Set NEWS_API_KEY to pull live headlines for this ticker.")

    st.caption(
        "This page uses a simplified, educational sensitivity model. Live prices & news make it feel like a hedge-fund blotter, "
        "but it's still not a real risk model or investment advice."
    )

    sector_df = compute_stock_sector_impacts(stock_move, primary_sector)

    scenario_name = f"{ticker or raw_stock_input} move {stock_move:+.1f}%"
    scenario_meta = {
        "Stock": ticker or raw_stock_input,
        "Move (%)": stock_move,
        "Primary Sector": primary_sector,
        "Move source": "Live market" if move_mode == "Use live market move" else "Manual slider",
    }
    st.session_state["sector_df"] = sector_df
    st.session_state["scenario_name"] = scenario_name
    st.session_state["scenario_meta"] = scenario_meta

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("#### Sector impact from this stock move")
        st.dataframe(
            sector_df.style.format({"Impact Score": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.markdown("#### Visual overview (impact score)")
        chart = (
            alt.Chart(sector_df)
            .mark_bar()
            .encode(
                x=alt.X("Sector:N", sort=None),
                y=alt.Y("Impact Score:Q"),
                color=alt.Color("HF Risk Bucket:N"),
                tooltip=["Sector", "HF Risk Bucket", "Impact Score", "Impact Label"],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Hedge-fund style quick take")
    sorted_df = sector_df.sort_values("Impact Score", ascending=False)
    winners = sorted_df.head(2)
    losers = sorted_df.tail(2)
    winner_text = ", ".join(
        f"{row.Sector} ({row['Impact Label']}, {row['HF Risk Bucket']})"
        for _, row in winners.iterrows()
    )
    loser_text = ", ".join(
        f"{row.Sector} ({row['Impact Label']}, {row['HF Risk Bucket']})"
        for _, row in losers.iterrows()
    )

    direction = "bullish" if stock_move > 0 else "bearish" if stock_move < 0 else "flat"
    st.markdown(
        f"- **Direction of shock:** {direction} single-name move of **{stock_move:+.2f}%**.  \n"
        f"- **Crowded long-style beneficiaries:** {winner_text}  \n"
        f"- **Natural hedge / short candidates (conceptually, not advice):** {loser_text}  \n"
        f"- **Decomposition (rule-of-thumb):** Treat ~70% of this move as sector/market factor and ~30% as idiosyncratic noise."
    )
    st.caption(
        "Language above mimics hedge-fund risk commentary, but this app does not generate trade ideas or recommendations."
    )

    # Save scenario to library
    if st.button("Save this scenario to library"):
        st.session_state["scenario_library"].append(
            {
                "name": scenario_name,
                "meta": scenario_meta.copy(),
                "sector_df": sector_df.copy(),
            }
        )
        st.success("Scenario saved to library for later comparison.")

# ---------- Portfolio Analyzer ----------
with tab_portfolio:
    st.subheader("Portfolio / Revenue Exposure Analyzer")
    st.markdown("Upload a CSV with columns: `Sector`, `Allocation` (percent or units).")

    uploaded_portfolio = st.file_uploader(
        "Upload portfolio CSV", type=["csv"], key="upload_portfolio_base"
    )
    sample = st.button("Download sample CSV")
    if sample:
        sample_df = pd.DataFrame({"Sector": sectors, "Allocation": [0] * len(sectors)})
        csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download sample CSV", csv_bytes, file_name="portfolio_sample.csv"
        )

    portfolio_df = None
    if uploaded_portfolio is not None:
        try:
            portfolio_df = pd.read_csv(uploaded_portfolio)
            st.write("Uploaded portfolio preview:")
            st.dataframe(portfolio_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    if st.button("Analyze current stock scenario exposure"):
        sector_df_current = st.session_state.get("sector_df")
        if sector_df_current is None:
            st.error(
                "No scenario found. Please configure a stock move on the 'Stock Impact Explorer' tab first."
            )
        elif portfolio_df is None:
            st.error("Please upload a portfolio CSV first.")
        else:
            try:
                score, breakdown = compute_portfolio_exposure(
                    portfolio_df, sector_df_current
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

# ---------- What-if Builder (no paid data, just math) ----------
with tab_whatif:
    st.subheader("What-if Portfolio Builder (No Extra Data)")

    st.markdown(
        "Upload a portfolio, then use sliders to see how **multiplying each sector's allocation** "
        "changes your exposure to the current stock scenario."
    )

    sector_df_current = st.session_state.get("sector_df")
    if sector_df_current is None:
        st.warning(
            "No scenario found. Configure a stock scenario first on the 'Stock Impact Explorer' tab."
        )
    else:
        uploaded_whatif = st.file_uploader(
            "Upload portfolio CSV for what-if (Sector, Allocation)",
            type=["csv"],
            key="upload_portfolio_whatif",
        )

        if uploaded_whatif is not None:
            try:
                base_df = pd.read_csv(uploaded_whatif)
            except Exception as e:
                base_df = None
                st.error(f"Error reading CSV: {e}")

            if base_df is not None:
                st.write("Base portfolio:")
                st.dataframe(base_df, use_container_width=True)

                st.markdown("#### Adjust sector weights (multiplier 0.0x to 2.0x)")
                multipliers = {}
                for _, row in base_df.iterrows():
                    sec = row["Sector"]
                    m = st.slider(
                        f"{sec} multiplier",
                        0.0,
                        2.0,
                        1.0,
                        0.05,
                        help="1.0 = keep allocation same, 2.0 = double, 0.5 = cut in half",
                    )
                    multipliers[sec] = m

                new_df = base_df.copy()
                new_df["Allocation"] = new_df.apply(
                    lambda r: r["Allocation"] * multipliers.get(r["Sector"], 1.0),
                    axis=1,
                )

                st.markdown("#### What-if portfolio (after multipliers)")
                st.dataframe(new_df, use_container_width=True)

                # Compute original vs new exposure
                try:
                    base_score, base_break = compute_portfolio_exposure(
                        base_df, sector_df_current
                    )
                    new_score, new_break = compute_portfolio_exposure(
                        new_df, sector_df_current
                    )
                    delta = new_score - base_score

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "Original Portfolio Impact Score",
                            f"{base_score:.3f}",
                        )
                    with col_b:
                        st.metric(
                            "What-if Portfolio Impact Score",
                            f"{new_score:.3f}",
                            delta=f"{delta:+.3f}",
                        )

                    st.markdown("#### What-if breakdown:")
                    st.dataframe(new_break, use_container_width=True)
                except Exception as e:
                    st.error(f"Error computing what-if exposure: {e}")
        else:
            st.info("Upload a portfolio CSV to start building what-if scenarios.")

# ---------- Scenario Library & Comparison ----------
with tab_scenarios:
    st.subheader("Scenario Library & Comparison")

    lib = st.session_state.get("scenario_library", [])
    if not lib:
        st.info("No saved scenarios yet. Save one from the 'Stock Impact Explorer' tab.")
    else:
        st.markdown("#### Saved scenarios")
        for i, sc in enumerate(lib):
            with st.expander(f"{i+1}. {sc['name']}"):
                st.write("Inputs:", sc["meta"])
                df_sc = sc["sector_df"]
                st.dataframe(df_sc, use_container_width=True)

        st.markdown("#### Compare two scenarios by sector")
        names = [sc["name"] for sc in lib]
        col1, col2 = st.columns(2)
        with col1:
            sel1 = st.selectbox("Scenario A", names, index=0)
        with col2:
            sel2 = st.selectbox("Scenario B", names, index=min(1, len(names) - 1))

        if sel1 and sel2 and sel1 != sel2:
            sc1 = next(sc for sc in lib if sc["name"] == sel1)
            sc2 = next(sc for sc in lib if sc["name"] == sel2)
            df1 = sc1["sector_df"].rename(
                columns={"Impact Score": "Impact A", "Impact Label": "Label A"}
            )
            df2 = sc2["sector_df"].rename(
                columns={"Impact Score": "Impact B", "Impact Label": "Label B"}
            )
            merged = df1.merge(df2, on="Sector", how="outer")
            st.dataframe(merged, use_container_width=True)
        else:
            st.caption("Select two different scenarios to compare.")

# ---------- AI Research Analyst ----------
with tab_ai:
    st.subheader("AI Research Analyst — Generate internal narratives")
    st.markdown(
        "Draft internal memos, scenario summaries, and risk notes. (AI must be configured.)"
    )

    if "ai_history" not in st.session_state:
        st.session_state.ai_history = []

    user_q = st.text_area(
        "Ask the AI Research Analyst (e.g., 'Write an executive summary of this stock scenario')",
        height=120,
    )
    ai_style = st.selectbox(
        "AI style", ["Professional", "Executive", "Technical"], index=0
    )

    if st.button("Run AI"):
        if client is None:
            st.error("AI not configured. Set GROQ_API_KEY.")
        else:
            sector_df_current = st.session_state.get("sector_df")
            scenario_name = st.session_state.get("scenario_name", "Current scenario")
            scenario_meta = st.session_state.get("scenario_meta", {})

            if sector_df_current is None:
                st.error(
                    "No scenario found. Please configure a stock move on the 'Stock Impact Explorer' tab first."
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
                st.session_state.ai_history.append({"role": "user", "content": prompt})
                with st.spinner("Generating AI summary..."):
                    out = call_ai_research(
                        st.session_state.ai_history, prompt, ai_style
                    )
                st.markdown("**AI output**")
                st.markdown(out)
                st.session_state.ai_history.append(
                    {"role": "assistant", "content": out}
                )
                st.session_state.ai_history = st.session_state.ai_history[-20:]

# ---------- Report Generation ----------
with tab_reports:
    st.subheader("Generate Downloadable Report")
    report_title = st.text_input("Report title", "Stock Scenario & Portfolio Insight")
    include_portfolio = st.checkbox(
        "Include uploaded portfolio exposure", value=True
    )
    ai_summary_for_report = st.text_area(
        "Optional: paste AI summary to include in report", height=120
    )

    if st.button("Create PDF Report"):
        sector_df_current = st.session_state.get("sector_df")
        scenario_name = st.session_state.get("scenario_name", "Current scenario")
        scenario_meta = st.session_state.get("scenario_meta", {})

        if sector_df_current is None:
            st.error(
                "No scenario found. Please configure a stock move on the 'Stock Impact Explorer' tab first."
            )
        else:
            portfolio_table = None
            if include_portfolio:
                st.info(
                    "To include portfolio details in the PDF, you can also run this app "
                    "server-side where the portfolio dataframe is kept in session."
                )

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
    "Katta MacroSuite — decision-support analytics. Not investment advice. For internal corporate use."
)
