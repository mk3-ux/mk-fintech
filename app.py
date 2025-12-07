# app.py — Katta MacroSuite (Sector → Stock Impact Version)

import os
import streamlit as st
import pandas as pd
import altair as alt
from groq import Groq
from fpdf import FPDF

# ---------------------------
# Config / Client
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
    "card": "#FFFFFF",
    "accent": "#0EA5E9",
    "subtle": "#E6EEF8",
}

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}
        .block-container {{ max-width: 1200px; padding-top: 1.2rem; padding-bottom: 2rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Model / Data
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
    Simple sector→sector sensitivity model.

    - Shocked sector gets sensitivity 1.0
    - Other sectors get spillover 0.4
    - Impact Score normalized to +/-5 and labeled.
    """
    rows = []
    for sec in sectors:
        sensitivity = 1.0 if sec == shocked_sector else 0.4
        raw = sensitivity * sector_move
        rows.append({"Sector": sec, "Raw": raw})

    df = pd.DataFrame(rows)
    max_abs = df["Raw"].abs().max()
    if not max_abs:
        max_abs = 1.0
    df["Impact Score"] = (df["Raw"] / max_abs * 5).round(2)
    df["Impact Label"] = df["Impact Score"].apply(label_from_score)
    return df[["Sector", "Impact Score", "Impact Label"]]


def compute_portfolio_exposure(portfolio_df: pd.DataFrame, sector_df: pd.DataFrame):
    df = portfolio_df.copy()
    if "Sector" not in df.columns or "Allocation" not in df.columns:
        raise ValueError("CSV must contain 'Sector' and 'Allocation' columns.")

    total = df["Allocation"].sum()
    if total == 0:
        total = 1.0
    df["Weight"] = df["Allocation"] / total

    merged = df.merge(sector_df, on="Sector", how="left")
    merged["Impact Score"].fillna(0, inplace=True)
    merged["Weighted Impact"] = merged["Weight"] * merged["Impact Score"]
    portfolio_score = merged["Weighted Impact"].sum()
    return portfolio_score, merged[
        ["Sector", "Allocation", "Weight", "Impact Score", "Weighted Impact"]
    ]


def call_ai_research(history, user_text: str, level: str):
    if client is None:
        return "AI not configured. Please set GROQ_API_KEY as an environment variable."

    styles = {
        "Professional": "Write as a corporate analyst, structured and concise.",
        "Executive": "Write a polished 2–3 paragraph C-Suite executive summary.",
        "Technical": "Write a detailed explanation including risks and assumptions.",
    }
    tone = styles.get(level, "Write clearly.")

    system = (
        "You are an internal research analyst. Convert sector scenarios and their impact "
        "on stocks and portfolios into insights. Do NOT make price targets or personalized "
        "investment recommendations. " + tone
    )

    msgs = [{"role": "system", "content": system}]
    msgs.extend(history[-6:])
    msgs.append({"role": "user", "content": user_text})

    try:
        out = client.chat.completions.create(
            model=MODEL_NAME,
            messages=msgs,
            temperature=0.2,
            max_tokens=700,
        )
        return out.choices[0].message.content.strip()
    except Exception as e:
        return f"AI service error: {e}"


def create_pdf_report(
    title,
    scenario_name,
    scenario_meta,
    sector_df,
    portfolio_df=None,
    ai_summary="",
):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=True)

    pdf.set_font("Helvetica", "", 12)
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

    if portfolio_df is not None and not portfolio_df.empty:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Portfolio Exposure:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for _, r in portfolio_df.iterrows():
            pdf.cell(
                0,
                6,
                f"{r['Sector']}: Allocation={r['Allocation']}, "
                f"Weight={r['Weight']:.2f}, Impact={r['Impact Score']:.2f}",
                ln=True,
            )

    if ai_summary:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "AI Research Summary:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for line in ai_summary.split("\n"):
            pdf.multi_cell(0, 6, line)

    return pdf.output(dest="S").encode("latin-1")


# ---------------------------
# Header
# ---------------------------

st.markdown("""
### 📊 **Katta MacroSuite — Sector → Stock Intelligence**
Sector shocks • Stock impact • Portfolio sensitivity • AI narratives • PDF reporting
""")

# ---------------------------
# Session State
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
# Tabs
# ---------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Sector → Stock Explorer",
        "Portfolio Analyzer",
        "AI Research Analyst",
        "Generate Report",
    ]
)

# ================================================================
# TAB 1 — SECTOR → STOCK EXPLORER
# ================================================================

with tab1:
    st.subheader("Sector Shock → Stock Impact Simulator")

    shocked_sector = st.selectbox("Sector to shock", sectors, index=0)
    sector_move = st.slider(
        "Assumed sector movement (%)",
        -20,
        20,
        0,
        help="Negative = sector underperforms, Positive = sector outperforms",
    )

    default_stocks = DEFAULT_SECTOR_STOCKS.get(shocked_sector, [])
    default_str = ", ".join(default_stocks)
    stock_str = st.text_input(
        f"Stocks in {shocked_sector} (comma-separated tickers)",
        value=default_str,
    )
    stocks = [s.strip().upper() for s in stock_str.split(",") if s.strip()]

    st.caption(
        "This simulates how a move in a **sector** flows through to its own stocks, "
        "and also shows spillover to other sectors for portfolio analysis."
    )

    # Sector-level impacts
    sector_df = compute_sector_impacts(sector_move, shocked_sector)

    # Save scenario for other tabs
    st.session_state["sector_df"] = sector_df
    st.session_state["scenario_name"] = f"{shocked_sector} sector move {sector_move:+.1f}%"
    st.session_state["scenario_meta"] = {
        "Shocked Sector": shocked_sector,
        "Sector Move (%)": sector_move,
        "Stocks": ", ".join(stocks),
    }

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("#### Sector Impact Overview")
        st.dataframe(
            sector_df.style.format({"Impact Score": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
    with col2:
        st.markdown("#### Sector Impact Chart")
        chart = (
            alt.Chart(sector_df)
            .mark_bar()
            .encode(
                x=alt.X("Sector:N", sort=None),
                y=alt.Y("Impact Score:Q"),
                tooltip=["Sector", "Impact Score", "Impact Label"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

    # Stock-level impact (sector → stock)
    st.markdown("### Stock Impact from this Sector Shock")

    if stocks:
        row = sector_df[sector_df["Sector"] == shocked_sector].iloc[0]
        stock_score = row["Impact Score"]
        stock_label = row["Impact Label"]

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

        col3, col4 = st.columns([2, 3])
        with col3:
            st.markdown("#### Stocks in Shocked Sector")
            st.dataframe(stock_df, use_container_width=True)
        with col4:
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

        st.markdown("#### Quick Take")
        st.write(
            f"- **Sector shocked:** {shocked_sector} ({stock_label})  
- **All listed stocks in this sector inherit that impact in this simple model.**"
        )
    else:
        st.info("Add at least one stock ticker above to see stock-level impact.")

# ================================================================
# TAB 2 — PORTFOLIO ANALYZER
# ================================================================

with tab2:
    st.subheader("Portfolio / Revenue Exposure Analyzer")
    st.markdown("Upload a CSV with columns: `Sector`, `Allocation` (percent or units).")

    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])

    if uploaded is not None:
        try:
            portfolio_df = pd.read_csv(uploaded)
            st.session_state["portfolio_df"] = portfolio_df
            st.markdown("#### Uploaded Portfolio Preview")
            st.dataframe(portfolio_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    if st.button("Analyze portfolio vs current sector scenario"):
        sector_df_current = st.session_state.get("sector_df")
        portfolio_df_current = st.session_state.get("portfolio_df")

        if sector_df_current is None:
            st.error(
                "No scenario is active yet.\n\nGo to the **Sector → Stock Explorer** tab, "
                "configure a sector shock, then come back here to analyze the portfolio."
            )
        elif portfolio_df_current is None:
            st.error("Please upload a portfolio CSV first.")
        else:
            try:
                score, breakdown = compute_portfolio_exposure(
                    portfolio_df_current, sector_df_current
                )
                st.metric("Portfolio Weighted Impact Score", f"{score:.3f}")
                st.markdown("#### Sector Breakdown vs Scenario")
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

# ================================================================
# TAB 3 — AI RESEARCH ANALYST
# ================================================================

with tab3:
    st.subheader("AI Research Analyst — Scenario Narratives")
    st.markdown(
        "Draft internal-style memos and explanations about how this sector shock and its stock impact "
        "could affect portfolios or business lines. (Requires GROQ_API_KEY.)"
    )

    user_q = st.text_area(
        "Ask the AI Research Analyst (e.g., 'Explain what this sector shock means for these stocks and a diversified portfolio')",
        height=120,
    )
    ai_style = st.selectbox(
        "AI style", ["Professional", "Executive", "Technical"], index=0
    )

    if st.button("Run AI"):
        sector_df_current = st.session_state.get("sector_df")
        scenario_name = st.session_state.get("scenario_name", "Current scenario")
        scenario_meta = st.session_state.get("scenario_meta", {})

        if sector_df_current is None:
            st.error(
                "No scenario is active for the AI to analyze.\n\nFirst, go to the **Sector → Stock Explorer** tab, "
                "set up a sector shock, then run the AI again."
            )
        else:
            context = f"Scenario: {scenario_name}\nScenario inputs: {scenario_meta}\nTop sector impacts:\n"
            top_n = sector_df_current.sort_values("Impact Score", ascending=False).head(3)
            for _, r in top_n.iterrows():
                context += f"- {r['Sector']}: {r['Impact Score']} ({r['Impact Label']})\n"

            prompt = context + "\nUser request:\n" + user_q
            st.session_state["ai_history"].append({"role": "user", "content": prompt})

            with st.spinner("Generating AI summary..."):
                out = call_ai_research(
                    st.session_state["ai_history"], prompt, ai_style
                )

            st.markdown("#### AI Output")
            st.markdown(out)
            st.session_state["ai_history"].append({"role": "assistant", "content": out})
            st.session_state["ai_history"] = st.session_state["ai_history"][-20:]

# ================================================================
# TAB 4 — REPORT GENERATOR
# ================================================================

with tab4:
    st.subheader("Generate Downloadable Report (PDF)")
    report_title = st.text_input("Report title", "Sector Shock & Portfolio Insight")
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
                "No scenario is available for the report.\n\nPlease create a sector scenario "
                "in the **Sector → Stock Explorer** tab before generating a PDF."
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

# ---------------------------
# Footer
# ---------------------------

st.markdown("---")
st.caption(
    "Katta MacroSuite — decision-support analytics. Not investment advice. For internal / educational use."
)
