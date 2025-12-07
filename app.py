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

# ---------------------------
# Helpers
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


def compute_portfolio_exposure(portfolio_df: pd.DataFrame, sector_scores: pd.DataFrame):
    """
    Expects portfolio_df with columns: Sector, Allocation (as percent or fraction)
    Returns weighted exposure and breakdown.
    """
    df = portfolio_df.copy()

    if "Sector" not in df.columns or "Allocation" not in df.columns:
        raise ValueError("Portfolio CSV must contain 'Sector' and 'Allocation' columns")

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


def compute_diversification_metrics(portfolio_df: pd.DataFrame):
    """
    Simple diversification measure using 1 - HHI (Herfindahl-Hirschman Index).
    """
    df = portfolio_df.copy()
    if "Allocation" not in df.columns:
        return 0.0, "Undefined (no allocation)"
    total = df["Allocation"].sum()
    if total == 0:
        return 0.0, "Undefined (no allocation)"

    weights = df["Allocation"] / total
    hhi = float((weights ** 2).sum())
    diversification = 1.0 - hhi

    if diversification >= 0.8:
        label = "Very well diversified"
    elif diversification >= 0.6:
        label = "Well diversified"
    elif diversification >= 0.4:
        label = "Moderately diversified"
    else:
        label = "Concentrated risk"

    return diversification, label


def call_ai_research(history, user_text: str, level: str = "Professional") -> str:
    """Call Groq Llama as a corporate research analyst (concepts, narratives)."""
    if client is None:
        return "AI Research Analyst not configured. Set GROQ_API_KEY."

    level_line = {
        "Professional": "Write as a corporate research analyst for internal use, concise and structured.",
        "Executive": "Write a polished executive summary for C-suite consumption, ~3 paragraphs.",
        "Technical": "Write a detailed technical memo with supporting reasoning and caveats.",
    }.get(level, "Write in clear professional language.")

    system_prompt = (
        "You are an internal AI research analyst for an institutional client. "
        "Your task: convert scenario inputs, sector outputs and portfolio tilts "
        "into concise internal research narratives, scenario summaries, risk notes, "
        "and suggested topics for internal follow-up. "
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
              <div style="font-size:12px;color:#475569;">Sector scenarios • Stock impact simulation • Portfolio analytics • Scenario library • Internal research automation</div>
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
            "Groq API not configured — AI Research Analyst will show a placeholder until GROQ_API_KEY is set."
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
if "scenario_library" not in st.session_state:
    st.session_state["scenario_library"] = []
if "sector_stocks" not in st.session_state:
    # deep copy defaults
    st.session_state["sector_stocks"] = {
        k: v[:] for k, v in DEFAULT_SECTOR_STOCKS.items()
    }

# ---------------------------
# Main tabs
# ---------------------------
tab_explorer, tab_portfolio, tab_whatif, tab_scenarios, tab_ai, tab_reports = st.tabs(
    [
        "Sector & Stock Simulator",
        "Portfolio Analyzer",
        "What-If Builder",
        "Scenario Library",
        "AI Research Analyst",
        "Generate Report",
    ]
)

# ---------- Sector & Stock Simulator (NEW first feature) ----------
with tab_explorer:
    st.subheader("Sector Scenario & Stock Impact Simulator")

    st.markdown(
        "Step 1: Define how each sector is currently impacted (–5 = strong negative, +5 = strong positive)."
    )

    sector_scores = {}
    c1, c2 = st.columns(2)
    for i, sector in enumerate(sectors):
        col = c1 if i % 2 == 0 else c2
        with col:
            val = st.slider(
                f"{sector}",
                -5.0,
                5.0,
                0.0,
                0.5,
            )
        sector_scores[sector] = float(val)

    # Build sector_df from sliders
    sector_rows = []
    for s in sectors:
        score = sector_scores[s]
        sector_rows.append(
            {
                "Sector": s,
                "Impact Score": round(score, 2),
                "Impact Label": label_from_score(score),
            }
        )
    sector_df = pd.DataFrame(sector_rows)

    # Scenario metadata
    scenario_name = "Custom sector scenario"
    scenario_meta = {s: sector_scores[s] for s in sectors}

    st.session_state["sector_df"] = sector_df
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

        avg_severity = float(
            sum(abs(v) for v in sector_scores.values()) / max(len(sector_scores), 1)
        )
        if avg_severity >= 3.5:
            shock_label = "High regime shift"
        elif avg_severity >= 2.0:
            shock_label = "Moderate dislocation"
        elif avg_severity >= 1.0:
            shock_label = "Low but meaningful move"
        else:
            shock_label = "Very mild environment"
        st.markdown(f"**Scenario severity (avg |score|):** {avg_severity:.2f} — {shock_label}")

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

    st.markdown("---")
    st.markdown("### Step 2: Map sectors to stocks (editable)")

    st.caption(
        "Edit which stocks belong to each sector. Use comma-separated tickers. "
        "The model will simulate which stocks are most affected based on the sector scores above."
    )

    sector_stocks = st.session_state["sector_stocks"]
    for sector in sectors:
        default_text = ", ".join(sector_stocks.get(sector, []))
        text_val = st.text_input(
            f"{sector} stocks",
            default_text,
            key=f"stocks_{sector}",
        )
        tickers = [t.strip().upper() for t in text_val.split(",") if t.strip()]
        sector_stocks[sector] = tickers

    st.session_state["sector_stocks"] = sector_stocks

    # Build stock-level impact table
    stock_rows = []
    for sector in sectors:
        score = sector_scores[sector]
        label = label_from_score(score)
        for ticker in sector_stocks.get(sector, []):
            stock_rows.append(
                {
                    "Sector": sector,
                    "Stock": ticker,
                    "Impact Score": round(score, 2),
                    "Impact Label": label,
                }
            )

    stock_df = pd.DataFrame(stock_rows) if stock_rows else pd.DataFrame(
        columns=["Sector", "Stock", "Impact Score", "Impact Label"]
    )

    st.markdown("### Step 3: Simulated stock-level impact")

    col3, col4 = st.columns([2, 3])
    with col3:
        st.markdown("#### Stock Impact Table")
        st.dataframe(stock_df, use_container_width=True)

    with col4:
        if not stock_df.empty:
            st.markdown("#### Stock Impact Chart")
            stock_chart = (
                alt.Chart(stock_df)
                .mark_bar()
                .encode(
                    x=alt.X("Stock:N", sort=None),
                    y=alt.Y("Impact Score:Q"),
                    color="Sector:N",
                    tooltip=["Stock", "Sector", "Impact Score", "Impact Label"],
                )
                .properties(height=320)
            )
            st.altair_chart(stock_chart, use_container_width=True)
        else:
            st.info("No stocks defined yet. Add tickers above to see stock-level impact.")

    st.markdown("#### Quick take")
    if not stock_df.empty:
        sorted_stocks = stock_df.sort_values("Impact Score", ascending=False)
        top_winners = sorted_stocks.head(3)
        top_losers = sorted_stocks.tail(3)
        winners_text = ", ".join(
            f"{row.Stock} ({row.Sector}, {row['Impact Label']})"
            for _, row in top_winners.iterrows()
        )
        losers_text = ", ".join(
            f"{row.Stock} ({row.Sector}, {row['Impact Label']})"
            for _, row in top_losers.iterrows()
        )
        st.markdown(
            f"- **Stocks with the most positive simulated impact:** {winners_text or 'N/A'}  \n"
            f"- **Stocks with the most negative simulated impact:** {losers_text or 'N/A'}"
        )
    else:
        st.caption("Define some stocks above to see which names are most impacted.")

    st.caption(
        "This is a simplified scenario engine that maps sector-level stress/benefit to stocks. "
        "It is for internal / educational use, not investment advice."
    )

    # Save scenario (sectors + stocks) to library
    if st.button("Save this sector & stock scenario to library"):
        st.session_state["scenario_library"].append(
            {
                "name": scenario_name,
                "meta": scenario_meta.copy(),
                "sector_df": sector_df.copy(),
                "stock_df": stock_df.copy(),
            }
        )
        st.success("Scenario saved to library for later comparison and analysis.")

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

            # Diversification metric
            diversification, div_label = compute_diversification_metrics(portfolio_df)
            st.markdown(
                f"**Diversification score (1−HHI):** {diversification:.3f} — {div_label}"
            )

            # Allocation chart
            alloc_chart = (
                alt.Chart(portfolio_df)
                .mark_bar()
                .encode(
                    x=alt.X("Sector:N", sort=None),
                    y=alt.Y("Allocation:Q"),
                    tooltip=["Sector", "Allocation"],
                )
                .properties(height=250, title="Sector Allocation")
            )
            st.altair_chart(alloc_chart, use_container_width=True)

        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    if st.button("Analyze current sector scenario exposure"):
        sector_df_current = st.session_state.get("sector_df")
        portfolio_df_current = st.session_state.get("portfolio_df")

        if sector_df_current is None:
            st.error(
                "No scenario found. Please configure sector scores on the 'Sector & Stock Simulator' tab first."
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

# ---------- What-If Builder ----------
with tab_whatif:
    st.subheader("What-If Portfolio Builder")
    st.markdown(
        "Use your uploaded portfolio and adjust sector allocations with multipliers "
        "to see how your exposure changes — no external data needed."
    )

    sector_df_current = st.session_state.get("sector_df")
    portfolio_df_current = st.session_state.get("portfolio_df")

    if sector_df_current is None:
        st.warning(
            "No scenario found. Configure sector scores first on the 'Sector & Stock Simulator' tab."
        )
    elif portfolio_df_current is None:
        st.warning(
            "No portfolio found. Upload a portfolio on the 'Portfolio Analyzer' tab first."
        )
    else:
        st.markdown("#### Adjust sector weights (multiplier 0.0x to 2.0x)")

        multipliers = {}
        for _, row in portfolio_df_current.iterrows():
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

        new_df = portfolio_df_current.copy()
        new_df["Allocation"] = new_df.apply(
            lambda r: r["Allocation"] * multipliers.get(r["Sector"], 1.0),
            axis=1,
        )

        st.markdown("#### What-if portfolio (after multipliers)")
        st.dataframe(new_df, use_container_width=True)

        base_score, _ = compute_portfolio_exposure(
            portfolio_df_current, sector_df_current
        )
        new_score, new_breakdown = compute_portfolio_exposure(
            new_df, sector_df_current
        )
        delta = new_score - base_score

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Original Portfolio Impact Score", f"{base_score:.3f}")
        with col_b:
            st.metric(
                "What-if Portfolio Impact Score",
                f"{new_score:.3f}",
                delta=f"{delta:+.3f}",
            )

        st.markdown("#### What-if breakdown:")
        st.dataframe(new_breakdown, use_container_width=True)

# ---------- Scenario Library ----------
with tab_scenarios:
    st.subheader("Scenario Library & Comparison")

    lib = st.session_state.get("scenario_library", [])
    if not lib:
        st.info("No saved scenarios yet. Save one from the 'Sector & Stock Simulator' tab.")
    else:
        st.markdown("#### Saved scenarios")
        for i, sc in enumerate(lib):
            with st.expander(f"{i+1}. {sc['name']}"):
                st.write("Inputs:", sc["meta"])
                st.markdown("**Sector view:**")
                st.dataframe(sc["sector_df"], use_container_width=True)
                stock_df_sc = sc.get("stock_df")
                if stock_df_sc is not None and not stock_df_sc.empty:
                    st.markdown("**Stock view:**")
                    st.dataframe(stock_df_sc, use_container_width=True)

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
        "Draft internal memos, scenario summaries, and risk notes. "
        "(AI must be configured with GROQ_API_KEY in the environment.)"
    )

    user_q = st.text_area(
        "Ask the AI Research Analyst (e.g., 'Write an executive summary of this sector scenario and the most impacted stocks')",
        height=120,
    )
    ai_style_choice = st.selectbox(
        "AI style", ["Professional", "Executive", "Technical"], index=0
    )

    if st.button("Run AI"):
        sector_df_current = st.session_state.get("sector_df")
        scenario_name = st.session_state.get("scenario_name", "Current scenario")
        scenario_meta = st.session_state.get("scenario_meta", {})
        portfolio_df_current = st.session_state.get("portfolio_df")

        if sector_df_current is None:
            st.error(
                "No scenario found. Please configure a sector scenario on the 'Sector & Stock Simulator' tab first."
            )
        else:
            diversification_txt = ""
            if portfolio_df_current is not None:
                div, div_label = compute_diversification_metrics(portfolio_df_current)
                diversification_txt = f"Diversification score: {div:.3f} ({div_label})."

            context = (
                f"Scenario: {scenario_name}\nScenario inputs: {scenario_meta}\n"
                f"{diversification_txt}\nTop sector impacts:\n"
            )
            top_n = sector_df_current.sort_values(
                "Impact Score", ascending=False
            ).head(3)
            for _, r in top_n.iterrows():
                context += (
                    f"- {r['Sector']}: {r['Impact Score']} ({r['Impact Label']})\n"
                )

            prompt = context + "\nUser request:\n" + user_q
            st.session_state["ai_history"].append({"role": "user", "content": prompt})

            out = call_ai_research(
                st.session_state["ai_history"], prompt, ai_style_choice
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
        scenario_name = st.session_state.get("scenario_name", "Current scenario")
        scenario_meta = st.session_state.get("scenario_meta", {})
        portfolio_df_current = st.session_state.get("portfolio_df")

        if sector_df_current is None:
            st.error(
                "No scenario found. Please configure a sector scenario on the 'Sector & Stock Simulator' tab first."
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
    "Katta MacroSuite — decision-support analytics. Not investment advice. For internal / educational use."
)
