# app.py or katta_macro_suite.py

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
def compute_stock_sector_impacts(stock_move: float, primary_sector: str) -> pd.DataFrame:
    """
    Simple stock→sector sensitivity model.

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
              <div style="font-size:12px;color:#475569;">Single-stock impact • Sector sensitivity • Portfolio exposure • Internal research automation</div>
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

    st.caption(
        "This platform is a decision-support tool. "
        "It does NOT provide buy/sell advice."
    )

# Init session state
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

# Main tabs
tab_explorer, tab_portfolio, tab_ai, tab_reports = st.tabs(
    [
        "Stock Impact Explorer",
        "Portfolio Analyzer",
        "AI Research Analyst",
        "Generate Report",
    ]
)

# ---------- Stock Impact Explorer ----------
with tab_explorer:
    st.subheader("Single-Stock Impact Explorer — Stock → Sector Sensitivity")

    stock_name = st.text_input("Stock name / ticker", "AAPL")
    primary_sector = st.selectbox(
        "Primary sector for this stock",
        sectors,
        index=0,
        help="Which sector best represents this stock?",
    )
    stock_move = st.slider(
        "Assumed stock price move (%)",
        -20,
        20,
        0,
        help="Negative = stock down, Positive = stock up",
    )

    st.caption(
        "This page uses a simple, illustrative sensitivity model: the chosen stock has the "
        "strongest impact on its primary sector, and a smaller spillover impact on other sectors."
    )

    sector_df = compute_stock_sector_impacts(stock_move, primary_sector)

    scenario_name = f"{stock_name} move {stock_move:+.1f}%"
    scenario_meta = {
        "Stock": stock_name,
        "Move (%)": stock_move,
        "Primary Sector": primary_sector,
    }

    # Save in session for other tabs
    st.session_state["sector_df"] = sector_df
    st.session_state["scenario_name"] = scenario_name
    st.session_state["scenario_meta"] = scenario_meta

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("#### Sector Impact from this Stock Move")
        st.dataframe(
            sector_df.style.format({"Impact Score": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
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
        f"- **Most positively exposed to this stock move:** {winner_text}  \n"
        f"- **Most negatively exposed / least helped:** {loser_text}"
    )
    st.caption(
        "This is a simplified, educational sensitivity model — not a real-world risk model or investment advice."
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

    if st.button("Analyze current stock scenario exposure"):
        sector_df_current = st.session_state.get("sector_df")
        portfolio_df_current = st.session_state.get("portfolio_df")

        if sector_df_current is None:
            st.error(
                "No scenario found. Please configure a stock move on the 'Stock Impact Explorer' tab first."
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

# ---------- AI Research Analyst ----------
with tab_ai:
    st.subheader("AI Research Analyst — Generate internal narratives")
    st.markdown(
        "Draft internal memos, scenario summaries, and risk notes. (AI must be configured with GROQ_API_KEY.)"
    )

    user_q = st.text_area(
        "Ask the AI Research Analyst (e.g., 'Write an executive summary of this stock scenario')",
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
                "No scenario found. Please configure a stock move on the 'Stock Impact Explorer' tab first."
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
    "Katta MacroSuite — decision-support analytics. Not investment advice. For internal corporate use."
)
