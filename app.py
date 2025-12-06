# katta_macro_suite.py
import os
import io
import streamlit as st
import pandas as pd
import altair as alt
from groq import Groq
from fpdf import FPDF

# ---------------------------
# Config / Client
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-70b-versatile"

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

st.set_page_config(
    page_title="Katta MacroSuite – Markets Intelligence",
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
macro_variables = [
    "Interest Rates",
    "Inflation",
    "GDP Growth",
    "Unemployment",
    "Oil Prices",
    "Geopolitical Tension",
]

sectors = [
    "Tech",
    "Real Estate",
    "Luxury / Discretionary",
    "Bonds",
    "Energy",
    "Consumer Staples",
    "Banks",
]

preset_scenarios = {
    "Rate Hike (Tightening Cycle)": {
        "description": (
            "Central banks raise interest rates to cool activity. Rate-sensitive, long-duration"
            " assets may be pressured while financials can benefit from wider margins."
        ),
        "macros": {
            "Interest Rates": 4,
            "Inflation": 1,
            "GDP Growth": -1,
            "Unemployment": 1,
            "Oil Prices": 0,
            "Geopolitical Tension": 0,
        },
    },
    "High Inflation Environment": {
        "description": (
            "Persistent inflation drives pricing pressure. Real-asset sectors and commodity"
            " sensitive sectors may outpace others."
        ),
        "macros": {
            "Interest Rates": 2,
            "Inflation": 4,
            "GDP Growth": -1,
            "Unemployment": 1,
            "Oil Prices": 1,
            "Geopolitical Tension": 0,
        },
    },
    "Recession / Slowdown": {
        "description": (
            "Growth contraction with rising unemployment. Defensive sectors and credit-sensitive"
            " exposures are important to monitor."
        ),
        "macros": {
            "Interest Rates": -1,
            "Inflation": -1,
            "GDP Growth": -4,
            "Unemployment": 3,
            "Oil Prices": -1,
            "Geopolitical Tension": 1,
        },
    },
    "Oil & Geopolitics Shock": {
        "description": (
            "Supply shock raises oil and commodity prices; energy benefits but many sectors face"
            " cost pressure."
        ),
        "macros": {
            "Interest Rates": 1,
            "Inflation": 3,
            "GDP Growth": -2,
            "Unemployment": 1,
            "Oil Prices": 4,
            "Geopolitical Tension": 2,
        },
    },
}

weights = {
    "Tech": {
        "Interest Rates": -1.6,
        "Inflation": -0.5,
        "GDP Growth": 1.4,
        "Unemployment": -1.0,
        "Oil Prices": -0.4,
        "Geopolitical Tension": -0.6,
    },
    "Real Estate": {
        "Interest Rates": -1.8,
        "Inflation": -0.8,
        "GDP Growth": 0.8,
        "Unemployment": -1.0,
        "Oil Prices": -0.3,
        "Geopolitical Tension": -0.5,
    },
    "Luxury / Discretionary": {
        "Interest Rates": -1.2,
        "Inflation": -1.0,
        "GDP Growth": 1.5,
        "Unemployment": -1.6,
        "Oil Prices": -0.7,
        "Geopolitical Tension": -0.6,
    },
    "Bonds": {
        "Interest Rates": -1.3,
        "Inflation": -1.0,
        "GDP Growth": -0.3,
        "Unemployment": 0.6,
        "Oil Prices": -0.4,
        "Geopolitical Tension": 0.2,
    },
    "Energy": {
        "Interest Rates": -0.3,
        "Inflation": 0.6,
        "GDP Growth": 0.6,
        "Unemployment": -0.4,
        "Oil Prices": 1.8,
        "Geopolitical Tension": 1.0,
    },
    "Consumer Staples": {
        "Interest Rates": 0.1,
        "Inflation": 0.4,
        "GDP Growth": 0.4,
        "Unemployment": 0.8,
        "Oil Prices": -0.2,
        "Geopolitical Tension": 0.2,
    },
    "Banks": {
        "Interest Rates": 1.8,
        "Inflation": 0.6,
        "GDP Growth": 0.7,
        "Unemployment": -0.5,
        "Oil Prices": 0.1,
        "Geopolitical Tension": -0.2,
    },
}

# ---------------------------
# Helpers
# ---------------------------
def compute_sector_scores(macro_values: dict) -> pd.DataFrame:
    rows = []
    for sector in sectors:
        score = 0.0
        for macro in macro_variables:
            score += weights[sector][macro] * macro_values.get(macro, 0)
        rows.append({"Sector": sector, "Raw Score": score})

    df = pd.DataFrame(rows)
    max_abs = df["Raw Score"].abs().max()
    # normalize to +/-5 scale (keep zero-handling)
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
    # normalize allocation
    if "Allocation" not in df.columns:
        raise ValueError("Uploaded portfolio must contain 'Allocation' column")
    total = df["Allocation"].sum()
    if total == 0:
        df["Weight"] = 0.0
    else:
        df["Weight"] = df["Allocation"] / total

    merged = df.merge(sector_scores, left_on="Sector", right_on="Sector", how="left")
    merged["Impact Score"].fillna(0.0, inplace=True)
    merged["Weighted Impact"] = merged["Weight"] * merged["Impact Score"]
    portfolio_score = merged["Weighted Impact"].sum()
    return portfolio_score, merged[["Sector", "Allocation", "Weight", "Impact Score", "Weighted Impact"]]


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
        "Your task: convert macro inputs and scenario outputs into concise internal research narratives, "
        "scenario summaries, risk notes and suggested topics for internal follow-up. "
        "You MUST NOT provide direct buy/sell recommendations, price targets or personalized investment advice. "
        + level_line
    )

    messages = [{"role": "system", "content": system_prompt}]
    # include limited history
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


def create_pdf_report(title: str, scenario_name: str, macro_vals: dict, sector_df: pd.DataFrame, portfolio_table: pd.DataFrame = None, ai_summary: str = ""):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="L")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Scenario: {scenario_name}", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Macro Inputs:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for k, v in macro_vals.items():
        pdf.cell(0, 7, f"{k}: {v}", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Sector Impacts:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for _, r in sector_df.iterrows():
        pdf.cell(0, 7, f"{r['Sector']}: {r['Impact Score']} ({r['Impact Label']})", ln=True)
    pdf.ln(4)
    if portfolio_table is not None and not portfolio_table.empty:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Portfolio Exposure (sample):", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for _, r in portfolio_table.iterrows():
            pdf.cell(0, 6, f"{r['Sector']}: Allocation {r['Allocation']}, WeightedImpact {r['Weighted Impact']:.3f}", ln=True)
        pdf.ln(4)
    if ai_summary:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "AI Research Summary:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        # wrap lines simply
        for line in ai_summary.split("\n"):
            pdf.multi_cell(0, 6, line)
    # return bytes
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
              <div style="font-size:12px;color:#475569;">Scenario analysis • Sector sensitivity • Portfolio exposure • Internal research automation</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_header()

with st.sidebar:
    st.title("Katta MacroSuite")
    ai_level = st.radio("AI output style:", ["Professional", "Executive", "Technical"], index=0)
    st.markdown("---")
    if client is None:
        st.warning("Groq API not configured — AI Research Analyst disabled until GROQ_API_KEY is set.")
    st.markdown("Upload a CSV portfolio (columns: Sector, Allocation). Allocation can be percent or units.")
    st.markdown("---")
    st.caption("This platform is a decision-support tool. It does NOT provide buy/sell advice.")

# Main tabs
tab_explorer, tab_portfolio, tab_ai, tab_reports = st.tabs(
    ["Scenario Explorer", "Portfolio Analyzer", "AI Research Analyst", "Generate Report"]
)

# ---------- Explorer Tab ----------
with tab_explorer:
    st.subheader("Scenario Explorer — Macro → Sector Sensitivity")
    mode = st.radio("Mode", ["Preset Scenarios", "Custom"], horizontal=True)
    if mode == "Preset Scenarios":
        scenario_name = st.selectbox("Choose scenario:", list(preset_scenarios.keys()))
        scenario = preset_scenarios[scenario_name]
        st.markdown(f"**{scenario_name}** — {scenario['description']}")
        macro_values = scenario["macros"].copy()
    else:
        st.markdown("Build a custom macro scenario (range -5 to +5)")
        macro_values = {}
        c1, c2 = st.columns(2)
        with c1:
            macro_values["Interest Rates"] = st.slider("Interest Rates", -5, 5, 0)
            macro_values["GDP Growth"] = st.slider("GDP Growth", -5, 5, 0)
            macro_values["Oil Prices"] = st.slider("Oil Prices", -5, 5, 0)
        with c2:
            macro_values["Inflation"] = st.slider("Inflation", -5, 5, 0)
            macro_values["Unemployment"] = st.slider("Unemployment", -5, 5, 0)
            macro_values["Geopolitical Tension"] = st.slider("Geopolitical Tension", -5, 5, 0)
        scenario_name = "Custom"

    sector_df = compute_sector_scores(macro_values)
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("#### Sector Impact Overview")
        st.dataframe(sector_df.style.format({"Impact Score": "{:+.2f}"}), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("#### Visual Overview")
        chart = alt.Chart(sector_df).mark_bar().encode(
            x=alt.X("Sector:N", sort=None),
            y=alt.Y("Impact Score:Q"),
            tooltip=["Sector", "Impact Score", "Impact Label"]
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Quick take")
    sorted_df = sector_df.sort_values("Impact Score", ascending=False)
    winners = sorted_df.head(2)
    losers = sorted_df.tail(2)
    winner_text = ", ".join(f"{row.Sector} ({row['Impact Label']})" for _, row in winners.iterrows())
    loser_text = ", ".join(f"{row.Sector} ({row['Impact Label']})" for _, row in losers.iterrows())
    st.markdown(f"- **Likely relative winners:** {winner_text}  \n- **Facing headwinds:** {loser_text}")
    st.caption("Model is a simplified decision-support tool — not investment advice.")

# ---------- Portfolio Analyzer ----------
with tab_portfolio:
    st.subheader("Portfolio / Revenue Exposure Analyzer")
    st.markdown("Upload a CSV with columns: `Sector`, `Allocation` (percent or units).")
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])
    sample = st.button("Download sample CSV")
    if sample:
        sample_df = pd.DataFrame({"Sector": sectors, "Allocation": [0]*len(sectors)})
        csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download sample CSV", csv_bytes, file_name="portfolio_sample.csv")

    portfolio_df = None
    if uploaded is not None:
        try:
            portfolio_df = pd.read_csv(uploaded)
            st.write("Uploaded portfolio preview:")
            st.dataframe(portfolio_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    if st.button("Analyze current scenario exposure"):
        # require sector_df from last tab: recompute if needed
        if 'sector_df' not in locals():
            sector_df = compute_sector_scores(macro_values)
        if portfolio_df is None:
            st.error("Please upload a portfolio CSV first.")
        else:
            try:
                score, breakdown = compute_portfolio_exposure(portfolio_df, sector_df)
                st.metric("Portfolio Weighted Impact Score", f"{score:.3f}")
                st.markdown("Breakdown:")
                st.dataframe(breakdown, use_container_width=True)

                # simple interpretation
                if score > 1.5:
                    st.success("Portfolio tilt: Mild to strong positive sensitivity to the scenario.")
                elif score < -1.5:
                    st.warning("Portfolio tilt: Mild to strong negative sensitivity to the scenario.")
                else:
                    st.info("Portfolio tilt: Largely neutral under the chosen scenario.")
            except Exception as e:
                st.error(f"Analysis error: {e}")

# ---------- AI Research Analyst ----------
with tab_ai:
    st.subheader("AI Research Analyst — Generate internal narratives")
    st.markdown("Draft internal memos, scenario summaries, and risk notes. (AI must be configured.)")

    if "ai_history" not in st.session_state:
        st.session_state.ai_history = []

    user_q = st.text_area("Ask the AI Research Analyst (e.g., 'Write an executive summary of this scenario')", height=120)
    ai_style = st.selectbox("AI style", ["Professional", "Executive", "Technical"], index=0)
    if st.button("Run AI"):
        if client is None:
            st.error("AI not configured. Set GROQ_API_KEY.")
        else:
            # include basic context: scenario + top sector impacts
            context = f"Scenario: {scenario_name}\nMacro inputs: {macro_values}\nTop sector impacts:\n"
            top_n = sector_df.sort_values("Impact Score", ascending=False).head(3)
            for _, r in top_n.iterrows():
                context += f"- {r['Sector']}: {r['Impact Score']} ({r['Impact Label']})\n"
            prompt = context + "\nUser request:\n" + user_q
            st.session_state.ai_history.append({"role": "user", "content": prompt})
            with st.spinner("Generating AI summary..."):
                out = call_ai_research(st.session_state.ai_history, prompt, ai_style)
            st.markdown("**AI output**")
            st.markdown(out)
            st.session_state.ai_history.append({"role": "assistant", "content": out})
            # prune
            st.session_state.ai_history = st.session_state.ai_history[-20:]

# ---------- Report Generation ----------
with tab_reports:
    st.subheader("Generate Downloadable Report")
    report_title = st.text_input("Report title", "Macro & Portfolio Insight")
    include_portfolio = st.checkbox("Include uploaded portfolio exposure", value=True)
    ai_summary_for_report = st.text_area("Optional: paste AI summary to include in report", height=120)
    if st.button("Create PDF Report"):
        # ensure sector_df exists
        sector_df = compute_sector_scores(macro_values)
        portfolio_table = None
        if include_portfolio and uploaded is not None:
            try:
                _, portfolio_table = compute_portfolio_exposure(pd.read_csv(uploaded), sector_df)
            except Exception as e:
                st.error(f"Cannot include portfolio: {e}")
                portfolio_table = None
        pdf_bytes = create_pdf_report(report_title, scenario_name, macro_values, sector_df, portfolio_table, ai_summary_for_report)
        st.download_button("Download PDF report", data=pdf_bytes, file_name="katta_report.pdf", mime="application/pdf")

# Footer / notes
st.markdown("---")
st.caption("Katta MacroSuite — decision-support analytics. Not investment advice. For internal corporate use.")
