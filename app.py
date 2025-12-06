import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from openai import OpenAI

# ------------------------------------------------------------
# OpenAI client (expects OPENAI_API_KEY in env / Streamlit secrets)
# ------------------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Katta Fintech – Sector & Stock Explorer",
    layout="wide",
)

# ------------------------------------------------------------
# Katta Fintech header / logo
# ------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.25rem;">
      <div style="
          background:#1D4ED8;
          color:white;
          border-radius:12px;
          padding:0.4rem 0.8rem;
          font-weight:700;
          font-size:1.1rem;
          letter-spacing:0.06em;
      ">
        KF
      </div>
      <div>
        <div style="font-size:1.3rem; font-weight:700;">Katta Fintech</div>
        <div style="font-size:0.9rem; color:#6B7280;">Sector &amp; Stock Impact Explorer</div>
      </div>
    </div>
    <hr style="margin-top:0.4rem; margin-bottom:0.8rem;">
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Sidebar: theme toggle & mode selection
# ------------------------------------------------------------
st.sidebar.subheader("Display")
dark_mode = st.sidebar.checkbox("🌙 Dark mode", value=False)

if dark_mode:
    bg_color = "#020617"          # very dark navy
    text_color = "#E5E7EB"        # light grey
    card_color = "#0F172A"
    accent_color = "#38BDF8"      # cyan / teal
else:
    bg_color = "#F8FAFF"          # light blue/white
    text_color = "#0F172A"        # dark navy
    card_color = "#FFFFFF"
    accent_color = "#1D4ED8"      # blue

# Apply global background + text color
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .block-container {{
            padding-top: 1.5rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Tabs: 1) Sector / scenario explorer  2) AI chatbot
tab_explorer, tab_chat = st.tabs(["📊 Sector & Scenario Explorer", "🤖 AI Stock & Finance Chatbot"])

# ------------------------------------------------------------
# Shared definitions for the explorer tab
# ------------------------------------------------------------
macro_variables = [
    "Interest Rates",
    "Inflation",
    "GDP Growth",
    "Unemployment",
    "Oil Prices",
    "Geopolitical Tension",
]

# Think of these as **stock sectors / ETFs**:
sectors = [
    "Tech",
    "Real Estate",
    "Luxury / Discretionary",
    "Bonds",
    "Energy",
    "Consumer Staples",
    "Banks",
]

# Pre-set scenarios, but described in stock/markets language
preset_scenarios = {
    "Rate Hike (Tightening Cycle)": {
        "description": (
            "Central banks raise interest rates to cool the economy or fight inflation. "
            "Growth and rate-sensitive stocks (Tech, Real Estate, Luxury) may struggle, "
            "while banks can benefit from higher net interest margins."
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
            "Prices are rising quickly. Central banks may keep hiking, consumers feel pressure, "
            "and sectors with pricing power or real assets may hold up relatively better."
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
            "Economic growth slows sharply or turns negative. Unemployment rises, earnings estimates "
            "get cut, and investors rotate into defensive, lower-volatility sectors."
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
            "Oil prices spike after a supply shock or conflict. Energy stocks may rally, while "
            "transportation, airlines, and some consumer segments face margin pressure."
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

# Simple weights model: macro → sector impact (still intuitive, but framed for stocks)
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
        "Interest Rates": -1.3,  # rising rates hurt existing bonds
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


def compute_sector_scores(macro_values: dict) -> pd.DataFrame:
    """Compute impact score for each sector based on macro sliders and weights."""
    rows = []
    for sector in sectors:
        score = 0.0
        for macro in macro_variables:
            score += weights[sector][macro] * macro_values[macro]
        rows.append({"Sector": sector, "Impact Score": score})
    df = pd.DataFrame(rows)
    # Normalize to roughly -5..+5 for nicer display
    max_abs = df["Impact Score"].abs().max()
    if max_abs > 0:
        df["Impact Score"] = df["Impact Score"] / max_abs * 5
    df["Impact Score"] = df["Impact Score"].round(1)

    def label(score):
        if score <= -3.5:
            return "Strong Negative"
        if score <= -1.5:
            return "Mild Negative"
        if score < 1.5:
            return "Neutral / Mixed"
        if score < 3.5:
            return "Mild Positive"
        return "Strong Positive"

    df["Impact Label"] = df["Impact Score"].apply(label)
    return df


# ------------------------------------------------------------
# TAB 1: Sector & Scenario Explorer
# ------------------------------------------------------------
with tab_explorer:
    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.title("Sector Impact under Different Market Environments")

    with col_side:
        st.markdown(
            """
            **How to use this tab**

            1. Choose a pre-set market environment or build your own.
            2. See which **stock sectors** are helped or hurt.
            3. Use the summary to practice explaining sector rotation.
            """
        )

    mode = st.radio(
        "Choose mode:",
        ["Pre-set Market Environments", "Build Your Own Scenario"],
    )

    if mode == "Pre-set Market Environments":
        scenario_name = st.selectbox("Select a market environment:", list(preset_scenarios.keys()))
        scenario = preset_scenarios[scenario_name]

        st.markdown(f"**Scenario: {scenario_name}**")
        st.markdown(scenario["description"])

        macro_values = scenario["macros"]

    else:
        st.markdown("Use the sliders to build your own **market environment**.")
        macro_values = {}
        c1, c2 = st.columns(2)
        with c1:
            macro_values["Interest Rates"] = st.slider("Interest Rates (higher = tighter)", -5, 5, 0)
            macro_values["GDP Growth"] = st.slider("GDP Growth (trend vs. shock)", -5, 5, 0)
            macro_values["Oil Prices"] = st.slider("Oil Prices", -5, 5, 0)
        with c2:
            macro_values["Inflation"] = st.slider("Inflation", -5, 5, 0)
            macro_values["Unemployment"] = st.slider("Unemployment", -5, 5, 0)
            macro_values["Geopolitical Tension"] = st.slider("Geopolitical Tension", -5, 5, 0)

    with st.expander("What do these variables mean in a stock context?"):
        st.markdown(
            """
            - **Interest Rates** – Higher rates can pressure growth stocks and real estate, but often help banks
              via higher net interest margins.
            - **Inflation** – High inflation can hurt consumers and bonds, but sometimes helps real assets and
              companies with strong pricing power.
            - **GDP Growth** – Strong growth tends to support cyclicals and discretionary sectors. Weak growth or
              contraction favors defensive sectors.
            - **Unemployment** – Rising unemployment usually pressures earnings, especially for cyclical and
              consumer-exposed sectors.
            - **Oil Prices** – High oil prices are usually positive for **Energy** stocks but negative for airlines,
              transportation, and some consumer names.
            - **Geopolitical Tension** – Conflicts and tensions can raise risk premiums, impact global trade, and
              make investors rotate into perceived safe havens.
            """
        )

    df = compute_sector_scores(macro_values)

    st.markdown("### Sector Impact Overview")

    col_table, col_chart = st.columns([2, 3])

    with col_table:
        st.dataframe(
            df.style.format({"Impact Score": "{:+.1f}"}),
            hide_index=True,
            use_container_width=True,
        )

    with col_chart:
        chart = (
            alt.Chart(df)
            .mark_bar(color=accent_color)
            .encode(
                x=alt.X("Sector:N", sort=None),
                y=alt.Y("Impact Score:Q"),
                tooltip=["Sector", "Impact Score", "Impact Label"],
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Summary (Practice Explaining Sector Rotation)")

    sorted_df = df.sort_values("Impact Score", ascending=False)
    winners = sorted_df.head(2)
    losers = sorted_df.tail(2)

    winner_text = ", ".join(f"{row.Sector} ({row['Impact Label']})" for _, row in winners.iterrows())
    loser_text = ", ".join(f"{row.Sector} ({row['Impact Label']})" for _, row in losers.iterrows())

    st.markdown(
        f"""
        - **Likely winners in this environment:** {winner_text}  
        - **Likely under pressure:** {loser_text}
        """
    )

    with st.expander("Model details (for teachers / reviewers)"):
        st.markdown(
            """
            This is a **teaching model**, not a trading system. It uses:

            1. Macro-style inputs (rates, inflation, growth, etc.) as proxies for the market environment.
            2. Intuitive **weights** that describe how each stock sector tends to react to those variables.
            3. A normalized score in the range roughly -5 to +5, translated into labels like
               *Strong Negative* or *Mild Positive*.

            Students can use this to practice:
            - Explaining why a given sector might outperform or underperform.
            - Connecting news headlines (rate hikes, oil shocks, recessions) to sector-level impacts.
            """
        )

# ------------------------------------------------------------
# TAB 2: AI Stock & Finance Chatbot
# ------------------------------------------------------------
with tab_chat:
    st.title("AI Stock & Finance Chatbot (Educational)")

    st.markdown(
        """
        Ask questions about **stocks, sectors, portfolio concepts, or market scenarios**.

        > ℹ️ This chatbot is for **education only** – it explains concepts and scenarios, but it
        does **not** give personalized investment advice or specific buy/sell recommendations.
        """
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about stocks, sectors, ETFs, risk, diversification, etc...")

    def call_ai_chat(history, user_text: str) -> str:
        """Call OpenAI chat API with a stock/finance educational assistant."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a calm, clear stock market and finance tutor. "
                    "You explain concepts about stocks, ETFs, sectors, risk, diversification, "
                    "macro environments, and long-term investing. "
                    "You do NOT give personalized financial advice, do NOT say 'buy' or 'sell' any "
                    "specific security, and you do NOT claim to know current or future prices. "
                    "Focus on education, frameworks, and examples."
                ),
            }
        ]
        # Add prior turns
        for m in history[-8:]:
            messages.append({"role": m["role"], "content": m["content"]})
        # Add latest user message
        messages.append({"role": "user", "content": user_text})

        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.4,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Sorry, I ran into an error talking to the AI service: `{e}`"

    if user_input:
        # Show user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get AI reply
        with st.chat_message("assistant"):
            with st.spinner("Thinking about your question..."):
                reply = call_ai_chat(st.session_state.chat_messages, user_input)
                st.markdown(reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
