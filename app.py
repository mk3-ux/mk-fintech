import os
import streamlit as st
import pandas as pd
import altair as alt
from groq import Groq

# ------------------------------------------------------------
# Groq client (expects GROQ_API_KEY in Streamlit Secrets)
# ------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-70b-versatile"  # strong, free Groq model

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Katta Fintech – Macro & Markets Explorer",
    layout="wide",
    page_icon="📈",
)

# --- Light theme styling (clean, centered, no dark mode) ---
COLORS = {
    "bg": "#F8FAFF",
    "text": "#0F172A",
    "card": "#FFFFFF",
    "accent": "#1D4ED8",
    "subtle": "#E5E7EB",
}

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {COLORS['bg']};
            color: {COLORS['text']};
        }}
        .block-container {{
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}
        /* Softer tabs */
        button[data-baseweb="tab"] {{
            border-radius: 999px !important;
            padding: 0.3rem 1.2rem !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Macro data + weights
# ------------------------------------------------------------
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
            "Prices are rising quickly. Consumers feel pressure, and sectors with pricing power "
            "or real assets may hold up relatively better."
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
            "Economic growth slows or turns negative. Unemployment rises, and investors rotate "
            "into defensive, lower-volatility sectors."
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

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def compute_sector_scores(macro_values: dict) -> pd.DataFrame:
    rows = []
    for sector in sectors:
        score = 0.0
        for macro in macro_variables:
            score += weights[sector][macro] * macro_values[macro]
        rows.append({"Sector": sector, "Impact Score": score})

    df = pd.DataFrame(rows)
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


def call_ai_chat(history, user_text: str, level: str) -> str:
    """Call Groq Llama 3.1 as a finance tutor (concepts only, no advice)."""
    if client is None:
        return (
            "The AI tutor is not configured yet. Please add GROQ_API_KEY in Secrets on "
            "Streamlit Cloud and reboot the app."
        )

    level_line = {
        "Beginner": "Explain everything in simple, high-school friendly language.",
        "Intermediate": "Use some finance vocabulary, but stay clear and structured.",
        "Advanced": "You may use more technical finance language and deeper concepts.",
    }.get(level, "Explain in clear, student-friendly language.")

    system_prompt = (
        "You are a calm, clear stock market and finance tutor. "
        "You explain concepts about stocks, ETFs, sectors, risk, diversification, "
        "macro environments, and long-term investing. "
        "You do NOT give personalized financial advice, do NOT say 'buy' or 'sell' "
        "any specific security, and you do NOT claim to know current or future prices. "
        "Focus on education, frameworks, analogies, and examples for students. "
        + level_line
    )

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    for m in history[-8:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_text})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=700,
            temperature=0.4,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I ran into an error talking to the AI service: `{e}`"


def render_header():
    # Gradient pill with logo + title (no "student project" label)
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #E0ECFF, #F5F3FF);
            border-radius: 18px;
            padding: 1.1rem 1.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            border: 1px solid {COLORS['subtle']};
            box-shadow: 0 10px 25px rgba(15,23,42,0.06);
            margin-bottom: 1.3rem;
        ">
          <div style="flex:0 0 auto;">
        """,
        unsafe_allow_html=True,
    )

    # Logo: use file if present, else KF badge
    if os.path.exists("logo.png"):
        st.image("logo.png", width=64)
    else:
        st.markdown(
            f"""
            <div style="
                width:64px;height:64px;border-radius:18px;
                background:{COLORS['accent']};
                display:flex;align-items:center;justify-content:center;
                color:white;font-weight:800;font-size:26px;
            ">
              KF
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
          </div>
          <div style="flex:1 1 auto;">
            <div style="font-size:1.7rem;font-weight:800;color:{COLORS['text']};">
              Katta Fintech – Macro & Markets Explorer
            </div>
            <div style="font-size:0.95rem;color:#4B5563;">
              Explore how different market environments can influence stock sectors,
              and use the built-in AI tutor to practice explaining your reasoning.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.title("Katta Fintech")
    level = st.radio(
        "Explanation level for the AI tutor:",
        ["Beginner", "Intermediate", "Advanced"],
        index=0,
    )
    st.markdown("---")
    st.markdown(
        """
        **What this app demonstrates**

        - Macro → sector intuition  
        - Interactive charts  
        - AI finance tutor (Groq Llama 3.1)  
        - Clean Streamlit UI
        """
    )

# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
render_header()

tab_explorer, tab_chat, tab_learn = st.tabs(
    ["📊 Sector & Scenario Explorer", "🤖 Finance Tutor", "📚 Learning Lab"]
)

# ------------ TAB 1: Explorer ------------
with tab_explorer:
    st.subheader("How different environments might influence stock sectors")

    mode = st.radio(
        "Choose mode:",
        ["Pre-set Market Environments", "Build Your Own Scenario"],
        horizontal=True,
    )

    if mode == "Pre-set Market Environments":
        scenario_name = st.selectbox("Select a market environment:", list(preset_scenarios.keys()))
        scenario = preset_scenarios[scenario_name]
        st.markdown(f"### {scenario_name}")
        st.markdown(scenario["description"])
        macro_values = scenario["macros"]
    else:
        st.markdown("### Build your own environment")
        st.caption("Move the sliders and imagine the kind of headlines you might see in the news.")
        macro_values = {}
        c1, c2 = st.columns(2)
        with c1:
            macro_values["Interest Rates"] = st.slider("Interest Rates (tight → loose)", -5, 5, 0)
            macro_values["GDP Growth"] = st.slider("GDP Growth (weak → strong)", -5, 5, 0)
            macro_values["Oil Prices"] = st.slider("Oil Prices (low → high)", -5, 5, 0)
        with c2:
            macro_values["Inflation"] = st.slider("Inflation (low → high)", -5, 5, 0)
            macro_values["Unemployment"] = st.slider("Unemployment (low → high)", -5, 5, 0)
            macro_values["Geopolitical Tension"] = st.slider("Geopolitical Tension", -5, 5, 0)

    df = compute_sector_scores(macro_values)

    col_table, col_chart = st.columns([2, 3])

    with col_table:
        st.markdown("#### Sector impact overview")
        st.dataframe(
            df.style.format({"Impact Score": "{:+.1f}"}),
            hide_index=True,
            use_container_width=True,
        )

    with col_chart:
        st.markdown("#### Visual impact by sector")
        chart = (
            alt.Chart(df)
            .mark_bar(color=COLORS["accent"])
            .encode(
                x=alt.X("Sector:N", sort=None),
                y=alt.Y("Impact Score:Q"),
                tooltip=["Sector", "Impact Score", "Impact Label"],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Quick verbal summary")
    sorted_df = df.sort_values("Impact Score", ascending=False)
    winners = sorted_df.head(2)
    losers = sorted_df.tail(2)

    winner_text = ", ".join(f"{row.Sector} ({row['Impact Label']})" for _, row in winners.iterrows())
    loser_text = ", ".join(f"{row.Sector} ({row['Impact Label']})" for _, row in losers.iterrows())

    st.markdown(
        f"""
        - **Likely relative winners:** {winner_text}  
        - **Facing more headwinds:** {loser_text}
        """
    )
    st.caption(
        "This is a simplified educational model – not investment advice or a trading system."
    )

# ------------ TAB 2: Finance Tutor ------------
with tab_chat:
    st.subheader("Ask the Finance Tutor (concepts only)")

    st.markdown(
        """
        Ask about **stocks, ETFs, sectors, diversification, risk, or macro environments**.  
        The tutor explains ideas and frameworks, but does **not** tell you what to buy or sell.
        """
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Sample prompts as buttons
    st.markdown("**Try a quick prompt:**")
    c1, c2, c3 = st.columns(3)
    sample_q = None
    if c1.button("📦 What is an ETF?"):
        sample_q = "Explain what an ETF is to a high-school student."
    if c2.button("📉 Rate hikes & growth stocks"):
        sample_q = "Why do rising interest rates often hurt growth stocks like tech?"
    if c3.button("🛡️ Diversification"):
        sample_q = "What does diversification mean in investing, and why does it matter?"

    user_input = st.chat_input("Type your question here...")
    if sample_q and not user_input:
        user_input = sample_q

    # Show chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = call_ai_chat(st.session_state.chat_messages, user_input, level)
                st.markdown(reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})

# ------------ TAB 3: Learning Lab ------------
with tab_learn:
    st.subheader("Learning Lab – How this app works")

    st.markdown(
        """
        This app is meant to help you **connect big-picture news to sector-level market moves**, and
        to practice talking through your reasoning the way an analyst might.
        """
    )

    st.markdown("#### 1. Macro → sector intuition")
    st.markdown(
        """
        - Changes in **interest rates, inflation, growth, unemployment, oil, and geopolitics**
          don't move all stocks the same way.  
        - Investors often talk in terms of **sectors** (Tech, Banks, Energy, etc.).  
        - This app encodes a simple rule-of-thumb model for how each sector *might* react.
        """
    )

    st.markdown("#### 2. Educational model, not a forecast")
    st.markdown(
        """
        - The scores here are **normalized impact scores**, not price targets or return forecasts.  
        - The goal is for you to practice *explaining stories* like:  
          *\"In a rate-hike environment, banks may benefit but growth stocks can be under pressure.\"*  
        - Real markets are messier and sometimes behave differently than textbook logic.
        """
    )

    st.markdown("#### 3. How you might describe this project")
    st.markdown(
        """
        *\"Built an interactive Macro & Markets Explorer web app in Python/Streamlit that models
        how different macroeconomic environments can affect stock market sectors.  
        Integrated a Groq Llama 3.1–powered finance tutor that explains concepts like sectors,
        ETFs, diversification, and rate hikes in student-friendly language.  
        Designed the app to help classmates experiment with scenarios and practice explaining
        their reasoning like a junior portfolio manager.\"*
        """
    )
