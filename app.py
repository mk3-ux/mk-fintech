import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import google.generativeai as genai

# ------------------------------------------------------------
# Gemini client (expects GOOGLE_API_KEY in env / Streamlit secrets)
# ------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Katta Fintech – Sector & Stock Explorer",
    layout="wide",
    page_icon="📈",
)


# ------------------------------------------------------------
# Theming helpers
# ------------------------------------------------------------
def get_theme_colors(dark_mode: bool):
    if dark_mode:
        return {
            "bg": "#020617",
            "text": "#E5E7EB",
            "card": "#0F172A",
            "accent": "#38BDF8",
        }
    else:
        return {
            "bg": "#F8FAFF",
            "text": "#0F172A",
            "card": "#FFFFFF",
            "accent": "#1D4ED8",
        }


# ------------------------------------------------------------
# Data for macro → sector model
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

# Pre-set environments the user can choose
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
            "Economic growth slows sharply or turns negative. Unemployment rises, earnings "
            "estimates get cut, and investors rotate into defensive, lower-volatility sectors."
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

# How each macro affects each sector (sign + strength)
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
# Utility functions
# ------------------------------------------------------------
def compute_sector_scores(macro_values: dict) -> pd.DataFrame:
    """Compute impact score for each sector based on macro sliders and weights."""
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
    """Call Gemini with a stock/finance educational assistant."""
    if gemini_model is None:
        return (
            "The AI chatbot is not configured yet. Please add GOOGLE_API_KEY to the "
            "Streamlit app secrets in Streamlit Cloud."
        )

    level_line = {
        "Beginner": "Explain everything in simple, high-school friendly language.",
        "Intermediate": "Use some financial vocabulary, but still be clear and structured.",
        "Advanced": "Feel free to use more technical finance language and deeper concepts.",
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

    convo_lines = [f"System: {system_prompt}"]
    for m in history[-8:]:
        role = "User" if m["role"] == "user" else "Assistant"
        convo_lines.append(f"{role}: {m['content']}")
    convo_lines.append(f"User: {user_text}")
    convo_lines.append("Assistant:")

    prompt = "\n\n".join(convo_lines)

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Sorry, I ran into an error talking to the AI service: `{e}`"


def render_header(colors):
    """Top logo & title bar."""
    col_logo, col_title, col_badge = st.columns([1, 4, 2])

    with col_logo:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=72)
        else:
            st.markdown(
                f"""
                <div style="
                    width:64px;height:64px;border-radius:18px;
                    background:{colors['accent']};
                    display:flex;align-items:center;justify-content:center;
                    color:white;font-weight:800;font-size:26px;
                    box-shadow:0 6px 16px rgba(0,0,0,0.18);
                ">
                  KF
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_title:
        st.markdown(
            f"""
            <div style="padding-top:0.2rem;">
              <div style="font-size:1.8rem;font-weight:800;color:{colors['text']};">
                Macro & Markets Explorer
              </div>
              <div style="font-size:0.95rem;color:#6B7280;">
                A Katta Fintech project – exploring how macro environments can influence stock sectors.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_badge:
        st.markdown(
            """
            <div style="
                margin-top:0.4rem;
                padding:0.5rem 0.75rem;
                border-radius:999px;
                background:rgba(34,197,94,0.08);
                color:#16A34A;
                font-size:0.8rem;
                font-weight:600;
                text-align:right;
            ">
              Student project · Educational only
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='margin:0.6rem 0 0.4rem 0;'>", unsafe_allow_html=True)


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.title("Katta Fintech")

dark_mode = st.sidebar.checkbox("🌙 Dark mode", value=False)
colors = get_theme_colors(dark_mode)

# Global background style
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {colors['bg']};
            color: {colors['text']};
        }}
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

level = st.sidebar.radio("Explanation level", ["Beginner", "Intermediate", "Advanced"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **For applications 👇**

    This app shows:
    - Data visualization
    - Macro & sector reasoning
    - AI integration (Gemini)
    - Clean UI/UX in Streamlit
    """
)

# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
render_header(colors)

tab_explorer, tab_chat, tab_learn = st.tabs(
    ["📊 Sector & Scenario Explorer", "🤖 Finance Chatbot", "📚 Learning Center"]
)


# ------------------------------------------------------------
# TAB 1 – Explorer
# ------------------------------------------------------------
with tab_explorer:
    col_main, col_side = st.columns([3, 1.4])

    with col_main:
        st.subheader("How different environments might influence stock sectors")

    with col_side:
        st.markdown(
            """
            **How to use**

            1. Pick a pre-set environment, or build your own.
            2. See the **impact score** for each sector.
            3. Practice explaining the story in words.
            """
        )

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

    with st.expander("What do these sliders roughly mean for markets?"):
        st.markdown(
            """
            - **Interest Rates** – Higher rates usually pressure growth stocks and real estate, but can help banks.  
            - **Inflation** – High inflation hurts bonds and some consumers, but can help real assets.  
            - **GDP Growth** – Strong growth supports cyclical sectors; weak growth favors defensives.  
            - **Unemployment** – Rising unemployment usually pressures earnings and risky assets.  
            - **Oil Prices** – High oil prices boost energy, but hurt transportation and some consumer names.  
            - **Geopolitical Tension** – Raises uncertainty and risk premiums, sometimes helping defense or energy.
            """
        )

    df = compute_sector_scores(macro_values)

    st.markdown("### Sector impact overview")

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
            .mark_bar(color=colors["accent"])
            .encode(
                x=alt.X("Sector:N", sort=None),
                y=alt.Y("Impact Score:Q"),
                tooltip=["Sector", "Impact Score", "Impact Label"],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Quick verbal summary")

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

    st.caption("Note: This is a simplified educational model. Real markets are more complex and noisy.")


# ------------------------------------------------------------
# TAB 2 – AI Chatbot
# ------------------------------------------------------------
with tab_chat:
    st.subheader("Ask the Finance Tutor (Gemini-powered, educational only)")

    if gemini_model is None:
        st.warning(
            "Gemini API key is not configured yet. Add GOOGLE_API_KEY to Secrets in Streamlit Cloud "
            "so this chatbot can answer questions."
        )

    st.markdown(
        """
        Ask about **stocks, ETFs, sectors, diversification, risk, or macro environments**.  
        The assistant explains concepts but **never tells you to buy or sell anything.**
        """
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Quick prompt buttons
    st.markdown("**Try a sample question:**")
    c1, c2, c3 = st.columns(3)
    sample_q = None
    if c1.button("📦 What is an ETF?"):
        sample_q = "Explain what an ETF is to a high-school student."
    if c2.button("📉 Why do rate hikes hurt growth stocks?"):
        sample_q = "Why do rising interest rates often hurt growth stocks like tech?"
    if c3.button("🛡️ What is diversification?"):
        sample_q = "What does diversification mean in investing, and why does it matter?"

    user_input = st.chat_input("Type your own question here...")

    if sample_q and not user_input:
        user_input = sample_q

    # Show history
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


# ------------------------------------------------------------
# TAB 3 – Learning Center
# ------------------------------------------------------------
with tab_learn:
    st.subheader("Key Ideas Behind This Project")

    st.markdown(
        """
        This app is designed as a **teaching tool** that a teacher, mentor, or college
        admissions reader can quickly understand.
        """
    )

    st.markdown("#### 1. Macro → Markets intuition")
    st.markdown(
        """
        - Changes in **interest rates, inflation, growth, unemployment, oil, and geopolitics**
          don't move all stocks the same way.  
        - Investors often talk about **sectors** (Tech, Banks, Energy, etc.).  
        - This app encodes a simple rule-of-thumb model for how each sector *might* react.
        """
    )

    st.markdown("#### 2. Educational model, not trading system")
    st.markdown(
        """
        - The numbers are **normalized impact scores**, not price targets or return forecasts.  
        - The goal is for students to practice *explaining stories* like:  
          *\"In a rate-hike environment, banks may benefit but growth stocks can be under pressure.\"*  
        - Real markets are more complex and sometimes do the opposite of what textbooks suggest.
        """
    )

    st.markdown("#### 3. Why this is a strong student project")
    st.markdown(
        """
        - Combines **coding (Python + Streamlit)**, **data visualization (Altair)**,
          and **economic/finance reasoning**.  
        - Shows how to integrate a modern **AI model (Gemini)** safely as an educational tutor.  
        - The UI is intentionally clean and easy for non-technical reviewers to explore.
        """
    )

    st.markdown("#### 4. How you might describe it in an application")
    st.markdown(
        """
        *\"Built an interactive Macro & Markets Explorer web app in Python/Streamlit that models
        how different macroeconomic environments can affect stock market sectors.  
        Added a Gemini-powered finance tutor that explains concepts like sectors, ETFs,
        diversification, and rate hikes in student-friendly language.  
        Designed the app for classmates to experiment with scenarios and practice
        explaining their reasoning like a junior portfolio manager.\"*
        """
    )
