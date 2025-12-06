import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Katta Fintech – Macro & Markets Explorer",
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
        <div style="font-size:0.9rem; color:#6B7280;">Macro &amp; Markets Explorer</div>
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

st.sidebar.subheader("Mode")
mode = st.sidebar.radio(
    "Choose mode:",
    ["Pre-set Macro Scenarios", "Build Your Own Scenario"],
)

# ------------------------------------------------------------
# Define macro variables & sectors
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

# ------------------------------------------------------------
# Scenario definitions (for preset mode)
# Values range roughly -5 (very negative) to +5 (very positive)
# ------------------------------------------------------------
preset_scenarios = {
    "Interest Rates Go Up": {
        "description": (
            "Central banks raise interest rates to cool the economy or fight inflation. "
            "Housing and growth stocks may fall, while banks may benefit."
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
    "High Inflation": {
        "description": (
            "Prices are rising quickly. Central banks may hike rates, consumers cut back on "
            "non-essential spending, and companies with pricing power do better."
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
    "Recession Risk": {
        "description": (
            "Economic growth slows sharply or turns negative. Unemployment rises and investors "
            "move into safer, defensive sectors."
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
    "Oil Shock": {
        "description": (
            "Oil prices spike after a supply shock. Energy companies can benefit, but "
            "transportation and consumers feel pressure."
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

# ------------------------------------------------------------
# Weights model: how each macro variable affects each sector
# (simple, transparent, Ivy-friendly explanation)
# Positive weight = helps the sector when the macro value is high
# Negative weight = hurts the sector when the macro value is high
# ------------------------------------------------------------
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
# Main layout
# ------------------------------------------------------------
col_main, col_side = st.columns([3, 1])

with col_main:
    st.title("Macro & Markets Explorer")

    if mode == "Pre-set Macro Scenarios":
        st.subheader("Pre-set Macro Scenarios")
    else:
        st.subheader("Build Your Own Scenario")

with col_side:
    st.markdown(
        """
        **How to use this app**

        1. Choose a mode on the left.  
        2. Pick a macro scenario or set your own sliders.  
        3. Read the sector impact table.  
        4. Use the chart to compare winners and losers.
        """
    )

st.markdown("")  # small spacing

# ------------------------------------------------------------
# Scenario selection / sliders
# ------------------------------------------------------------
if mode == "Pre-set Macro Scenarios":
    scenario_name = st.selectbox("Select a macro scenario:", list(preset_scenarios.keys()))
    scenario = preset_scenarios[scenario_name]

    st.markdown(f"**Scenario: {scenario_name}**")
    st.markdown(scenario["description"])

    macro_values = scenario["macros"]

else:
    st.markdown("Use the sliders to build your own macro scenario.")

    macro_values = {}
    # Use columns to keep sliders tidy
    c1, c2 = st.columns(2)
    with c1:
        macro_values["Interest Rates"] = st.slider("Interest Rates", -5, 5, 0)
        macro_values["GDP Growth"] = st.slider("GDP Growth", -5, 5, 0)
        macro_values["Oil Prices"] = st.slider("Oil Prices", -5, 5, 0)
    with c2:
        macro_values["Inflation"] = st.slider("Inflation", -5, 5, 0)
        macro_values["Unemployment"] = st.slider("Unemployment", -5, 5, 0)
        macro_values["Geopolitical Tension"] = st.slider("Geopolitical Tension", -5, 5, 0)

# ------------------------------------------------------------
# Explain macro variables (tooltips via expander)
# ------------------------------------------------------------
with st.expander("What do these macro variables mean?"):
    st.markdown(
        """
        - **Interest Rates** – How expensive it is to borrow money. Higher rates usually slow
          the economy and hurt growth stocks and housing, but can help banks.
        - **Inflation** – How quickly prices are rising. High inflation erodes purchasing power
          and can force central banks to raise interest rates.
        - **GDP Growth** – How fast the total economy is growing. Strong growth supports
          corporate earnings; weak growth or negative GDP signals recession risk.
        - **Unemployment** – The percentage of people without jobs. Rising unemployment usually
          means weaker demand and pressure on most sectors.
        - **Oil Prices** – Cost of energy and transportation. High oil prices can help energy
          companies but hurt transportation and consumer spending.
        - **Geopolitical Tension** – Wars, conflicts, and international disputes that can disrupt
          trade, supply chains, and investor confidence.
        """
    )

# ------------------------------------------------------------
# Compute & display sector impacts
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Summary: winners & losers in plain English
# ------------------------------------------------------------
st.markdown("### Summary")

sorted_df = df.sort_values("Impact Score", ascending=False)
winners = sorted_df.head(2)
losers = sorted_df.tail(2)

winner_text = ", ".join(f"{row.Sector} ({row['Impact Label']})" for _, row in winners.iterrows())
loser_text = ", ".join(f"{row.Sector} ({row['Impact Label']})" for _, row in losers.iterrows())

st.markdown(
    f"""
    - **Likely winners:** {winner_text}  
    - **Likely under pressure:** {loser_text}
    """
)

# ------------------------------------------------------------
# Model details (for college / interviewer)
# ------------------------------------------------------------
with st.expander("How this model works (for teachers / reviewers)"):
    st.markdown(
        """
        This app is **not** a trading system. It is a learning tool that uses a simple,
        transparent scoring model:

        1. Each macro variable (interest rates, inflation, etc.) is scaled from **-5** (strong
           negative shock) to **+5** (strong positive shock).
        2. For each sector, we assign intuitive **weights** that represent how sensitive that
           sector is to each macro variable. For example:
           - Banks have a **positive weight** to interest rates (they often benefit when rates rise).
           - Tech and Real Estate have **negative weights** to higher interest rates.
           - Energy has a **strong positive weight** to oil prices.
        3. The app calculates a raw score for each sector as a weighted sum of macro values, then
           rescales scores roughly into the range **-5 to +5**.
        4. Finally, we convert these scores into labels like *Strong Negative*, *Mild Positive*, etc.

        This approach makes the logic easy to explain to students, teachers, or interviewers while
        still reflecting real economic intuition.
        """
    )
