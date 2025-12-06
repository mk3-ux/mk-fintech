
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Macro & Markets Explorer", page_icon="📈", layout="wide")

SECTORS = [
    "Tech", "Banks", "Real Estate", "Consumer Staples",
    "Luxury / Discretionary", "Energy", "Bonds",
]

SECTOR_SENSITIVITIES = {
    "Tech": {"rates": -2, "inflation": -1, "gdp": 2, "unemployment": -1,
             "oil": -1, "geopolitics": -2, "supply_chain": -2, "usd": 1,
             "confidence": 2, "commodities": -1},
    "Banks": {"rates": 2, "inflation": 1, "gdp": 1, "unemployment": -1,
              "oil": 0, "geopolitics": 0, "supply_chain": 0, "usd": 1,
              "confidence": 1, "commodities": 0},
    "Real Estate": {"rates": -2, "inflation": -1, "gdp": 1, "unemployment": -1,
                    "oil": -1, "geopolitics": -1, "supply_chain": -1, "usd": 0,
                    "confidence": 1, "commodities": -1},
    "Consumer Staples": {"rates": 0, "inflation": 1, "gdp": 0, "unemployment": 1,
                         "oil": 0, "geopolitics": 0, "supply_chain": 0, "usd": 0,
                         "confidence": -1, "commodities": -1},
    "Luxury / Discretionary": {"rates": -1, "inflation": -1, "gdp": 2, "unemployment": -2,
                               "oil": -1, "geopolitics": -1, "supply_chain": -1, "usd": 0,
                               "confidence": 2, "commodities": -1},
    "Energy": {"rates": 0, "inflation": 2, "gdp": 1, "unemployment": 0,
               "oil": 2, "geopolitics": 1, "supply_chain": 1, "usd": -1,
               "confidence": 0, "commodities": 2},
    "Bonds": {"rates": -2, "inflation": -1, "gdp": -1, "unemployment": 1,
              "oil": 0, "geopolitics": 1, "supply_chain": 0, "usd": 1,
              "confidence": -1, "commodities": -1},
}

PRESET_SCENARIOS = {
    "Interest Rates Go Up": {
        "factors": {"rates": 2, "inflation": 0, "gdp": 0, "unemployment": 0,
                    "oil": 0, "geopolitics": 0, "supply_chain": 0, "usd": 1,
                    "confidence": -1, "commodities": 0},
        "description": (
            "Central banks raise interest rates to cool the economy or fight inflation. "
            "Housing and growth stocks may fall, while banks may benefit."
        ),
    },
    "High Inflation Shock": {
        "factors": {"rates": 1, "inflation": 2, "gdp": -1, "unemployment": 0,
                    "oil": 1, "geopolitics": 0, "supply_chain": 1, "usd": 0,
                    "confidence": -1, "commodities": 2},
        "description": (
            "Inflation rises quickly. Prices go up, central banks may react, and companies face pressure."
        ),
    },
    "Recession Fears": {
        "factors": {"rates": -1, "inflation": 0, "gdp": -2, "unemployment": 2,
                    "oil": -1, "geopolitics": 0, "supply_chain": 0, "usd": 1,
                    "confidence": -2, "commodities": -1},
        "description": (
            "Economic slowdown causes lower spending, higher unemployment, and market uncertainty."
        ),
    },
    "Economic Boom": {
        "factors": {"rates": 1, "inflation": 1, "gdp": 2, "unemployment": -2,
                    "oil": 1, "geopolitics": 0, "supply_chain": 0, "usd": 0,
                    "confidence": 2, "commodities": 1},
        "description": (
            "Strong GDP growth, strong hiring, and rising company profits lift many sectors."
        ),
    },
}

def calculate_sector_scores(factors):
    rows = []
    for sector in SECTORS:
        sens = SECTOR_SENSITIVITIES[sector]
        score = sum(factors[k] * sens[k] for k in factors)
        label = (
            "Strong Negative" if score <= -3 else
            "Mild Negative" if score < 0 else
            "Neutral" if score == 0 else
            "Mild Positive" if score < 3 else
            "Strong Positive"
        )
        rows.append({"Sector": sector, "Impact Score": score, "Impact Label": label})
    return pd.DataFrame(rows).sort_values("Impact Score")

def explain_factors(f):
    out = []
    if f["rates"] != 0:
        out.append("Interest rates rising slows borrowing; falling rates boost borrowing.")
    if f["inflation"] != 0:
        out.append("Inflation affects prices, purchasing power, and company costs.")
    if f["gdp"] != 0:
        out.append("GDP growth reflects the economic cycle: expansion vs recession.")
    if f["unemployment"] != 0:
        out.append("Unemployment affects income, spending, and consumer demand.")
    if f["oil"] != 0:
        out.append("Oil affects transport, manufacturing, and energy company profits.")
    if f["geopolitics"] != 0:
        out.append("Geopolitical tension can increase market risk and volatility.")
    return out or ["Neutral environment with no major macro pressures."]

st.title("📈 Macro & Markets Explorer")

mode = st.sidebar.radio("Choose mode:", ["Pre-set Macro Scenarios", "Build Your Own Scenario"])

if mode == "Pre-set Macro Scenarios":
    scenario_name = st.sidebar.selectbox("Select a macro scenario:", list(PRESET_SCENARIOS.keys()))
    scenario = PRESET_SCENARIOS[scenario_name]
    st.subheader(scenario_name)
    st.write(scenario["description"])
    factors = scenario["factors"]
    df = calculate_sector_scores(factors)
    st.dataframe(df)
    st.bar_chart(df.set_index("Sector")["Impact Score"])
else:
    st.subheader("Build Your Own Scenario")
    factors = {
        "rates": st.slider("Interest Rates", -2, 2, 0),
        "inflation": st.slider("Inflation", -2, 2, 0),
        "gdp": st.slider("GDP Growth", -2, 2, 0),
        "unemployment": st.slider("Unemployment", -2, 2, 0),
        "oil": st.slider("Oil Prices", -2, 2, 0),
        "geopolitics": st.slider("Geopolitical Tension", 0, 3, 0),
        "supply_chain": st.slider("Supply Chain Stress", 0, 3, 0),
        "usd": st.slider("USD Strength", -2, 2, 0),
        "confidence": st.slider("Consumer Confidence", -2, 2, 0),
        "commodities": st.slider("Commodity Pressure", -2, 2, 0),
    }
    df = calculate_sector_scores(factors)
    st.dataframe(df)
    st.bar_chart(df.set_index("Sector")["Impact Score"])
    st.write("### Explanation")
    for line in explain_factors(factors):
        st.write("- " + line)
