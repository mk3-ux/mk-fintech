import os
import streamlit as st
import pandas as pd
from groq import Groq
from fpdf import FPDF

# -------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Katta FinSight – Macro & Markets Explorer",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------------------------
# CUSTOM GLOBAL CSS (fonts, spacing, colors)
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    .kfs-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        margin-bottom: 0.1rem;
    }

    .kfs-subtitle {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 0.2rem;
    }

    .kfs-author {
        font-size: 0.95rem;
        color: #777;
        margin-bottom: 0.4rem;
    }

    .kfs-tag {
        display: inline-block;
        padding: 0.28rem 0.8rem;
        border-radius: 999px;
        background-color: #E6F4EA;
        color: #137333;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    button[role="tab"] > div {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }

    h2, .kfs-section-title {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }

    h3 {
        font-size: 1.15rem !important;
        font-weight: 650 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# HEADER WITH LOGO + TITLE
# -------------------------------------------------------------------
col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.image("kattafinsight_logo.png", use_column_width=True)

with col_title:
    st.markdown('<div class="kfs-title">Katta FinSight</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="kfs-subtitle">Macro & Markets Explorer – understanding how macro environments influence stock sectors.</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="kfs-author">by Menaj Katta</div>', unsafe_allow_html=True)
    # Remove this next line if you don't want the badge
    st.markdown('<span class="kfs-tag">Educational Project</span>', unsafe_allow_html=True)


# -------------------------------------------------------------------
# CREATE TABS
# -------------------------------------------------------------------
tab_sector, tab_chatbot, tab_analyst, tab_report, tab_about = st.tabs(
    [
        "📊 Sector & Scenario Explorer",
        "🤖 Finance Chatbot",
        "🧠 AI Research Analyst",
        "📄 Generate Report",
        "ℹ️ About Us",
    ]
)

# -------------------------------------------------------------------
# TAB 1 – MACRO SCENARIO + SECTOR IMPACTS
# -------------------------------------------------------------------
with tab_sector:

    st.markdown("## Sector & Scenario Explorer")

    st.write("Adjust macro variables and explore how they influence stock sectors.")

    # Example sliders (replace with your real logic)
    interest = st.slider("Interest Rates", -5, 5, 0)
    inflation = st.slider("Inflation", -5, 5, 0)
    gdp = st.slider("GDP Growth", -5, 5, 0)

    st.write("### Example Output (replace with your real sector model)")
    df = pd.DataFrame({
        "Sector": ["Tech", "Finance", "Real Estate"],
        "Impact": [interest * -1, interest + gdp, inflation * -1]
    })
    st.dataframe(df)


# -------------------------------------------------------------------
# TAB 2 – FINANCE CHATBOT (Groq API)
# -------------------------------------------------------------------
with tab_chatbot:

    st.markdown("## Finance Chatbot (Educational Only)")
    st.write("Ask questions about stocks, ETFs, diversification, inflation, etc.")

    groq_key = st.secrets.get("GROQ_API_KEY", None)

    if not groq_key:
        st.error("GROQ_API_KEY missing in Secrets.")
    else:
        client = Groq(api_key=groq_key)

        user_q = st.text_input("Ask a finance question:")
        if st.button("Ask AI"):
            if user_q.strip() == "":
                st.warning("Please enter a question.")
            else:
                try:
                    response = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[
                            {"role": "system", "content": "You are a finance tutor. Explain clearly but do NOT give investment advice."},
                            {"role": "user", "content": user_q}
                        ]
                    )
                    st.success(response.choices[0].message["content"])
                except Exception as e:
                    st.error(f"AI error: {e}")


# -------------------------------------------------------------------
# TAB 3 – AI RESEARCH ANALYST
# -------------------------------------------------------------------
with tab_analyst:

    st.markdown("## AI Research Analyst — Generate Internal Narratives")

    prompt = st.text_area("Describe what you want the analyst to write:")

    if st.button("Run AI Research Analyst"):
        if not groq_key:
            st.error("GROQ_API_KEY missing in Secrets.")
        else:
            try:
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": "You write internal research memos. No financial advice."},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.write(response.choices[0].message["content"])
            except Exception as e:
                st.error(f"AI error: {e}")


# -------------------------------------------------------------------
# TAB 4 – PDF REPORT (FPDF)
# -------------------------------------------------------------------
with tab_report:

    st.markdown("## Generate PDF Report")

    report_text = st.text_area("Enter content for the PDF:")

    if st.button("Generate PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in report_text.split("\n"):
            pdf.cell(0, 10, txt=line, ln=True)

        pdf.output("report.pdf")
        st.success("PDF created!")
        with open("report.pdf", "rb") as f:
            st.download_button("Download PDF", f, file_name="KattaFinSight_Report.pdf")


# -------------------------------------------------------------------
# TAB 5 – ABOUT US
# -------------------------------------------------------------------
with tab_about:

    st.markdown("## About Katta FinSight")
    st.write("**Coming soon…**")
