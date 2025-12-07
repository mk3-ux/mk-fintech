with st.sidebar:
    st.title("Katta MacroSuite")

    ai_level = st.radio(
        "AI output style:",
        ["Professional", "Executive", "Technical"],
        index=0
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
