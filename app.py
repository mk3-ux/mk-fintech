import os
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import requests
from fpdf import FPDF
from textblob import TextBlob
import yfinance as yf
from groq import Groq

# ---------------------------
# CONFIG / API KEYS
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # <-- Set in environment or Streamlit Secrets
MODEL_NAME = "llama-3.1-8b-instant"
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY_HERE"  # <-- Replace with your NewsAPI key

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ---------------------------
# THEME & STYLING
# ---------------------------
st.set_page_config(page_title="Katta MacroSuite", layout="wide", page_icon="📊")
COLORS = {"bg": "#F7FAFF","text": "#0F172A","accent": "#0EA5E9","subtle": "#E6EEF8"}
st.markdown(f"""
<style>
    .stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}
    .block-container {{ max-width: 1400px; padding-top: 1.2rem; padding-bottom: 2rem; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# DATA DEFINITIONS
# ---------------------------
sectors = ["Tech","Real Estate","Luxury / Discretionary","Bonds","Energy","Consumer Staples","Banks"]

# ---------------------------
# HELPERS
# ---------------------------
def compute_stock_sector_impacts(stock_move: float, primary_sector: str) -> pd.DataFrame:
    rows = []
    for sec in sectors:
        sensitivity = 1.0 if sec==primary_sector else 0.4
        raw_score = sensitivity * stock_move
        rows.append({"Sector":sec,"Sensitivity":sensitivity,"Raw Score":raw_score})
    df = pd.DataFrame(rows)
    max_abs = df["Raw Score"].abs().max()
    df["Impact Score"] = (df["Raw Score"]/max_abs*5).round(2) if max_abs else 0
    def label(score):
        if score<=-3.5: return "Strong Negative"
        if score<=-1.5: return "Mild Negative"
        if score<1.5: return "Neutral"
        if score<3.5: return "Mild Positive"
        return "Strong Positive"
    df["Impact Label"] = df["Impact Score"].apply(label)
    return df[["Sector","Impact Score","Impact Label"]]

def compute_portfolio_exposure(portfolio_df: pd.DataFrame, sector_scores: pd.DataFrame):
    df = portfolio_df.copy()
    if "Allocation" not in df.columns: raise ValueError("Portfolio CSV must include 'Allocation'")
    df["Weight"] = df["Allocation"]/df["Allocation"].sum() if df["Allocation"].sum()>0 else 0
    merged = df.merge(sector_scores, on="Sector", how="left")
    merged["Impact Score"].fillna(0, inplace=True)
    merged["Weighted Impact"] = merged["Weight"]*merged["Impact Score"]
    portfolio_score = merged["Weighted Impact"].sum()
    health_score = min(max(50+portfolio_score*10,0),100)  # Gamified health
    return portfolio_score, health_score, merged

def call_ai_research(history, user_text: str, level: str = "Professional") -> str:
    if client is None: return "AI not configured. Set GROQ_API_KEY."
    level_line = {"Professional":"Concise corporate analyst",
                  "Executive":"Executive summary",
                  "Technical":"Detailed memo"}.get(level,"Professional")
    system_prompt = f"You are an internal analyst. {level_line}. Do NOT give buy/sell advice."
    messages = [{"role":"system","content":system_prompt}] + history[-6:] + [{"role":"user","content":user_text}]
    try:
        completion = client.chat.completions.create(model=MODEL_NAME, messages=messages, max_tokens=700, temperature=0.2)
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"AI error: {e}"

def fetch_news(keyword="finance", page_size=5):
    url = "https://newsapi.org/v2/everything"
    params = {"q": keyword, "language":"en","sortBy":"publishedAt","pageSize":page_size,"apiKey":NEWSAPI_KEY}
    try:
        r = requests.get(url, params=params)
        return r.json().get("articles", [])
    except:
        return []

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "Positive" if polarity>0.1 else "Negative" if polarity<-0.1 else "Neutral"

def get_stock_data(ticker, period="1y"):
    df = yf.Ticker(ticker).history(period=period)
    df.reset_index(inplace=True)
    return df

def create_pdf_report(title, scenario_name, scenario_meta, sector_df, portfolio_table=None, ai_summary=""):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="L")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Scenario: {scenario_name}", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Scenario Inputs:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for k,v in scenario_meta.items(): pdf.cell(0,7,f"{k}: {v}",ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0,8,"Sector Impacts:",ln=True)
    pdf.set_font("Helvetica","",11)
    for _,r in sector_df.iterrows(): pdf.cell(0,7,f"{r['Sector']}: {r['Impact Score']} ({r['Impact Label']})",ln=True)
    pdf.ln(4)
    if portfolio_table is not None and not portfolio_table.empty:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0,8,"Portfolio Exposure:",ln=True)
        pdf.set_font("Helvetica","",10)
        for _,r in portfolio_table.iterrows():
            pdf.cell(0,6,f"{r['Sector']}: Allocation {r['Allocation']}, WeightedImpact {r['Weighted Impact']:.3f}",ln=True)
        pdf.ln(4)
    if ai_summary:
        pdf.set_font("Helvetica","B",12)
        pdf.cell(0,8,"AI Research Summary:",ln=True)
        pdf.set_font("Helvetica","",10)
        for line in ai_summary.split("\n"): pdf.multi_cell(0,6,line)
    return pdf.output(dest="S").encode("latin-1")

# ---------------------------
# SESSION STATE
# ---------------------------
if "sector_df" not in st.session_state: st.session_state["sector_df"]=None
if "scenario_name" not in st.session_state: st.session_state["scenario_name"]=None
if "scenario_meta" not in st.session_state: st.session_state["scenario_meta"]={}
if "portfolio_df" not in st.session_state: st.session_state["portfolio_df"]=None
if "ai_history" not in st.session_state: st.session_state["ai_history"]=[]

# ---------------------------
# UI HEADER
# ---------------------------
st.markdown(f"""
<div style="background:linear-gradient(90deg,#ECFEFF,#F0F9FF);padding:12px;border-radius:14px;border:1px solid {COLORS['subtle']};margin-bottom:14px;">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="width:56px;height:56px;border-radius:12px;background:{COLORS['accent']};color:white;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:20px;">
      KM
    </div>
    <div>
      <div style="font-size:18px;font-weight:800;color:{COLORS['text']};">Katta MacroSuite — Market Intelligence</div>
      <div style="font-size:12px;color:#475569;">Stock impact • Sector sensitivity • Portfolio exposure • AI Research • Live news</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.title("Katta MacroSuite")
    ai_level = st.radio("AI output style:", ["Professional","Executive","Technical"], index=0)
    st.markdown("---")
    st.markdown("Upload a CSV portfolio (columns: Sector, Allocation)")
    st.markdown("---")
    st.caption("Decision-support only — no buy/sell advice.")

# ---------------------------
# MAIN TABS
# ---------------------------
tab_explorer, tab_portfolio, tab_ai, tab_reports, tab_news = st.tabs([
    "Stock Explorer","Portfolio Analyzer","AI Analyst","Reports","Finance News"
])

# ---------- STOCK EXPLORER ----------
with tab_explorer:
    st.subheader("Stock Impact Explorer")
    ticker = st.text_input("Stock ticker","AAPL")
    primary_sector = st.selectbox("Primary sector", sectors)
    stock_move = st.slider("Assumed price move (%)",-20,20,0)
    st.caption("Primary sector strongest impact, others spillover.")
    sector_df = compute_stock_sector_impacts(stock_move, primary_sector)
    st.session_state["sector_df"] = sector_df
    st.session_state["scenario_name"] = f"{ticker} {stock_move:+.1f}% move"
    st.session_state["scenario_meta"] = {"Stock":ticker,"Move (%)":stock_move,"Primary Sector":primary_sector}

    col1,col2 = st.columns([2,3])
    with col1:
        st.dataframe(sector_df,use_container_width=True)
    with col2:
        df_price = get_stock_data(ticker)
        if not df_price.empty:
            fig = px.line(df_price,x="Date",y="Close",title=f"{ticker} Price History")
            st.plotly_chart(fig,use_container_width=True)

# ---------- PORTFOLIO ----------
with tab_portfolio:
    st.subheader("Portfolio Analyzer")
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])
    if uploaded:
        portfolio_df = pd.read_csv(uploaded)
        st.session_state["portfolio_df"]=portfolio_df
        st.dataframe(portfolio_df.head(20), use_container_width=True)
    if st.button("Analyze exposure"):
        sector_df_current = st.session_state.get("sector_df")
        portfolio_df_current = st.session_state.get("portfolio_df")
        if sector_df_current and portfolio_df_current:
            score, health, breakdown = compute_portfolio_exposure(portfolio_df_current, sector_df_current)
            st.metric("Weighted Impact Score", f"{score:.3f}")
            st.metric("Portfolio Health", f"{health:.0f}/100")
            st.dataframe(breakdown,use_container_width=True)
        else:
            st.warning("Stock scenario or portfolio missing.")

# ---------- AI ANALYST ----------
with tab_ai:
    st.subheader("AI Research Analyst")
    user_q = st.text_area("AI request",height=120)
    ai_style = st.selectbox("Style", ["Professional","Executive","Technical"])
    if st.button("Run AI"):
        sector_df_current = st.session_state.get("sector_df")
        scenario_name = st.session_state.get("scenario_name","Current scenario")
        scenario_meta = st.session_state.get("scenario_meta",{})
        context = f"Scenario: {scenario_name}\nInputs: {scenario_meta}\nTop sectors:\n"
        for _,r in sector_df_current.sort_values("Impact Score",ascending=False).head(3).iterrows():
            context += f"- {r['Sector']}: {r['Impact Score']} ({r['Impact Label']})\n"
        prompt = context + "\nUser request:\n"+user_q
        st.session_state["ai_history"].append({"role":"user","content":prompt})
        out = call_ai_research(st.session_state["ai_history"], prompt, ai_style)
        st.markdown("**AI Output**")
        st.markdown(out)
        st.session_state["ai_history"].append({"role":"assistant","content":out})
        st.session_state["ai_history"]=st.session_state["ai_history"][-20:]

# ---------- REPORT ----------
with tab_reports:
    st.subheader("Generate PDF Report")
    report_title = st.text_input("Report title","Stock Scenario & Portfolio Insight")
    include_portfolio = st.checkbox("Include portfolio",value=True)
    ai_summary_for_report = st.text_area("AI summary",height=120)
    if st.button("Create PDF"):
        sector_df_current = st.session_state.get("sector_df")
        portfolio_df_current = st.session_state.get("portfolio_df")
        portfolio_table = None
        if include_portfolio and portfolio_df_current is not None:
            _, _, portfolio_table = compute_portfolio_exposure(portfolio_df_current, sector_df_current)
        pdf_bytes = create_pdf_report(report_title, st.session_state.get("scenario_name","Current scenario"), st.session_state.get("scenario_meta",{}), sector_df_current, portfolio_table, ai_summary_for_report)
        st.download_button("Download PDF", data=pdf_bytes, file_name="katta_report.pdf", mime="application/pdf")

# ---------- FINANCE NEWS ----------
with tab_news:
    st.subheader("Finance News & Sentiment")
    keyword = st.text_input("Keyword","finance")
    articles = fetch_news(keyword)
    if articles:
        for art in articles:
            sentiment = analyze_sentiment(art.get("title","")+art.get("description",""))
            st.markdown(f"**{art['title']}** ({sentiment})")
            st.write(art.get("description",""))
            st.write(f"[Read more]({art['url']})")
            st.write("---")
    else:
        st.write("No news found.")

st.caption("Katta MacroSuite — Internal decision-support. Not investment advice.")
