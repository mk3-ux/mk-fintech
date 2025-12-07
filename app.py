import os
import streamlit as st
import pandas as pd
import altair as alt
import requests
from fpdf import FPDF
from groq import Groq
from textblob import TextBlob
import yfinance as yf

# ---------------------------
# CONFIG / KEYS
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY_HERE"

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

st.set_page_config(page_title="Katta MacroSuite 2.0", layout="wide", page_icon="📊")

# ---------------------------
# COLORS / CSS
# ---------------------------
COLORS = {"bg":"#F7FAFF","text":"#0F172A","accent":"#0EA5E9","subtle":"#E6EEF8"}
st.markdown(f"""
<style>
    .stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}
    .block-container {{ max-width: 1400px; padding-top: 1rem; padding-bottom: 2rem; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# DATA / SECTORS
# ---------------------------
sectors = ["Tech","Real Estate","Luxury / Discretionary","Bonds","Energy","Consumer Staples","Banks"]

# ---------------------------
# HELPERS
# ---------------------------
def compute_stock_sector_impacts(stock_move: float, primary_sector: str) -> pd.DataFrame:
    rows=[]
    for sec in sectors:
        sensitivity = 1.0 if sec==primary_sector else 0.4
        raw_score = sensitivity*stock_move
        rows.append({"Sector":sec,"Sensitivity":sensitivity,"Raw Score":raw_score})
    df=pd.DataFrame(rows)
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
    df=portfolio_df.copy()
    if "Allocation" not in df.columns: raise ValueError("Portfolio CSV must include 'Allocation'")
    total=df["Allocation"].sum()
    df["Weight"]=df["Allocation"]/total if total else 0
    merged=df.merge(sector_scores,on="Sector",how="left")
    merged["Impact Score"].fillna(0,inplace=True)
    merged["Weighted Impact"]=merged["Weight"]*merged["Impact Score"]
    portfolio_score = merged["Weighted Impact"].sum()
    return portfolio_score, merged[["Sector","Allocation","Weight","Impact Score","Weighted Impact"]]

def call_ai_research(history,user_text:str,level="Professional") -> str:
    if client is None: return "AI not configured. Set GROQ_API_KEY."
    level_line = {"Professional":"Concise analyst notes",
                  "Executive":"Executive summary for management",
                  "Technical":"Detailed technical memo"}.get(level,"Professional")
    system_prompt=f"You are an internal AI research analyst. {level_line}. Do NOT give buy/sell advice."
    messages=[{"role":"system","content":system_prompt}]+history[-6:]+[{"role":"user","content":user_text}]
    try:
        completion=client.chat.completions.create(model=MODEL_NAME,messages=messages,max_tokens=700,temperature=0.2)
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"AI error: {e}"

def fetch_news(keyword="finance",page_size=5):
    url="https://newsapi.org/v2/everything"
    params={"q":keyword,"language":"en","sortBy":"publishedAt","pageSize":page_size,"apiKey":NEWSAPI_KEY}
    try:
        r=requests.get(url,params=params)
        return r.json().get("articles",[])
    except:
        return []

def analyze_sentiment(text):
    polarity=TextBlob(text).sentiment.polarity
    if polarity>0.1: return "Positive"
    elif polarity<-0.1: return "Negative"
    return "Neutral"

def get_stock_data(ticker,period="1y"):
    df=yf.Ticker(ticker).history(period=period)
    df.reset_index(inplace=True)
    return df

def create_pdf_report(title, scenario_name, scenario_meta, sector_df, portfolio_table=None, ai_summary=""):
    from fpdf import FPDF
    pdf=FPDF()
    pdf.set_auto_page_break(True,12)
    pdf.add_page()
    pdf.set_font("Helvetica","B",16)
    pdf.cell(0,10,title,ln=True)
    pdf.set_font("Helvetica","",11)
    pdf.cell(0,8,f"Scenario: {scenario_name}",ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8,"Scenario Inputs:",ln=True)
    pdf.set_font("Helvetica","",11)
    for k,v in scenario_meta.items(): pdf.cell(0,7,f"{k}: {v}",ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8,"Sector Impacts:",ln=True)
    pdf.set_font("Helvetica","",11)
    for _,r in sector_df.iterrows(): pdf.cell(0,7,f"{r['Sector']}: {r['Impact Score']} ({r['Impact Label']})",ln=True)
    pdf.ln(4)
    if portfolio_table is not None and not portfolio_table.empty:
        pdf.set_font("Helvetica","B",12)
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
# HEADER
# ---------------------------
st.markdown(f"""
<div style="background:linear-gradient(90deg,#ECFEFF,#F0F9FF);padding:12px;border-radius:14px;border:1px solid {COLORS['subtle']};margin-bottom:14px;">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="width:56px;height:56px;border-radius:12px;background:{COLORS['accent']};color:white;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:20px;">KM</div>
    <div>
      <div style="font-size:18px;font-weight:800;color:{COLORS['text']};">Katta MacroSuite 2.0 — Market Intelligence</div>
      <div style="font-size:12px;color:#475569;">Multi-stock simulation • Sector sensitivity • Portfolio exposure • AI Insights • News intelligence</div>
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
# TABS
# ---------------------------
tab_sim, tab_portfolio, tab_ai, tab_reports, tab_news = st.tabs([
    "Multi-Stock Simulator","Portfolio Analyzer","AI Insights","Reports","Live News"
])

# ---------- MULTI-STOCK SIMULATION ----------
with tab_sim:
    st.subheader("Scenario Simulation Dashboard")
    uploaded = st.file_uploader("Upload CSV with columns: Ticker, Move%", type=["csv"])
    if uploaded:
        try:
            scenario_df = pd.read_csv(uploaded)
            st.session_state["scenario_df_multi"] = scenario_df
            st.dataframe(scenario_df, use_container_width=True)
        except Exception as e:
            st.error(f"CSV error: {e}")

    if st.button("Run Simulation") and "scenario_df_multi" in st.session_state:
        sim_df = st.session_state["scenario_df_multi"]
        sector_results = []
        for _, row in sim_df.iterrows():
            ticker = row["Ticker"]
            move = row["Move%"]
            primary_sector = row.get("PrimarySector", "Tech")
            sec_df = compute_stock_sector_impacts(move, primary_sector)
            sec_df["Ticker"]=ticker
            sector_results.append(sec_df)
        combined = pd.concat(sector_results)
        pivot = combined.pivot_table(index="Sector",columns="Ticker",values="Impact Score")
        st.markdown("### Sector Impact Heatmap")
        chart = alt.Chart(combined).mark_rect().encode(
            x='Ticker:N',
            y='Sector:N',
            color='Impact Score:Q',
            tooltip=['Sector','Ticker','Impact Score','Impact Label']
        )
        st.altair_chart(chart,use_container_width=True)
        st.session_state["sector_df"]=combined.groupby("Sector")[["Impact Score"]].mean().reset_index()

# ---------- PORTFOLIO ANALYZER ----------
with tab_portfolio:
    st.subheader("Portfolio Exposure Analyzer")
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"], key="portfolio_tab")
    if uploaded:
        portfolio_df = pd.read_csv(uploaded)
        st.session_state["portfolio_df"] = portfolio_df
        st.dataframe(portfolio_df.head(20))
    if st.button("Analyze Exposure", key="analyze_portfolio"):
        if st.session_state.get("sector_df") is None or st.session_state.get("portfolio_df") is None:
            st.warning("Run scenario simulation and upload portfolio first.")
        else:
            score, breakdown = compute_portfolio_exposure(st.session_state["portfolio_df"], st.session_state["sector_df"])
            st.metric("Weighted Impact Score", f"{score:.3f}")
            st.dataframe(breakdown)

# ---------- AI INSIGHTS ----------
with tab_ai:
    st.subheader("AI-Generated Insights & Risk Alerts")
    user_q = st.text_area("Request AI Analysis",height=120)
    if st.button("Generate AI Insights"):
        sector_df_current = st.session_state.get("sector_df")
        if sector_df_current is not None:
            context = f"Scenario summary: sectors impacted\n{sector_df_current.to_dict(orient='records')}\nUser request: {user_q}"
            out = call_ai_research(st.session_state["ai_history"], context, ai_level)
            st.markdown(out)
            st.session_state["ai_history"].append({"role":"assistant","content":out})
        else:
            st.warning("Run scenario simulation first.")

# ---------- REPORTS ----------
with tab_reports:
    st.subheader("Download Report")
    report_title = st.text_input("Report Title","Katta Scenario Report")
    ai_summary_for_report = st.text_area("AI summary to include",height=120)
    include_portfolio = st.checkbox("Include portfolio exposure")
    if st.button("Create PDF"):
        sector_df_current = st.session_state.get("sector_df")
        portfolio_df_current = st.session_state.get("portfolio_df") if include_portfolio else None
        pdf_bytes = create_pdf_report(report_title,"Scenario","Inputs",sector_df_current,portfolio_df_current,ai_summary_for_report)
        st.download_button("Download PDF",pdf_bytes,file_name="katta_report.pdf",mime="application/pdf")

# ---------- LIVE NEWS ----------
with tab_news:
    st.subheader("Live Market Intelligence & News")
    keyword = st.text_input("News keyword","finance")
    articles = fetch_news(keyword,10)
    for art in articles:
        sentiment = analyze_sentiment(art.get("title","")+art.get("description",""))
        st.markdown(f"**{art['title']}** ({sentiment})")
        st.write(art.get("description",""))
        st.write(f"[Read more]({art['url']})")
        st.write("---")

st.caption("Katta MacroSuite 2.0 — Decision-support only. Not investment advice.")
