import streamlit as st
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
import re

st.set_page_config(page_title="Stock Recommendation Dashboard", page_icon="", layout="wide", initial_sidebar_state="expanded")

PRIMARY_COLOR = "#1f3b73"
SECONDARY_COLOR = "#4a4a4a"
HIGH_RISK_COLOR = "#ff4d4d"
MEDIUM_RISK_COLOR = "#ffa500"
LOW_RISK_COLOR = "#33cc33"

st.markdown(f"""
<style>
:root {{ --primary: {PRIMARY_COLOR}; --text: {SECONDARY_COLOR}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color: var(--primary); }}
.stSidebar .sidebar-content {{ background: linear-gradient(160deg, {PRIMARY_COLOR} 0%, #264a9a 100%); color: white; }}
.card {{ background: #fff; padding: 16px; border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 12px; transition: transform 150ms ease, box-shadow 150ms ease; border: 1px solid rgba(0,0,0,0.06); }}
.card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 18px rgba(0,0,0,0.12); }}
.kpi {{ font-size: 22px; font-weight: 700; color: var(--primary); margin: 4px 0 0 0; }}
.kpi-sub {{ font-size: 13px; color: #777; margin: 0; }}
</style>
""", unsafe_allow_html=True)

st.title("Stock Recommendation Dashboard")

# Data files
cards_path = Path("/Users/ishanlahiru/Documents/invesment-portfolio-analysis/processed-rag/cards.json")
if not cards_path.exists():
    st.error(f"JSON file not found: {cards_path}")
    st.stop()
with open(cards_path, "r", encoding="utf-8") as f:
    stocks_data = json.load(f)
symbols = [s["symbol"] for s in stocks_data]

# Ticker normalization
def base_key(s: str) -> str:
    return re.sub(r'[^A-Z0-9]+', '', s.upper())
def extract_symbol_from_header(header: str) -> str | None:
    if not header: return None
    m = re.search(r'\*\*([A-Z0-9\.\-:]+)\b', header, flags=re.IGNORECASE)
    if m: return base_key(m.group(1))
    m = re.search(r'^\s*#+\s+([A-Z0-9\.\-:]+)\b', header, flags=re.IGNORECASE)
    if m: return base_key(m.group(1))
    m = re.search(r'([A-Z0-9]{1,8})(?=\s*\()', header)
    if m: return base_key(m.group(1))
    return None

# Load financial_analysis.json → summary_map
summary_path = Path("/Users/ishanlahiru/Documents/invesment-portfolio-analysis/processed-rag/financial_analysis.json")
summaries = []
summary_map: dict[str, str] = {}
if summary_path.exists():
    with open(summary_path, "r", encoding="utf-8") as f:
        try:
            summaries = json.load(f) or []
        except Exception:
            summaries = []
for item in summaries:
    head = item.get("summary", "") or ""
    details = item.get("details", []) or []
    sym_key = extract_symbol_from_header(head)
    if sym_key:
        block = head + "\n" + "\n".join(details)
        summary_map[sym_key] = block
def find_summary_block_fallback(sym: str) -> str | None:
    skey = base_key(sym)
    for item in summaries:
        head = (item.get("summary", "") or "")
        details = item.get("details", []) or []
        hay = base_key(head)
        if skey in hay or hay in skey:
            return head + "\n" + "\n".join(details)
    return None

# Sidebar controls
st.sidebar.header("Controls")
symbol = st.sidebar.selectbox("Select Stock", symbols, index=0)
stock = next((s for s in stocks_data if s["symbol"] == symbol), None)

# Assistant navigation: switch to registered page under pages/
st.sidebar.divider()
st.sidebar.subheader("Assistant")
if st.sidebar.button("🤖 Open Chatbot", use_container_width=True):
    st.switch_page("pages/Chatbot.py")  # must exist under frontend/pages/Chatbot.py

# Alternative link style (internal):
# st.sidebar.page_link("pages/Chatbot.py", label="🤖 Open Chatbot", icon="💬", use_container_width=True)

# Load history
@st.cache_data(ttl=3600, show_spinner=False)
def load_history(sym: str) -> pd.DataFrame:
    hist = yf.Ticker(sym).history(period="6mo")
    hist.reset_index(inplace=True)
    return hist
hist = load_history(symbol)

# Tabs (Chart at top of Overview)
tab_overview, tab_news, tab_data = st.tabs(["Overview", "News", "Data"])

with tab_overview:
    st.subheader(f"{symbol} Price Trend (Last 6 Months)")
    show_markers = st.toggle("Show markers", value=True, help="Toggle point markers for the price series.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], mode="lines+markers" if show_markers else "lines",
                             name="Close", line=dict(color=PRIMARY_COLOR, width=2), marker=dict(size=4)))
    fig.update_layout(template="plotly_white", font=dict(color=SECONDARY_COLOR), legend=dict(x=0.01, y=0.99),
                      xaxis_title="Date", yaxis_title="Price ($)", hovermode="x unified",
                      margin=dict(l=20, r=20, t=40, b=20))
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(xaxis=dict(rangeselector=dict(buttons=list([
        dict(count=7, label="7d", step="day", stepmode="backward"),
        dict(count=1, label="1m", step="month", stepmode="backward"),
        dict(count=3, label="3m", step="month", stepmode="backward"),
        dict(step="all"),
    ]))))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"{symbol} Overview")
    col1, col2, col3 = st.columns(3)
    last_traded_date = datetime.strptime(stock['lastTradedTime'].split(' ')[0], '%Y-%m-%d')
    col1.markdown(f"<div class='card'><h4>Current Price</h4><p class='kpi'>${stock['current_price']:.2f}</p><p class='kpi-sub'>{last_traded_date.date()}</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><h4>Predicted 5-Day</h4><p class='kpi'>${stock['pred_5d_price']:.2f}</p><p class='kpi-sub'>{(last_traded_date + timedelta(days=5)).date()}</p></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><h4>Predicted 10-Day</h4><p class='kpi'>${stock['pred_10d_price']:.2f}</p><p class='kpi-sub'>{(last_traded_date + timedelta(days=10)).date()}</p></div>", unsafe_allow_html=True)

    st.subheader("Risk Overview")
    risk = stock.get("risk", {})
    risk_level = risk.get("RiskLevel", "Medium")
    advice = risk.get("InvestmentAdvice", "")
    if risk_level.lower() == "high":
        color = HIGH_RISK_COLOR; text = "High risk: Price may fluctuate a lot. Suitable for aggressive investors only."
    elif risk_level.lower() == "medium":
        color = MEDIUM_RISK_COLOR; text = "Moderate risk: Price may vary moderately. Suitable for balanced portfolios."
    else:
        color = LOW_RISK_COLOR; text = "Low risk: Price is relatively stable. Suitable for conservative investors."
    st.markdown(f"<div style='background-color:{color}; padding:12px; border-radius:12px; color:white;'>{text}</div>", unsafe_allow_html=True)
    st.caption(f"Advice: {advice}")

    selected_key = base_key(symbol)
    summary_block = summary_map.get(selected_key) or find_summary_block_fallback(symbol)
    if summary_block:
        st.markdown(summary_block)
    else:
        st.warning(f"Summary not available for {symbol} in financial_analysis.json")

with tab_news:
    st.subheader("Latest News")
    if stock.get("news"):
        for news_item in stock["news"]:
            st.markdown(f"- {news_item}")
    else:
        st.info("No news available.")

with tab_data:
    st.subheader(" Full JSON Data")
    with st.expander("Show JSON", expanded=False):
        st.json(stock)
