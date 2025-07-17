import streamlit as st
import pandas as pd

@st.cache_data
def load_prices():
    df = pd.read_csv("market_price/data/prices.csv", dtype=str)
    date_col = None
    for col in ("arrival_date","date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.dropna(subset=[col])
            date_col = col
            break
    if not date_col:
        st.error("No date column found")
        st.stop()
    for numcol in ("modal_price","min_price","max_price"):
        if numcol in df.columns:
            df[numcol] = pd.to_numeric(df[numcol], errors="coerce")
    return df, date_col

df_prices, date_col = load_prices()

st.title("📊 Market Price Dashboard")
commodity = st.sidebar.selectbox("Commodity", sorted(df_prices["commodity_name"].dropna().unique()))
markets   = df_prices[df_prices["commodity_name"]==commodity]["market_name"].dropna().unique()
market    = st.sidebar.selectbox("Market", sorted(markets))

df = df_prices[
    (df_prices["commodity_name"]==commodity) &
    (df_prices["market_name"]==market)
].sort_values(date_col)

if df.empty:
    st.warning("No data for selection.")
    st.stop()

latest = df.iloc[-1]
dt = latest[date_col]
st.metric("Latest Date", dt.strftime("%Y-%m-%d"))
st.metric("Modal Price", f"{latest['modal_price']:.2f} ₹")
st.line_chart(df.set_index(date_col)["modal_price"])
