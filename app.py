# Libraries 
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, date
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid



def build_sidebar():

    st.image('imagens/image.png')

    # Obtendo os Tickers
    ticker_df = pd.read_csv('IBOVDia_28-03-24.csv', encoding='latin-1', header=1, sep=';', index_col= False)
    tickers_series = ticker_df['Código'][:-2]
    tickers = st.multiselect(label = 'Selecione O ticker', options = tickers_series)
    tickers = [t+'.SA' for t in tickers]

    # Datas
    start_date = st.date_input('De', format='DD/MM/YYYY', value=date.today())
    end_date = st.date_input('Até',format='DD/MM/YYYY', value=date.today())

    if tickers:
        prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if len(tickers) == 1:
            prices = prices.to_frame()
            prices.columns = [tickers[0].rstrip(".SA")]
        prices.columns = prices.columns.str.rstrip(".SA")
        # prices['IBOV'] = yf.download("^BVSP", start=start_date, end=end_date)["Adj Close"]
        return tickers, prices
    return None,None



def build_main(tickers, prices):
    # print('Prices',prices)
    weights = np.ones(len(tickers))/len(tickers)
    prices['portfolio'] = prices @ weights
    st.dataframe(prices)
    norm_prices = 100 * prices / prices.iloc[0]
    returns = prices.pct_change()[1:]
    vols = returns.std()*np.sqrt(252)
    rets = (norm_prices.iloc[-1]-100)/100

    my_grid = grid(5, 5, 5, 5, 5, 5, vertical_align='top')
    for ativo in prices.columns:
        c = my_grid.container(border=True)
        c.subheader(ativo, divider = 'red')
        col1, col2, col3 = c.columns(3)
        if ativo != 'portfolio' :
            col1.image(f'https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{ativo}.png', width=85)
        col2.metric(label="retorno", value=f"{rets[ativo]:.0%}")
        col3.metric(label="volatilidade", value=f"{vols[ativo]:.0%}")
        style_metric_cards(background_color='rgba(255,255,255,0)')
            

    pass

st.set_page_config(layout = 'wide')

with st.sidebar:
    tickers, prices = build_sidebar()

st.title('Markowwitz Portfolio')
if tickers:
    build_main(tickers, prices)
