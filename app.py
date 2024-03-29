# Libraries 
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

import plotly
import cufflinks

# (*) To communicate with Plotly's server, sign in with credentials file
import chart_studio.plotly as py  

# (*) Useful Python/Plotly tools
import plotly.tools as tls   

# (*) Graph objects to piece together plots
from plotly.graph_objs import *

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, date
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid

import functions_utils



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

    # Numero de portfolios

    num_port = st.number_input('Simulação de Portfolios')

    if tickers:
        prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if len(tickers) == 1:
            prices = prices.to_frame()
            prices.columns = [tickers[0].rstrip(".SA")]
        prices.columns = prices.columns.str.rstrip(".SA")
        prices['IBOV'] = yf.download("^BVSP", start=start_date, end=end_date)["Adj Close"]
        return tickers, prices, num_port
    return None,None,None



def build_main(tickers, prices, number_port):
    
    weights = np.ones(len(tickers))/len(tickers)
    prices['portfolio'] = prices.drop("IBOV", axis=1) @ weights
    # st.dataframe(prices)
    norm_prices = 100 * prices / prices.iloc[0]
    returns = prices.pct_change()[1:]
    vols = returns.std()*np.sqrt(252)
    rets = (norm_prices.iloc[-1]-100)/100

    my_grid = grid(5, 5, 5, 5, 5, 5, vertical_align='top')
    for ativo in prices.columns:
        c = my_grid.container(border=True)
        c.subheader(ativo, divider = 'red')
        col1, col2, col3 = c.columns(3)
        if ativo != 'portfolio' and ativo != 'IBOV' :
            col1.image(f'https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{ativo}.png', width=85)
        col2.metric(label="retorno", value=f"{rets[ativo]:.0%}")
        col3.metric(label="volatilidade", value=f"{vols[ativo]:.0%}")
        style_metric_cards(background_color='rgba(255,255,255,0)')
    col3, col4, col5 = st.columns(3, gap='large')

    with col3:
        st.subheader('Desempenho Relativo')
        st.line_chart(norm_prices, height=600)

    with col4:
        st.subheader("Risco-Retorno")
        fig = px.scatter(
            x=vols,
            y=rets,
            text=vols.index,
            color=rets/vols,
            color_continuous_scale=px.colors.sequential.Bluered_r
        )
        fig.update_traces(
            textfont_color='white', 
            marker=dict(size=45),
            textfont_size=10,                  
        )
        fig.layout.yaxis.title = 'Retorno Total'
        fig.layout.xaxis.title = 'Volatilidade (anualizada)'
        fig.layout.height = 600
        fig.layout.xaxis.tickformat = ".0%"
        fig.layout.yaxis.tickformat = ".0%"        
        fig.layout.coloraxis.colorbar.title = 'Sharpe'
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        st.subheader('Dispersão portfolios')

        n_portfolios = int(number_port)
        means, stds = np.column_stack([
        functions_utils.random_portfolio(returns) 
        for _ in range(n_portfolios)
        ])
        fig = px.scatter(
            x=stds.ravel(),
            y=means.ravel(),
        )
        fig.layout.yaxis.title = 'Média'
        fig.layout.xaxis.title = 'Variancia'
        fig.layout.height = 600
        fig.layout.xaxis.tickformat = ".0%"
        fig.layout.yaxis.tickformat = ".0%"        
        # fig.layout.coloraxis.colorbar.title = 'Sharpe'
        st.plotly_chart(fig, use_container_width=True)

        # fig = plt.figure()
        # plt.plot(stds, means, 'o', markersize=5)
        # plt.xlabel('std')
        # plt.ylabel('mean')
        # plt.title('Mean and standard deviation of returns of randomly generated portfolios')
        # py.iplot_mpl(fig, filename='mean_std', strip_style=True)

        

    

st.set_page_config(layout = 'wide')

with st.sidebar:
    tickers, prices, number_port = build_sidebar()

st.title('Markowwitz Portfolio')
if tickers:
    build_main(tickers, prices, number_port)
