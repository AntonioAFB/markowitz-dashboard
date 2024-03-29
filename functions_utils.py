import requests 
import pandas as pd

import yfinance as yf
from datetime import date
import numpy as np

ticker_df = pd.read_csv('IBOVDia_28-03-24.csv', encoding='latin-1', header=1, sep=';', index_col= False)
ticker_df['Código'][:-2]


tickers = ['AZUL4.SA','PETR4.SA']
start_date = date(2024,3,14)
end_date = date.today()
prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
returns = prices.pct_change()[1:]

n_portfolios = 500
means, stds = np.column_stack([
random_portfolio(returns) 
for _ in range(n_portfolios)
])
import plotly.express as px
fig = px.scatter(
    x=means.ravel(),
    y=stds.ravel,
)
fig.layout.yaxis.title = 'Média'
fig.layout.xaxis.title = 'Variancia'
fig.layout.height = 600
fig.layout.xaxis.tickformat = ".0%"
fig.layout.yaxis.tickformat = ".0%"        
fig.layout.coloraxis.colorbar.title = 'Sharpe'


# prices['portfolio'] = prices @ weights



# from zipline.utils.factory import load_bars_from_yahoo
# end = pd.Timestamp.utcnow()
# start = end - 2500 * pd.tseries.offsets.BDay()

# data = load_bars_from_yahoo(stocks=['IBM', 'GLD', 'XOM', 'AAPL', 
#                                     'MSFT', 'TLT', 'SHY'],
#                             start=start, end=end)


def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)
