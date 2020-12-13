# Standard packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
import random


# Collect data from yahoo finance
from pandas_datareader.data import DataReader

# For time stamps
from datetime import datetime


# Configuration class
class config:
    # Select time length in years, months and days
    yrs = 3
    mths = 0
    dys = 0

    # Starting of with google stocks
    stock_names_compare = ['GOOG', 'AAPL', 'MSFT', 'AMZN']
    stock_names = ['GOOG']

    moving_averages = [int(np.floor(365 / 2)), int(np.floor(365 / 4)), 30, 14, 7]

    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
              'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
              'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
              'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
              'rgb(188, 189, 34)', 'rgb(23, 190, 207)']


# Start and end times for the time-series
def get_timestamps(yrs=0, mths=0, dys=0):
    '''
        Input:  yrs - number of years back in time to track
                mths - number of months back in time to track
                dys - number of days back in time to track

        Output: start_time and end_time as a list
    '''
    end_time = datetime.now()
    start_time = datetime(end_time.year - yrs, end_time.month - mths, end_time.day - dys)

    return [start_time, end_time]


# Collecting data from yahoo finance to dataframe

def collect_data(timestamps, stock_name, moving_averages=None, include_gain=True):
    '''
        Input: timestamps - start and end time of the time period to track time
               stock_name - code of the stock from the specific company
               moving_averages - list of the time period to compute moving averages (default None)
               invlude_gain - boolean if include the daily change of the stock price (default True)

        Output: Dataframe of the stock for the selected time period
    '''

    locals()[stock_name] = DataReader(stock_name, 'yahoo', timestamps[0], timestamps[1])
    company_stock = [vars()[stock_name]]
    company_stock_name = [stock_name]
    for comp, name in zip(company_stock, company_stock_name):
        comp["Company stock name"] = name

    df_stock = pd.concat(company_stock, axis=0)

    if moving_averages is not None:
        for ma in moving_averages:
            if 3 * ma < len(df_stock):
                column_name = f"{ma} days MA"
                df_stock[column_name] = df_stock['Adj Close'].rolling(ma).mean()

    if include_gain:
        change = (df_stock['Adj Close'] / df_stock['Open']).tolist()
        df_stock['Change %'] = change
        df_stock['Daily Return'] = df_stock['Adj Close'].pct_change()

    return df_stock


# Plot functions
def plot_closing(df, moving_averages=True, intervals=None):
    '''
        Input: df - dataframe of the stock
               intervals - list of ints of time periods to split the dataframe

        Output: Figure of closing price of the stock

    '''

    colors = config.colors
    fig = go.Figure()

    x = [str(df.index[i]).split()[0] for i in range(len(df))]
    fig.add_trace(
        go.Scatter(x=x, y=df['Adj Close'], mode='lines', line_color=colors[0], line_width=3,
                   name='Adjusted Closing Price'))
    i_color = 1
    for c in df.columns:

        if c.endswith('MA'):
            fig.add_trace(go.Scatter(x=x,
                                     y=df[c],
                                     mode='lines',
                                     line_color=colors[i_color],
                                     line_width=2,
                                     name=c))
            i_color += 1

    fig.update_layout(showlegend=True)
    fig.update_layout(title=dict(text=f'"{df["Company stock name"][0]}" stocks from {x[0]} to {x[len(df) - 1]}',
                                 xanchor='auto'),
                      xaxis=go.layout.XAxis(
                          title=go.layout.xaxis.Title(
                              text="Date")),
                      yaxis=go.layout.YAxis(
                          title=go.layout.yaxis.Title(
                              text="Adjusted closing price USD ($)"))
                      )
    return fig


def plot_gain(df):
    '''
        Input: df - Dataframe of the stock

        Output: Histograms of the daily returns and the daily change in percentage of the stock

    '''

    xDR = np.arange(min(df['Daily Return'].dropna().tolist()),
                    max(df['Daily Return'].dropna().tolist()),
                    len(df['Daily Return'].dropna()))
    xC = np.arange(min(df['Change %'].dropna().tolist()),
                   max(df['Change %'].dropna().tolist()),
                   len(df['Change %'].dropna()))

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f'Daily Return of stock "{df["Company stock name"][0]}"',
                                        f'Daily change in % of stock "{df["Company stock name"][0]}"'))

    fig.add_trace(go.Histogram(x=df['Daily Return'].dropna(), marker_color='#330C73', opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=xDR, y=df['Daily Return'].dropna(), mode='lines', line_color='#330C73'), row=1, col=1)

    fig.add_trace(go.Histogram(x=df['Change %'].dropna(), marker_color='#330C73', opacity=0.8), row=1, col=2)
    fig.add_trace(go.Scatter(x=xC, y=df['Change %'].dropna(), mode='lines', line_width=5, line_color='#330C73'), row=1,
                  col=2)

    fig.update_layout(showlegend=False)

    fig.update_xaxes(title_text="Price USD ($)", row=1, col=1)
    fig.update_xaxes(title_text="Percentage %", row=1, col=2)

    fig.update_yaxes(title_text="Counts", row=1, col=1)
    fig.update_yaxes(title_text="Counts", row=1, col=2)

    fig.update_layout(
        bargap=0.1,
        bargroupgap=0.1
    )

    return fig


def compare_stocks(dfs, timestamps):
    '''
        Input: dfs - list of dataframes for the different stocks to be compared
               timestamps - list of start and end time of the time period to be analysed

        Output: daily_returns - dataframe of the daily returns of all the stocks
                fig1 - correlation grid of the adjusted closing price of all the stocks
                fig2 - correlation matrix of the daily returns of all the stocks

    '''

    closing = DataReader(dfs, 'yahoo', timestamps[0], timestamps[1])['Adj Close']
    daily_returns = closing.pct_change()
    x = [str(daily_returns.dropna().index[i]).split()[0] for i in range(len(daily_returns.dropna()))]

    fig1 = sns.PairGrid(daily_returns.dropna(), )
    fig1.map_upper(plt.scatter, color='#330C73')

    fig1.map_lower(sns.kdeplot, cmap='RdPu_r')

    fig1.map_diag(plt.hist, bins=30)
    fig1.fig.suptitle(
        f'Graphical correlation between the different stocks for the daily returns from {x[0]} to {x[len(x) - 1]}',
        fontsize=18, y=1.03)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(closing.corr(), annot=True, cmap='PuBu', ax=ax1)
    sns.heatmap(daily_returns.corr(), annot=True, cmap='PuRd', ax=ax2)

    fig2.suptitle(
        f'Correlation between the different stocks for the closing price and the daily returns from {x[0]} to {x[len(x) - 1]}',
        fontsize=18)
    ax1.set_title('Adjusted Closing Price USD ($)')
    ax2.set_title('Daily returns USD ($)')
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax1.set_ylabel('')
    ax2.set_ylabel('')

    return daily_returns, fig1, fig2





