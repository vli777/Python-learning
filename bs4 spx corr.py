import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib import cm
import time
import numpy as np

import bs4 as bs
import pickle
import requests
import os

def spx():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text,'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500.pickle",'wb') as f:
        pickle.dump(tickers,f)
    return tickers

def get_data (reload_sp500 = False): #set True for refresh
    if reload_sp500:
        tickers = spx()
    else:
        with open('sp500.pickle','rb') as f:
            tickers = pickle.load(f)
            tickers = sorted(tickers)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    # def time period
    start = dt.datetime(2013,1,1)
    end = dt.datetime(2018,3,15)

    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'google', start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

def compile_data():
    #converters = {0: lambda s: float(s.strip('"'))}
    df = pd.read_csv('wiki_list.csv',skiprows=1)
    tickers = df.iloc[:,1].values
  
    main_df = pd.DataFrame()
    
    for ticker in tickers:
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Close':ticker}, inplace=True)
        df.drop(['Open','High','Low','Volume'],axis=1, inplace=True)
        #df['Date']=time.strftime('%Y-%m-%d')

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

    print(main_df.head())
    #main_df = main_df.dropna(how='all')
    main_df.to_csv('sp500_joined_close.csv')

style.use('dark_background')

def corr():
    df = pd.read_csv('sp500_joined_close.csv')
    corr = df.corr()
    print(corr.head())

    y1 = corr.values
    fig1 = plt.figure()
    ax1= fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(y1,cmap=plt.cm.cool)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(y1.shape[1])+.5, minor = False)
    ax1.set_yticks(np.arange(y1.shape[0])+.5, minor = False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()

    col_labels = corr.columns
    row_labels = corr.index

    ax1.set_xticklabels(col_labels)
    ax1.set_yticklabels(row_labels)

    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    plt.tight_layout()
    plt.show()

# main
# Notes: Google Finance will sometimes block requests for awhile and it breaks if data is missing. 
# Historical prices are now deprecated so get_data and compile_data are commented out.

#get_data()
#compile_data()
#corr()


