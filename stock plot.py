
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

style.use('ggplot')
# def time period
start = dt.datetime(2013,1,1)
end = dt.datetime(2018,3,15)

#import from yhoo finance
#df = web.DataReader('SOXL', 'google', start, end)
#df.to_csv('soxl.csv')

df=pd.read_csv('soxl.csv', parse_dates = True, index_col=0)

df_ohlc = df['Close'].resample('5d').ohlc()
df_volume = df['Volume'].resample('5D').sum()
df_ohlc = df_ohlc.reset_index()
df_ohlc['Date']= df_ohlc['Date'].map(mdates.date2num)

df['ma20'] = df['Close'].rolling(window=20, min_periods=0).mean()
df['ma50'] = df['Close'].rolling(window=50, min_periods=0).mean()
df['ma100'] = df['Close'].rolling(window=100, min_periods=0).mean()
df['ma200'] = df['Close'].rolling(window=200, min_periods=0).mean()

df['bb20u'] = df['ma20'] + df['Close'].rolling(window=20, min_periods=0).std()*2 
df['bb20d'] = df['ma20'] - df['Close'].rolling(window=20, min_periods=0).std()*2

print (df.head())

# subplot for volume 
fig = plt.figure()
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=2,colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values, color = '#0079a3', alpha=0.3)

date = df.index
close = df['Close']

ax1.fill_between(date, close, close[0],where=(close > close[0]), facecolor='g', alpha=0.1)
ax1.fill_between(date, close, close[0],where=(close < close[0]), facecolor='r', alpha=0.1)
ax1.axhline(close[0], color='k', linewidth=3)

labels=(['bb20d','bb20u','ma50','ma100','ma200'])

for label in labels :
    ax1.plot(df.index, df[label], label=label)

ax1.legend()
ax1.grid(True, linestyle ='-',linewidth=1)
plt.suptitle('SOXL', fontsize=12)
plt.xlabel('Date')
ax1.set_ylabel('Price')

plt.show()
