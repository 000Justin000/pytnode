import sys
import numpy as np
from yahoo_finance_api2 import share

stocks = ['AAPL', '^IXIC']

stocks_data = []
for stock in stocks:
    STOCK = share.Share(stock)
    STOCK_DATA = STOCK.get_historical(share.PERIOD_TYPE_YEAR, 20, share.FREQUENCY_TYPE_WEEK, 1)
    stocks_data.append(STOCK_DATA['open'])
    stocks_data.append(STOCK_DATA['close'])

T = np.array(STOCK_DATA['timestamp'])
T0 = T/1000 + 9.5*60*60
T1 = T/1000 + 112*60*60

stocks_data = [T0, T1] + stocks_data

np.savetxt("stock_price", np.array(stocks_data).T, fmt = ['%15d', '%15d'] + ['%14.3f', '%14.3f'] * len(stocks), 
        header = "{:>13s}{:>16s}".format("open", "close") + "".join(map(lambda s: "{:>15s}".format(s), sum([[stock+"(open)", stock+"(close)"] for stock in stocks], []))))
