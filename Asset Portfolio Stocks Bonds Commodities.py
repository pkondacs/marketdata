# Databricks notebook source
# MAGIC %md
# MAGIC # Install some necessary packages

# COMMAND ----------

# activates the matplotlib to be displayed inline
%matplotlib inline 
%pip install pandas-datareader
%pip install quandl
%pip install python-dotenv
%pip install yfinance

# COMMAND ----------

# MAGIC %md ##Using the yfinance package

# COMMAND ----------

import yfinance as yf

# Fetch stock price data for Apple and Microsoft
aapl = yf.download("AAPL")
msft = yf.download("MSFT")
natgas = yf.download("CL=F")
# Concatenate Apple and Microsoft stock price dataframes
combined_df = pd.concat([aapl, msft, natgas], axis=1)
combined_df.head()

# COMMAND ----------

import matplotlib.pyplot as plt

# Plot the combined stock price data
combined_df['Close'].plot(figsize=(10, 6))

# Set the chart title and labels
plt.title('Stock Price Comparison')
plt.xlabel('Date')
plt.ylabel('Closing Price')

# Display the chart
plt.show()

# COMMAND ----------

# relative change in prices, price at t divided by the first price row
returns_na = (historical/historical.iloc[0]) #.fillna(method='backfill')
returns = (historical/historical.iloc[0]).fillna(method='backfill') 
_=returns_na.plot()

# COMMAND ----------

prf_prices = pd.concat((stock_hist(symbol) for symbol in prf_stocks), axis = 1, keys = prf_stocks)
prf_returns = (prf_prices.pct_change() + 1)[1:]
log_returns = np.log(prf_returns)
corr = log_returns.corr()
sns.heatmap(corr, annot=True)

# COMMAND ----------




# COMMAND ----------

aapl.Close.plot()

# COMMAND ----------

visa = web.DataReader("V.US","quandl","2015-01-01", "2016-01-01", api_key = "FX-yxABrQh8R4VCJu8_q")
visa.Close.plot()
# to adjust for the stock split
visa.AdjClose.plot()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Utility: get adjusted close, cache recent years

# COMMAND ----------

cached_data = {}
#def stock_hist (symbol, start=None, end=None): 
#    """Convenience function to get cached data"""
#    start = start if start else default_start
#    end = end if end else default_end
#    if not symbol in cached_data:
#        cached_data[symbol] = web.DataReader(symbol + ".US", "quandl", all_data_start, all_data_end, api_key = "FX-yxABrQh8R4VCJu8_q")
#        print(f"Loaded {symbol} num values = {len(cached_data[symbol])}")
#    return cached_data[symbol][start:end]
def stock_hist (symbol, start=None, end=None):
    start = start if start else default_start
    end = end if end else default_end
    cached_data[symbol] = web.DataReader(symbol + ".US", "quandl", start, end , api_key = "FX-yxABrQh8R4VCJu8_q")['AdjClose']
    return cached_data[symbol]

# COMMAND ----------

start_dt = "2016-01-01"
end_dt = "2016-01-10"
web.DataReader("AAPL.US", "quandl", start = start_dt, end = end_dt , api_key = "FX-yxABrQh8R4VCJu8_q")['AdjClose']
# result for one stock
stock_hist("AAPL").head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Look at a basket of stocks

# COMMAND ----------

tickers_list = ['AAPL','FB','GOOG']
N = len(tickers_list)
historical = pd.concat((stock_hist(symbol) for symbol in tickers_list), axis = 1, keys = tickers_list)
_=historical.plot() # underscore can be used if don't want to assign the value to anything specific
historical.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an equally weighted portfolio

# COMMAND ----------

# the glitch is caused by missing values, can be avoided by including the fillna method in the previous step
returns_na['PORTFOLIO'] = returns_na.iloc[:,0:N].sum(axis=1) / N
# iloc below: it can be omitted. It slices all rows (as defined by the row index in iloc ":"), and 0:N columns, which is basically all columns
# axis = 1 means sum along the columns, N was defined above as the length of ticker list
returns['PORTFOLIO'] = returns.iloc[:,0:N].sum(axis=1) / N
returns_na.plot()

# COMMAND ----------

symbols = ['AAPL','TSLA','FB','IBM','GOOG']
prices = [stock_hist(symbol) for symbol in symbols]
unit_pos = [p / p[-5] for p in prices]
df = pd.DataFrame(unit_pos)

basket = sum(u for u in unit_pos) / len(unit_pos)
df = pd.DataFrame(basket)
df

# COMMAND ----------

# **active is a **kwargs
# **kwargs allows you to pass keyworded variable length of arguments to a function.
def diversicheck(symbols, start_day=0, **active):
    # The strptime() method creates a datetime object from the given string.
    start = datetime.datetime.strptime(default_start, "%Y-%m-%d") + datetime.timedelta(start_day)
    end = start + datetime.timedelta(days=365)
    # filtered is a kind of check if symbol is valid
    filtered = [symbol for symbol in symbols if active.get(symbol, True)]
    prices = [stock_hist(symbol) for symbol in filtered]
    unit_pos = [p / p[-1-start_day] for p in prices]
    basket = sum(u for u in unit_pos) / len(unit_pos)
    for p in unit_pos:
        p.plot(color='b', alpha=0.3)
    basket.plot(figsize=(20,10))
    print(f"Basket from {start} to {end}")

# COMMAND ----------

# Notes: 
# To use interact, you need to define a function that you want to explore. Hence diversicheck is created in the step before.
# fixed() is used within interact which fixes arguments to specific non-modifiable values.

prf_stocks = ['AAPL','TSLA','FB','IBM','GOOG']
active = dict(zip(prf_stocks,[True]*len(prf_stocks)))
_= interact(diversicheck, 
    symbols= fixed(prf_stocks),
    start_day = IntSlider(min=0,max=365, step=1,
                    description='Start date:',
                    disabled=False,
                    continuous_update=True,
                    orientation='horizontal',
                    readout=True,
                    readout_format='d'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate forwards

# COMMAND ----------

vols = log_returns.std() * np.sqrt(252)
avg_returns = (prf_returns-1).mean()
fig, ax = plt.subplots()
ax.scatter(vols, avg_returns)
for i, txt in enumerate(prf_stocks):
    ax.annotate(txt, (vols[i],avg_returns[i]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bootstrap (Monte-carlo simulation)

# COMMAND ----------

# k=60 Returns a list with 60 items, ie. 60 samples are taken 1000 times (60,000 samples in total)
simulated = pd.DataFrame([((prf_returns.iloc[random.choices(
    range(len(prf_returns)),k=60)]).mean(axis=1)).cumprod().values
    for x in range(1000)]).T
simulated.head()

# COMMAND ----------

simulated.plot(legend=False, linewidth=1, alpha=0.1, color="blue")

# COMMAND ----------

simulated.quantile([0.05,0.5,0.95], axis=1).T.plot()
