# -*- coding: utf-8 -*-
"""
Created on Fri Mar  24 

@author: Miguel Azuero

Data Transformation and Preprocessing of Financial Data

Downloaded from Yahoo finance using yahoofinancials api

https://github.com/JECSand/yahoofinancials/blob/master/README.rst



https://www.python-graph-gallery.com/basic-time-series-with-matplotlib

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from yahoofinancials import YahooFinancials

tickers = ['^GSPC']

yahoo_financials_info = YahooFinancials(tickers)


daily_gspc_prices = yahoo_financials_info.get_historical_price_data('2016-01-01', '2022-12-31', 'daily')

#daily_gspc_prices.keys()


dict_list = daily_gspc_prices['^GSPC']['prices']


gspc_prices = pd.DataFrame(dict_list)

gspc_prices.head()
gspc_prices.tail()



# Late format in pandas
gspc_prices["formatted_date"] = pd.to_datetime(gspc_prices["formatted_date"])

gspc_prices['time_trend'] = np.arange(len( gspc_prices))


gspc_prices['adjclose_lag1'] = gspc_prices['adjclose'].shift(1)

gspc_prices["returns"]= (gspc_prices['adjclose'] -gspc_prices['adjclose_lag1'])/gspc_prices['adjclose_lag1']
gspc_prices.head()


#Save
#test_path = "C:/Users/migue/Dropbox/JKU EBA Masters/Master Thesis/Data/"

#gspc_prices.to_csv(test_path+"GSPC.csv")

dates_list = gspc_prices["formatted_date"]
adjclose_list = gspc_prices["adjclose"]
returns_list = gspc_prices["returns"]



import matplotlib.dates as mdates


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(dates_list, adjclose_list)
plt.axvline(dt.datetime(2019, 12, 31),color="black",linewidth=0.5,ls=":")
plt.axvline(dt.datetime(2020, 1, 30),color="tab:red",linewidth=0.5,ls="-.")
plt.axvline(dt.datetime(2022, 2, 24),color="tab:red",linewidth=0.5,ls="-.")
plt.xlabel("Year-month")
plt.ylabel("Adj Close")
plt.show()



fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(dates_list, returns_list)
plt.xlabel("Year-month")
plt.ylabel("Returns")
plt.show()

gspc_prices.columns

gspc_prices.dtypes

gspc_prices.returns.describe()


#Boxplot
plt.boxplot(gspc_prices.adjclose)
plt.show()

#Violinplot
plt.violinplot(gspc_prices.adjclose)
plt.show()


#Histogram
plt.hist(gspc_prices.returns,50,density=1)
plt.show()


import seaborn as sns

sns.kdeplot(gspc_prices.returns)
plt.show()


sns.displot(gspc_prices.returns)
plt.show()


gspc_prices.columns



#-------------------------
#------OLS Regression-----
#-------------------------
gspc_prices = gspc_prices.dropna()
gspc_prices.head()

import statsmodels.api as sm
import statsmodels.formula.api as smf

#Using Formula
results = smf.ols('adjclose ~ time_trend + adjclose_lag1', data=gspc_prices).fit()
results.summary()

#Using numpy arrays
columns = ["time_trend","adjclose_lag1"]
selected_df = gspc_prices[columns]

X = selected_df.to_numpy()
X = sm.add_constant(X)

y = gspc_prices["adjclose"]


results = sm.OLS(y, X).fit()

# Inspect the results
print(results.summary())
