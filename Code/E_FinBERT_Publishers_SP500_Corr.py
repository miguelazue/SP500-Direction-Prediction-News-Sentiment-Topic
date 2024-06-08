"""
Created on Thu Apr 13 14:24:23 2023

colors link
https://matplotlib.org/stable/gallery/color/named_colors.html


@author: migue
"""



import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

import time
from datetime import datetime, timedelta

import seaborn as sn
import matplotlib.pyplot as plt

import os

#-------------------------
#-----Path Definition-----
#-------------------------


# Use the current working directory or define the path for the working directory
current_dir = os.getcwd()  # Use the current working directory
current_dir = "C:/Users/migue/Dropbox/JKU EBA Masters/Master Thesis/"  # Define the path for the working directory

consolidated_data_path = os.path.join(current_dir,'Data/consolidated_data/')
consolidated_results_path = os.path.join(current_dir,'Data/consolidated_results/')
categories_data_path = os.path.join(current_dir,'Data/sa_publishers_category/')
topics_sa_path = os.path.join(current_dir,'Data/sa_top_topics/')

# Heavy Data directory
heavy_data_path = "C:/Users/migue/Downloads/Thesis Data/"
parquet_path = os.path.join(heavy_data_path,'FilteredNewsParquets/')
ebf_parquet_path = os.path.join(heavy_data_path,'FilteredNewsParquets/EBF_News/')
ebf_sa_path = os.path.join(heavy_data_path,'SentimentAnalysisEBF/')
topics_path = os.path.join(heavy_data_path,'BERTopics/')

#-------------------------
#---Load Financial Data---
#-------------------------

gspc_data = pd.read_csv(consolidated_data_path+"GSPC.csv")


selected_columns = ["formatted_date","time_trend","adjclose","returns"]
gspc_df = gspc_data[selected_columns].copy()

gspc_df["formatted_date"] = pd.to_datetime(gspc_df["formatted_date"])

gspc_df['year'] = gspc_df['formatted_date'].dt.year
gspc_df['month'] = gspc_df['formatted_date'].dt.month

#gspc_sample = gspc_df[(gspc_df["year"]==2017)&(gspc_df["month"]==10)]
gspc_sample = gspc_df[(gspc_df["year"]==2016)]
gspc_sample = gspc_df.copy()
#gspc_sample.head()
#gspc_sample.tail()


#-------------------------
#-Load SA Categories Data-
#-------------------------

#LOAD

#sa_summary = pd.read_csv(categories_data_path+"sa_publishers_category_201710.csv")
#sa_summary['date'] = pd.to_datetime(sa_summary['date'], format='%Y-%m-%d')
#sa_summary.dtypes


#Iterate over periods
#years = [2017,2018,2019]
years =[2016]
months = [1,2,3,4,5,6,7,8,9,10,11,12]
end_yearmonth = "201701"

years = [2016,2017,2018,2019]
end_yearmonth = "202001"
#end_yearmonth = "202004"


sa_summary = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'),
                   'EBF': pd.Series(dtype='float64'),
                   'GEN': pd.Series(dtype='float64'),
                   'POL': pd.Series(dtype='float64'),
                   'ENT': pd.Series(dtype='float64')
                   })



break_flag = False


#NEW FOR
for year in years:
    for month in months:
        yearmonth = str(year*100+month)


        #break month loop
        if yearmonth == end_yearmonth:
            break_flag = True
            break


        current_df = pd.read_csv(categories_data_path+"sa_publishers_category_"+yearmonth+".csv",index_col=0)
        sa_summary = pd.concat([sa_summary, current_df], ignore_index=True, sort=False)
    #break year loop
    if break_flag == True:
        break

sa_summary['date'] = pd.to_datetime(sa_summary['date'], format='%Y-%m-%d')

sa_summary = sa_summary.sort_values(by=['date'])
sa_summary = sa_summary.reset_index(drop=True)


sa_summary.head()
sa_summary.tail()


sa_summary.date.dt.year.value_counts()

sa_summary.describe()

#-------------------------
#--------Joining Data-----
#-------------------------


sa_and_prices = pd.merge(sa_summary,gspc_sample, left_on='date', right_on='formatted_date', how='inner')


#test_path = "C:/Users/migue/Downloads/"

#sa_and_prices.to_csv(test_path+"my_test.csv")

sa_and_prices.head()
sa_and_prices.tail()




#NORMALIZATION

sa_and_prices["adjclose_z"] = (sa_and_prices["adjclose"] - sa_and_prices["adjclose"].mean()) / sa_and_prices["adjclose"].std()
sa_and_prices["returns_z"] = (sa_and_prices["returns"] - sa_and_prices["returns"].mean()) / sa_and_prices["returns"].std()
sa_and_prices["EBF_z"] = (sa_and_prices["EBF"] - sa_and_prices["EBF"].mean()) / sa_and_prices["EBF"].std()
sa_and_prices["GEN_z"] = (sa_and_prices["GEN"] - sa_and_prices["GEN"].mean()) / sa_and_prices["GEN"].std()
sa_and_prices["POL_z"] = (sa_and_prices["POL"] - sa_and_prices["POL"].mean()) / sa_and_prices["POL"].std()
sa_and_prices["ENT_z"] = (sa_and_prices["ENT"] - sa_and_prices["ENT"].mean()) / sa_and_prices["ENT"].std()

#LAG VARIABLES

sa_and_prices['adjclose_z_lag'] = sa_and_prices['adjclose_z'].shift(1)
sa_and_prices['returns_z_lag'] = sa_and_prices['returns_z'].shift(1)
sa_and_prices['EBF_z_lag'] = sa_and_prices['EBF_z'].shift(1)
sa_and_prices['GEN_z_lag'] = sa_and_prices['GEN_z'].shift(1)
sa_and_prices['POL_z_lag'] = sa_and_prices['POL_z'].shift(1)
sa_and_prices['ENT_z_lag'] = sa_and_prices['ENT_z'].shift(1)

#sa_and_prices = sa_and_prices.dropna()
sa_and_prices.head()



#-------------------------
#--Correlation Analysis---
#-------------------------

correlation_columns = ['returns','adjclose',
                       'time_trend', 'year', 'month',
                       'EBF', 'GEN', 'POL', 'ENT']

correlation_columns2 = ['returns','adjclose',
                       'time_trend', 'year', 'month',
                       'EBF', 'GEN', 'POL', 'ENT',
                       'adjclose_z_lag', 'returns_z_lag', 
                       'EBF_z_lag', 'GEN_z_lag', 'POL_z_lag','ENT_z_lag']


#Selection of columns and order
corr_matrix = sa_and_prices[correlation_columns].corr()

heatmap_plot = sn.heatmap(corr_matrix, annot=True,vmin=-1, vmax=1,cmap= "icefire_r")
heatmap_plot.set(xlabel="", ylabel="")
plt.show()

# creating mask
# mask = np.triu(np.ones_like(corr_matrix)) 
# Better with lower triangle
mask = np.tril(np.ones_like(corr_matrix))

heatmap_plot = sn.heatmap(corr_matrix, annot=True,vmin=-1, vmax=1,cmap= "icefire_r",mask=mask)
heatmap_plot.set(xlabel="", ylabel="")
plt.show()


#Selection of columns and order
corr_matrix2 = sa_and_prices[correlation_columns2].corr()

heatmap_plot2 = sn.heatmap(corr_matrix2, annot=True,vmin=-1, vmax=1,cmap= "icefire_r")
heatmap_plot2.set(xlabel="", ylabel="")
plt.show()


# creating mask
# mask = np.triu(np.ones_like(corr_matrix)) 
# Better with lower triangle
mask = np.tril(np.ones_like(corr_matrix2))

heatmap_plot2 = sn.heatmap(corr_matrix2, annot=True,vmin=-1, vmax=1,cmap= "icefire_r",mask=mask)
heatmap_plot2.set(xlabel="", ylabel="")
plt.show()




#-------------------------
#---------PLOTS-----------
#-------------------------

import matplotlib.pyplot as plt


sa_and_prices_plot = sa_and_prices.copy()

#sa_and_prices_plot = sa_and_prices_plot[(sa_and_prices_plot["year"]>2016)]
#sa_and_prices_plot = sa_and_prices_plot[(sa_and_prices_plot["month"]==5)]




fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["returns_z"],label = "Returns",color="tab:purple")
ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["EBF_z"],label = "EBF",color="tab:blue")
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["GEN_z"],label = "GEN_z",color="tab:green")
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["POL_z"],label = "POL_z",color="darkblue")
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["ENT_z"],label = "ENT_z",color="tab:orange")
plt.xlabel("date")
plt.ylabel("normalized_value")
plt.legend()
plt.show()



fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["returns_z"],label = "Returns",color="tab:purple")
ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["EBF_z_lag"],label = "EBF Lag",color="tab:blue")
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["GEN_z_lag"],label = "GEN Lag",color="tab:green")
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["POL_z_lag"],label = "POL Lag",color="darkblue")
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["ENT_z_lag"],label = "ENT Lag",color="tab:orange")
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["returns_z_lag"],label = "Returns Lag",color="violet",linewidth=0.3)
plt.xlabel("date")
plt.ylabel("normalized_value")
plt.legend()
plt.show()


#--------------------
#PLOT ESPECIFIC MONTH
#--------------------

sa_and_prices_month_plot = sa_and_prices_plot[(sa_and_prices_plot["year"]==2016)&(sa_and_prices_plot["month"]==1)]
sa_and_prices_month_plot = sa_and_prices_plot[(sa_and_prices_plot["year"]==2017)&(sa_and_prices_plot["month"]==10)]
sa_and_prices_month_plot = sa_and_prices_plot[(sa_and_prices_plot["year"]==2018)&(sa_and_prices_plot["month"]==12)]
sa_and_prices_month_plot = sa_and_prices_plot[(sa_and_prices_plot["year"]==2019)&(sa_and_prices_plot["month"]==11)]



fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["returns_z"],label = "Returns_z",color="tab:purple",linewidth=3)
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["EBF_z"],label = "EBF_z",color="tab:blue")
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["GEN_z"],label = "GEN_z",color="tab:green")
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["POL_z"],label = "POL_z",color="darkblue")
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["ENT_z"],label = "ENT_z",color="tab:orange")
plt.xlabel("date")
plt.ylabel("normalized_value")
plt.title("Returns and Sentiment Analysis per Category")
plt.legend()
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["returns_z"],label = "Returns",color="tab:purple",linewidth=3)
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["EBF_z_lag"],label = "EBF Lag",color="tab:blue")
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["GEN_z_lag"],label = "GEN Lag",color="tab:green")
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["POL_z_lag"],label = "POL Lag",color="darkblue")
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["ENT_z_lag"],label = "ENT Lag",color="tab:orange")
#ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["returns_z_lag"],label = "Returns Lag")
plt.xlabel("date")
plt.ylabel("normalized_value")
plt.legend()
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["returns_z"],label = "Returns",color="tab:purple",linewidth=3)
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["returns_z_lag"],label = "Returns Lag",color="tab:purple")
plt.xlabel("date")
plt.ylabel("normalized_value")
plt.legend()
plt.show()



fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["returns_z"],label = "Returns",color="tab:purple")
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["EBF_z"],label = "EBF_z",color="tab:blue")
ax.plot(sa_and_prices_month_plot["date"], sa_and_prices_month_plot["EBF_z_lag"],label = "EBF_z_lag",color="tab:blue",linewidth=0.5)
plt.xlabel("date")
plt.ylabel("normalized_value")
plt.legend()
plt.show()



#---------------
#PLOT BY MONTHS
#---------------


# Create a column associating the year month
sa_and_prices_plot['date_month'] = pd.to_datetime(sa_and_prices_plot['date']).dt.to_period('M')

sa_and_prices_plot.head()



# Get the start and end months

#group by month

months_data = sa_and_prices_plot.groupby(["date_month"])[["returns_z","EBF_z", "GEN_z", "POL_z","ENT_z"]].mean().reset_index()


months_data['month_trend'] = np.arange(len( months_data))


months_data

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(months_data["month_trend"], months_data["returns_z"],label = "Returns_z",color="tab:purple",linewidth=3)
ax.plot(months_data["month_trend"], months_data["EBF_z"],label = "EBF_z",color="tab:blue")
ax.plot(months_data["month_trend"], months_data["GEN_z"],label = "GEN_z",color="tab:green")
ax.plot(months_data["month_trend"], months_data["POL_z"],label = "POL_z",color="darkblue")
ax.plot(months_data["month_trend"], months_data["ENT_z"],label = "ENT_z",color="tab:orange")
plt.xlabel("year-month")
plt.ylabel("normalized_value")
plt.legend()
plt.show()



corr_matrix3 = months_data.corr()

heatmap_plot3 = sn.heatmap(corr_matrix3, annot=True,vmin=-1, vmax=1,cmap= "icefire_r")
heatmap_plot3.set(xlabel="", ylabel="")
plt.show()




#-------------------------
#------OLS Regression-----
#-------------------------

#https://www.kaggle.com/code/ryanholbrook/linear-regression-with-time-series
#https://realpython.com/linear-regression-in-python/ OLS
#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8



sa_and_prices.head()

sa_and_prices.columns

import statsmodels.api as sm
import statsmodels.formula.api as smf

#Using numpy arrays
#explanatory_columns = ["time_trend","returns_z_lag","EBF_z_lag","GEN_z_lag","POL_z_lag"]
explanatory_columns = ["time_trend","EBF_z_lag","GEN_z_lag","POL_z_lag"]

selected_df = sa_and_prices[explanatory_columns]
selected_df = selected_df[1:].copy()
X = selected_df.to_numpy()
X = sm.add_constant(X)

y = sa_and_prices["returns"]


results = sm.OLS(y, X).fit()

# Inspect the results
print(results.summary())

results.summary()

#--------
#USING FORMULA
#--------

results = smf.ols('adjclose ~ time_trend + EBF_z_lag + GEN_z_lag + POL_z_lag + ENT_z_lag', data=sa_and_prices).fit()
results.summary()


results = smf.ols('returns ~ returns_z_lag + EBF_z + GEN_z + POL_z + ENT_z', data=sa_and_prices).fit()
print(results.summary())
results.summary2()

results = smf.ols('returns ~ returns_z_lag + EBF_z + GEN_z + POL_z + ENT_z + EBF_z_lag + GEN_z_lag + POL_z_lag + ENT_z_lag', data=sa_and_prices).fit()
print(results.summary())
results.summary2()

results = smf.ols('returns ~ returns_z_lag+ EBF_z_lag + GEN_z_lag + POL_z_lag + ENT_z_lag', data=sa_and_prices).fit()
results.summary()


results = smf.ols('returns ~ EBF_z + GEN_z + POL_z + ENT_z', data=sa_and_prices).fit()
print(results.summary())
results.summary2()


#-------------------------
#--Logistic Regression----
#-------------------------

sa_and_prices = sa_and_prices.copy()

sa_and_prices.loc[sa_and_prices["returns"]>=0, "direction"] = 1
sa_and_prices.loc[sa_and_prices["returns"]< 0, "direction"] = 0

sa_and_prices.loc[sa_and_prices["returns_z_lag"]>=0, "direction_lag"] = 1
sa_and_prices.loc[sa_and_prices["returns_z_lag"]< 0, "direction_lag"] = 0

sa_and_prices.head()


#Matrix way
explanatory_columns = ["time_trend","direction_lag","EBF_z_lag","GEN_z_lag","POL_z_lag"]
selected_df = sa_and_prices[explanatory_columns]
X = selected_df.to_numpy()
X = sm.add_constant(X)
y = sa_and_prices["direction"]
logit_model=sm.Logit(y,X)



#Formula way
logit_model = smf.logit('direction ~ returns_z_lag+direction_lag+ EBF_z_lag + GEN_z_lag + POL_z_lag + ENT_z_lag', data=sa_and_prices)
result=logit_model.fit()
print(result.summary2())
print(result.summary())
result.summary()


logit_model2 = smf.logit('direction ~ returns_z_lag+direction_lag+EBF_z + GEN_z + POL_z + ENT_z + EBF_z_lag + GEN_z_lag + POL_z_lag + ENT_z_lag', data=sa_and_prices)
result=logit_model2.fit()
print(result.summary2())
print(result.summary())
result.summary()


logit_model3 = smf.logit('direction ~ EBF_z + GEN_z + POL_z + ENT_z', data=sa_and_prices)
result=logit_model3.fit()
print(result.summary2())
print(result.summary())
result.summary()


# FORMATING RESULTS
from statsmodels.iolib.summary2 import summary_col

summary = summary_col([result], float_format='%0.4f', stars=True, model_names=['Model'])
# Get summary of regression results
summary = result.summary2()

summary.tables[1]
# Convert summary to HTML format
summary_html1 = summary.tables[0].to_html()
summary_html2 = summary.tables[1].to_html()

summary_df1 = summary.tables[0]
summary_df2 = summary.tables[1]


# Write HTML to file or display it
with open('logit_results1.html', 'w') as file:
    file.write(summary_html1)

with open('logit_results2.html', 'w') as file:
    file.write(summary_html2)

    
# Convert summary to HTML format
summary_html = summary.as_html()

# Write HTML to file or display it
with open('logit_results.html', 'w') as file:
    file.write(summary_html)


# RESULTS AS EXCEL

summary_df1.to_excel(consolidated_results_path+'summary_df1.xlsx', index=False)
summary_df2.to_excel(consolidated_results_path+'summary_df2.xlsx', index=False)
    
# Convert summary to DataFrame
results_df = pd.concat([table for table in summary.tables], axis=1)

da = summary.tables[0]

# Write DataFrame to Excel
results_df.to_excel('logit_results.xlsx', index=False)





logit_model4 = smf.logit('direction ~direction_lag+ EBF_z_lag+GEN_z_lag + POL_z_lag + ENT_z_lag', data=sa_and_prices)
result=logit_model4.fit()
print(result.summary2())
print(result.summary())
result.summary()





logit_model2 = smf.logit('direction ~ direction_lag + EBF_z_lag + GEN_z_lag + POL_z_lag + ENT_z_lag', data=sa_and_prices)
result=logit_model2.fit()
print(result.summary2())
print(result.summary())
result.summary()




logit_model2 = smf.logit('direction ~ EBF_z + GEN_z + POL_z + ENT_z + EBF_z_lag + GEN_z_lag + POL_z_lag + ENT_z_lag', data=sa_and_prices)
result=logit_model2.fit()
print(result.summary2())
print(result.summary())
result.summary()
