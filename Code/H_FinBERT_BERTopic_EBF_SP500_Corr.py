"""
Created on Sun May 13 16:44:23 2023

Correlation analysis between News SA, Topics and S&P500


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

from bertopic import BERTopic

import seaborn as sn
import matplotlib.pyplot as plt

import os

#-------------------------
#-----Path Definition-----
#-------------------------

# Working Directory Definition
# Use the current working directory or define the path for the working directory. Use option 1 by by commenting option 2
current_dir = os.getcwd()  # Option 1: Use the current working directory.
current_dir = "C:/Users/migue/Dropbox/JKU EBA Masters/Master Thesis/Data/"  # Option 2: Define the path for the working directory

# Large Size Data Directory Definition
# It is Recommended to store this data locally, if space is not a problem larga data directory can be included in the current dir by commenting option 2
large_data_path = current_dir # Option 1: Large data directory in the working directory
large_data_path = "C:/Users/migue/Downloads/Large_Data/" # Option 2: Define the path for large data storage

#-------------------------
#-----Relative Paths------
#-------------------------

# relative paths associated with current_dir
consolidated_data_path = os.path.join(current_dir,'consolidated_data/')
consolidated_results_path = os.path.join(current_dir,'consolidated_results/')
categories_data_path = os.path.join(current_dir,'sa_publishers_category/')
topics_sa_path = os.path.join(current_dir,'sa_top_topics/')

# relative paths associated with large_data_path
parquet_path = os.path.join(large_data_path,'filtered_news_parquets/')
ebf_parquet_path = os.path.join(large_data_path,'filtered_news_parquets/EBF_News/')
ebf_sa_path = os.path.join(large_data_path,'sentiment_analysis_EBF/')
ebf_sa_full_article_path = os.path.join(large_data_path,'sentiment_analysis_full_article_EBF/')
topics_path = os.path.join(large_data_path,'BERTopics/')

#-------------------------
#-Load Utility Script-----
#-------------------------

os.chdir(current_dir)
import A_ThesisFunctions as tf


#-------------------------
#--Execution Parameters---
#-------------------------

#Make sure to modify and select the parameters to be executed


# BERTopic model to be loaded
model_year = "2016"
model_headlines_count = "20"

# News to be loaded and processed by the BERTopic model
news_year = "2016"




#-------------------------
#---Load Financial Data---
#-------------------------

gspc_data = pd.read_csv(consolidated_data_path+"GSPC.csv")


selected_columns = ["formatted_date","time_trend","adjclose","returns"]
gspc_df = gspc_data[selected_columns].copy()

gspc_df["formatted_date"] = pd.to_datetime(gspc_df["formatted_date"])

gspc_df['year'] = gspc_df['formatted_date'].dt.year
gspc_df['month'] = gspc_df['formatted_date'].dt.month

#gspc_headlines = gspc_df[(gspc_df["year"]==2017)&(gspc_df["month"]==10)]
gspc_sample = gspc_df.copy()
gspc_sample.head()
#gspc_sample.tail()




#-------------------------
#----Load Topic Model-----
#-------------------------


#current_topic_model = BERTopic.load(topics_path+"topics_model2016")
#current_topic_model = BERTopic.load(topics_path+"topics_model2016_100topics")


model_name = "BERTopicModel" + model_year + "Headlines" + model_headlines_count +"k"
#model_name

current_topic_model = BERTopic.load(topics_path+model_name)

current_topic_model.get_topic_info()

#-------------------------
#-------Load SA EBF-------
#-------------------------


sa_ebf_path_year = ebf_sa_path + news_year + '/'
# sa_ebf_path_year


dataset = pq.ParquetDataset(sa_ebf_path_year,use_legacy_dataset=False)
news_sa_df = dataset.read(columns=["headline","date","year","sa_score"]).to_pandas()

#Converting date to datetime format
news_sa_df['date'] = pd.to_datetime(news_sa_df['date'], format='%Y-%m-%d')

news_sa_df = news_sa_df.sort_values(by=['date'])

#news_sa_df.head()
#news_sa_df.tail()

#-------------------------
#-Associate News to topics
#-------------------------

new_documents = news_sa_df['headline'].tolist()
associated_dates = news_sa_df['date'].tolist()
associated_sa_score = news_sa_df['sa_score'].tolist()


associated_topics, associated_probs = current_topic_model.transform(new_documents)

new_documents_df = pd.DataFrame({"document":new_documents,
                                 "date":associated_dates,
                                 "sa_score":associated_sa_score,
                                 "topic":associated_topics,
                                 "probs":associated_probs})


new_documents_df.head()

#len(new_documents_df)


#-------------------------
#--Correlation Analysis---
#-------------------------

# Summarize the data frame sentiment analysis score per day per topic
sa_topic_summary = tf.summarize_sa_topic(new_documents_df)

# Join financial data with summarized data
sa_topics_and_prices = pd.merge(gspc_sample,sa_topic_summary, left_on='formatted_date', right_on='date', how='inner')

# Drop unnecesary columns
sa_topics_and_prices = sa_topics_and_prices.drop(columns=['time_trend', 'adjclose','year','month'])

# Find the most correlated topics
result_corr_matrix = tf.top_returns_correlations(sa_topics_and_prices,corr_treshold=0.1)


heatmap_plot = sn.heatmap(result_corr_matrix, annot=True,vmin=-1, vmax=1,cmap= "icefire_r")
heatmap_plot.set(xlabel="", ylabel="")
plt.show()


#-------------------------
#TOP 30 SELECTION OF TOPICS
#-------------------------

#Reduce until finding top 30 topics
result_corr_matrix2 = tf.top_returns_correlations(sa_topics_and_prices,corr_treshold=0.1)
len(result_corr_matrix2['returns'])

top_returns_corr = result_corr_matrix2.reset_index()
top_returns_corr_df = top_returns_corr[['index','returns']]
top_returns_corr_df.rename(columns = {'index':'topic','returns':'corr_returns'}, inplace = True)

top_returns_corr_df['abs_corr_returns'] = abs(top_returns_corr_df['corr_returns'])

top_returns_corr_df = top_returns_corr_df[top_returns_corr_df['topic']!='returns']
top_returns_corr_df = top_returns_corr_df.sort_values(by="abs_corr_returns",ascending=False).reset_index(drop=True)
top_returns_corr_df = top_returns_corr_df[:30]

top_returns_corr_df['topic_number'] = top_returns_corr_df['topic'].str.split('_').str[1]
top_returns_corr_df['topic_number'] = pd.to_numeric(top_returns_corr_df['topic_number'])


top_returns_corr_df = top_returns_corr_df.sort_values(by="topic_number",ascending=True)



# Save Top Returns Corr
file_name2 = "top_returns_corr_df_" + "topics_BERTopicModel"+model_year+"Headlines"+model_headlines_count+"k_news"+news_year+'.csv'
file_name2

top_returns_corr_df.to_csv(topics_path+file_name2)




selected_topics_250k = top_returns_corr_df['topic_number'].tolist()
selected_topics_250k

len(selected_topics_250k)

selected_topics_250k = [5, 61, 17, 4, 60, 212, 95, 100, 20, 53, 33, 187, 62, 73, 155, 114, 203, 131, 43, 254, 105, 3, 57, 51, 176, 117, 108, 30, 89, 271]


selected_topics_250K = [4,5,17,20,33,39,48,53,57,60,61,62,73,79,95,99,100,102,106,114,117,119,131,155,170,203,212,239,247,271]

top_topics_indfo_df = current_topic_model.get_topic_info()
top_topics_indfo_df = top_topics_indfo_df[top_topics_indfo_df["Topic"].isin(selected_topics_250K)]



columns = ["Topic","Name"]
top_topics_indfo_df = top_topics_indfo_df.reset_index(drop=True)
formatted_results3 = top_topics_indfo_df[columns].head(30)

formatted_results3.to_excel(topics_path+"top_correlated_topics.xlsx")

# Code for seeing selected topics and the correlation values
top_topics_indfo_df.dtypes
top_returns_corr_df.dtypes
top_topics_indfo_df = top_topics_indfo_df.rename(columns={"Topic": "topic_number", "Name": "name"})
top_topics_all_info = pd.merge(top_topics_indfo_df, top_returns_corr_df, how="inner", on=["topic_number"])

columns = ["topic","name","corr_returns"]
top_topics_all_info[columns]



#-------------------------
#---------PLOTS-----------
#-------------------------

import matplotlib.pyplot as plt


#38
#165

#Plotting data

columns_to_keep = ['date']
columns_to_keep.extend(result_corr_matrix.columns.to_list())

sa_and_prices_plot = sa_topics_and_prices.copy()
sa_and_prices_plot = sa_and_prices_plot[columns_to_keep]
sa_and_prices_plot = tf.df_normalization(sa_and_prices_plot)



#sa_and_prices_plot = sa_and_prices_plot[(sa_and_prices_plot["date"]<"2017-01-30")]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["returns_z"],label = "Returns",color="tab:purple",linewidth=1.5)
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["topic_318_z"],label = "topic_318",color="tab:blue")
ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["topic_5_z"],label = "topic_5_z",color="tab:blue")
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["topic_228_z"],label = "topic_228_z",color="tab:blue")
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["topic_309_z"],label = "topic_309_z",color="tab:blue")
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["topic_309_z"],label = "topic_309_z",color="tab:blue")
plt.xlabel("date")
plt.ylabel("normalized_value")
plt.legend()
plt.show()



fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["returns_z"],label = "Returns",color="tab:purple",linewidth=1.5)
#ax.scatter(sa_and_prices_plot["date"], sa_and_prices_plot["topic_318_z"],label = "topic_318",color="tab:blue",s=20)
ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["topic_53_z"],label = "topic_53_z",color="tab:orange")
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["topic_138_z"],label = "topic_138_z",color="tab:green")
#ax.scatter(sa_and_prices_plot["date"], sa_and_prices_plot["topic_243_z"],label = "topic_243_z",color="tab:red")
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["topic_61_z"],label = "topic_61_z",color="tab:brown")
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["topic_94_z"],label = "topic_94_z",color="tab:cyan")
#ax.plot(sa_and_prices_plot["date"], sa_and_prices_plot["topic_243_z"],label = "topic_243_z",color="tab:olive")
plt.xlabel("date")
plt.ylabel("normalized_value")
plt.legend()
plt.show()


current_topic_model.get_topic_info(5)
current_topic_model.get_topic_info(138)
current_topic_model.get_topic_info(53)

new_documents_df[new_documents_df["topic"] == 5]
new_documents_df[new_documents_df["topic"] == 53]
new_documents_df[new_documents_df["topic"] == 138]

sa_and_prices_plot[sa_and_prices_plot['topic_138'].isna()==False]
new_documents_df[new_documents_df['topic']==5]



#-------------------------
#----SUMMARY FILE---------
#-------------------------


#My Test without filtering any news
sa_topic_summary = tf.summarize_sa_topic(new_documents_df,n_dates_threshold=0,min_prob=-1)
sa_topics_and_prices = pd.merge(gspc_sample,sa_topic_summary, left_on='formatted_date', right_on='date', how='inner')
sa_topics_and_prices = sa_topics_and_prices.drop(columns=['time_trend', 'adjclose','year','month'])

#Summary file, get count for each topic and its correlation with S&P500 returns

topic_counts = new_documents_df.groupby(["topic"])['document'].count().reset_index()
news_count = sum(topic_counts['document'])
topic_counts['distribution'] = topic_counts["document"]/news_count
topic_counts["topic"] = "topic_" + topic_counts["topic"].apply(str)

topic_counts

returns_corr = sa_topics_and_prices.corr().reset_index()
returns_corr_df = returns_corr[['index','returns']]

returns_corr_df.rename(columns = {'index':'topic','returns':'corr_returns'}, inplace = True)

topics_corr_summary = pd.merge(topic_counts,returns_corr_df, left_on='topic', right_on='topic', how='inner')
topics_corr_summary.reset_index()

file_name = "topics_BERTopicModel"+model_year+"Headlines"+model_headlines_count+"k_news"+news_year+'.csv'
file_name
topics_corr_summary.to_csv(topics_path+file_name)



#-------------------------
#--COMPARE SUMMARY FILES--
#-------------------------



# Load correlations between topics and returns 
# Topics Path

model_year = "2016"
headlines_size = "250"
news_year1 = "2016"
news_year2 = "2017"

file_name1 = 'topics_BERTopicModel'+model_year+'Headlines'+headlines_size+'k_news'+news_year1+'.csv'
file_name2 = 'topics_BERTopicModel'+model_year+'Headlines'+headlines_size+'k_news'+news_year2+'.csv'
my_file16 = pd.read_csv(topics_path+file_name1)
my_file17 = pd.read_csv(topics_path+file_name2)



topics_list = my_file16['topic']
count_2016 = my_file16['document']
distribution_2016 = my_file16['distribution']
corr_returns_2016 = my_file16['corr_returns']
abs_corr_returns_2016 = abs(my_file16['corr_returns'])


count_2017 = my_file17['document']
distribution_2017 = my_file17['distribution']
corr_returns_2017 = my_file17['corr_returns']
abs_corr_returns_2017 = abs(my_file17['corr_returns'])


# Topic Info
topics_name_df = current_topic_model.get_topic_info()
topic_names = topics_name_df["Name"]



comparison_df = pd.DataFrame({"topic":topics_list,
                              "topic_name":topic_names,
                              
                              "count_2016":count_2016,
                              "count_2017":count_2017,

                              "distribution_2016":distribution_2016,
                              "distribution_2017":distribution_2017,

                              "corr_returns_2016":corr_returns_2016,
                              "corr_returns_2017":corr_returns_2017,

                              "abs_corr_returns_2016":abs_corr_returns_2016,
                              "abs_corr_returns_2017":abs_corr_returns_2017

                              })


comparison_df.head()

#comparison_df = comparison_df.sort_values(["abs_corr_returns_2016","count_2016"],ascending=[False,True]).reset_index()

relevant_columns1 = ['topic', 'topic_name', 'count_2016', 'count_2017','corr_returns_2016', 'corr_returns_2017']
relevant_columns2 = ['topic', 'topic_name', 'corr_returns_2016', 'corr_returns_2017']
relevant_columns3 = ['topic', 'topic_name', 'distribution_2016', 'distribution_2017','count_2016','count_2017']



comparison_df.iloc[:30][relevant_columns1]
comparison_df.iloc[:30][relevant_columns3]



comparison_df['topic_number'] = comparison_df['topic'].str.split('_').str[1]
comparison_df['topic_number'] = pd.to_numeric(comparison_df['topic_number'])

selected_topics_250k = [5, 61, 17, 4, 60, 212, 95, 100, 20, 53, 33, 187, 62, 73, 155, 114, 203, 131, 43, 254, 105, 3, 57, 51, 176, 117, 108, 30, 89, 271]

top_topics_comparison = comparison_df[comparison_df['topic_number'].isin(selected_topics_250k)]

top_topics_comparison[relevant_columns1]



comparison_df_order = comparison_df[:30]
comparison_df_order
comparison_df_order['count_2017'] = comparison_df_order['count_2017'].astype('int')

comparison_df_order.iloc[:30][relevant_columns3]





