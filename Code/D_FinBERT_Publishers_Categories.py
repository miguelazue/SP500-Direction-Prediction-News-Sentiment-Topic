"""
Created on Thu Apr 20 


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

import os

os.chdir("C:/Users/migue/Dropbox/JKU EBA Masters/Master Thesis/")

from A_ThesisFunctions import *

#-------------------------
#---------ITERATION-------
#-------------------------



#Iterate over periods
#years = [2016,2017,2018,2019,2020]
years = [2020]
#months = [12]
months = [1,2,3,4]
#months = [1,2,3,4,5,6,7,8,9,10,11,12]
#months = [10]

for iteration_year in years:
    for iteration_month in months:

        print("Processing Year "+str(iteration_year)+" - month "+str(iteration_month))

        #Read all parquets
        parquet_path =  "C:/Users/migue/Downloads/Thesis Data/FilteredNewsParquets/"
        dataset = pq.ParquetDataset(parquet_path,use_legacy_dataset=False)

        #To Pandas by selecting Columns
        news_df = dataset.read(columns=["date","title","publication","section","year"]).to_pandas()

        #Converting date to datetime format
        news_df['date'] = pd.to_datetime(news_df['date'], format='%Y-%m-%d')

        news_df = news_df[(news_df['date'].dt.year == iteration_year)&(news_df['date'].dt.month == iteration_month)]
        
        df_size = round(round(news_df.memory_usage().sum()/1024,2)/1024,2)

        number_of_news = len(news_df)
        print("Dataframe size:", df_size,"MB"," - ",number_of_news," news")


        # get the start time
        st = time.time()

        #preprocessed_df = preprocess_news_categorization(news_df.head(30))
        #preprocessed_df = preprocess_news_categorization(news_df[42000::])

        preprocessed_df = preprocess_news_categorization(news_df)

        preprocessed_df.head()
        
        #Sentiment Analysis on Dataframe
        sa_df = sentiment_analysis_on_categories_df(preprocessed_df)

        #Summarizing the results
        summarized_result = summarize_sa(sa_df)

        # get the end time
        et = time.time()

        # get the execution time
        elapsed_time_seconds = round(et - st,2)
        elapsed_time_minutes = round(elapsed_time_seconds/60,2)
        print("Execution time:", elapsed_time_seconds, "seconds")
        print("Execution time:", elapsed_time_minutes, "minutes")

        #SAVE
        save_path = "C:/Users/migue/Dropbox/JKU EBA Masters/Master Thesis/Data/sa_publishers_category/"

        summarized_result.to_csv(save_path+"sa_publishers_category_"+str(iteration_year*100+iteration_month)+".csv")






#-------------------------
#--Load Financial News----
#-------------------------

# #Read only EBF Parquets
# # ebf_parquet_path =  "C:/Users/migue/Downloads/Thesis Data/FilteredNewsParquets/EBF_News/"
# #dataset = pq.ParquetDataset(ebf_parquet_path,use_legacy_dataset=False)

# #Read all parquets
# parquet_path =  "C:/Users/migue/Downloads/Thesis Data/FilteredNewsParquets/"
# dataset = pq.ParquetDataset(parquet_path,use_legacy_dataset=False)


#Read by Filtered year
# parquet_path =  "C:/Users/migue/Downloads/Thesis Data/FilteredNewsParquets/"
# parquet_path =  "C:/Users/migue/Downloads/Thesis Data/FilteredNewsParquets/EBF_News/"
# dataset = pq.ParquetDataset(parquet_path,use_legacy_dataset=False,filters=[("year","=",2016),("publication","=","Business Insider")])
# dataset = pq.ParquetDataset(parquet_path,use_legacy_dataset=False,filters=[("year","=",2016)])
# dataset = pq.ParquetDataset(parquet_path,use_legacy_dataset=False)

# #To Pandas by selecting Columns
# news_df = dataset.read(columns=["date","title","publication","section","year"]).to_pandas()
# news_df = dataset.read(columns=["date","publication","section"]).to_pandas()

#news_df = dataset.read().to_pandas()

# #Converting date to datetime format
# news_df['date'] = pd.to_datetime(news_df['date'], format='%Y-%m-%d')
# news_df = news_df.sort_values(by=['date'], ascending=True)

# n_articles = len(news_df)
# n_na = sum(news_df["section"].isna())
# n_na / n_articles

# news_df.section.value_counts()


# 9
# sample_news = news_df.sample(30,random_state=9)
# print(sample_news)

# save_path = "C:/Users/migue/Downloads/"
# sample_news.to_excel(save_path+"Sample_News.xlsx",engine='xlsxwriter')  

# #min(news_df['date'])
# #max(news_df['date'])

# #news_df.dtypes
# #news_df.memory_usage()
# #news_df.info(verbose = False)

# #round(news_df.memory_usage().sum()/1024,2)/1024

# #news_df.publication.value_counts()

# #news_df = news_df[(news_df['date'].dt.year == 2016)&(news_df['date'].dt.month == 7)]
# # news_df = news_df[(news_df['date'].dt.year == 2016)]

# #news_df['date'].dt.month.value_counts()
# #round(news_df.memory_usage().sum()/1024,2)/1024






#-------------------------
#----TESTING FUNCTIONS----
#-------------------------

# news_df_sample = news_df.sample(10)
# #news_df_sample = news_df[(news_df['date'].dt.year == 2016)&(news_df['date'].dt.month == 7)]
# #news_df_sample = news_df_sample[(news_df['publication']=="CNN")]

# test_df = preprocess_news_categorization(news_df_sample)
# result = sentiment_analysis_on_df(test_df)
# summarized_result = summarize_sa(result)
# summarized_result


#LOAD
#test_path = "C:/Users/migue/Dropbox/JKU EBA Masters/Master Thesis/Data/"

#sa_summary = pd.read_csv(test_path+"sa_publishers_category_201710.csv")
#sa_summary['date'] = pd.to_datetime(sa_summary['date'], format='%Y-%m-%d')
#sa_summary.dtypes
