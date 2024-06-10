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

        preprocessed_df = tf.preprocess_news_categorization(news_df)

        preprocessed_df.head()
        
        #Sentiment Analysis on Dataframe
        sa_df = tf.sentiment_analysis_on_categories_df(preprocessed_df)

        #Summarizing the results
        summarized_result = tf.summarize_sa(sa_df)

        # get the end time
        et = time.time()

        # get the execution time
        elapsed_time_seconds = round(et - st,2)
        elapsed_time_minutes = round(elapsed_time_seconds/60,2)
        print("Execution time:", elapsed_time_seconds, "seconds")
        print("Execution time:", elapsed_time_minutes, "minutes")

        
        summarized_result.to_csv(categories_data_path+"sa_publishers_category_"+str(iteration_year*100+iteration_month)+".csv")




