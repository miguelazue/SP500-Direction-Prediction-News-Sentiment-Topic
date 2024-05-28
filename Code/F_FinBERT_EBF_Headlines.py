# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:19:23 2023
https://huggingface.co/ProsusAI/finbert

#How to use finbert
https://heartbeat.comet.ml/financial-sentiment-analysis-using-finbert-e25f215c11ba


FinBERT Sentiment Analysis for Long Text Corpus with much more than 512 Tokens NLP | PART -1
https://www.youtube.com/watch?v=WEAAs_0etJQ&ab_channel=Rohan-Paul-AI
https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel/blob/master/NLP/FinBERT_Long_Text.ipynb


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
months = [1,2,3,4]
#months = [1,2,3,4,5,6,7,8,9,10,11,12]
#months = [10]


for iteration_year in years:
    for iteration_month in months:

        print("Processing Year "+str(iteration_year)+" - month "+str(iteration_month))

        #Read all parquets
        parquet_path =  "C:/Users/migue/Downloads/Thesis Data/FilteredNewsParquets/EBF_News/"
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

        preprocessed_df = preprocess_news_date(news_df)

        preprocessed_df.head()
        
        #Sentiment Analysis on Dataframe
        #sa_df = sentiment_analysis_on_df(preprocessed_df.iloc[:100])
        sa_df = sentiment_analysis_on_df(preprocessed_df)
        # get the end time
        et = time.time()

        # get the execution time
        elapsed_time_seconds = round(et - st,2)
        elapsed_time_minutes = round(elapsed_time_seconds/60,2)
        print("Execution time:", elapsed_time_seconds, "seconds")
        print("Execution time:", elapsed_time_minutes, "minutes")

        #SAVE
        #current_path = "C:/Users/migue/Downloads/Thesis Data/SentimentAnalysisEBF/"
        #current_path = "C:/Users/migue/Downloads/Thesis Data/SentimentAnalysisEBF/"+str(iteration_year)+"/"+str(iteration_month)+"/"
        current_path = "C:/Users/migue/Downloads/Thesis Data/SentimentAnalysisEBF/"+str(iteration_year)+"/"+str(iteration_year*100+iteration_month)+"/"
        
        df_to_multiple_plarquets(df_to_slice = sa_df,save_path = current_path)




#-------------------------
#-----Read Results--------
#-------------------------

#full_path = "C:/Users/migue/Downloads/Thesis Data/SentimentAnalysisEBF/"

#dataset1 = pq.ParquetDataset(full_path,use_legacy_dataset=False)
#new_test = dataset1.read(columns=["headline","date","year","sa_score"]).to_pandas()
#Converting date to datetime format
#new_test['date'] = pd.to_datetime(new_test['date'], format='%Y-%m-%d')

#new_test.head()
#new_test.tail()