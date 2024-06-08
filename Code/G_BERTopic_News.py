# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:19:23 2023

Testing BERTopic

https://github.com/MaartenGr/BERTopic


Learn BertTOPIC
https://hackernoon.com/nlp-tutorial-topic-modeling-in-python-with-bertopic-372w35l9

#Testing LDA
https://github.com/lda-project/lda


@author: migue
"""


import pandas as pd 
import numpy as np


from bertopic import BERTopic

import pyarrow as pa
import pyarrow.parquet as pq

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

import time


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


# Heavy Data directory - Recommended to store this data locally
heavy_data_path = "C:/Users/migue/Downloads/Thesis Data/"

parquet_path = os.path.join(heavy_data_path,'FilteredNewsParquets/')
ebf_parquet_path = os.path.join(heavy_data_path,'FilteredNewsParquets/EBF_News/')
ebf_sa_path = os.path.join(heavy_data_path,'SentimentAnalysisEBF/')
topics_path = os.path.join(heavy_data_path,'BERTopics/')


#-------------------------
#--Load Financial News----
#-------------------------

#Read EBF Parquets
dataset = pq.ParquetDataset(ebf_parquet_path,use_legacy_dataset=False)

#parquet_path =  "C:/Users/migue/Downloads/Thesis Data/FilteredNewsParquets/"
#dataset = pq.ParquetDataset(parquet_path,use_legacy_dataset=False)

#Read by Filtered year
dataset = pq.ParquetDataset(ebf_parquet_path,use_legacy_dataset=False,filters=[("year","=",2016)])
#dataset = pq.ParquetDataset(ebf_parquet_path,use_legacy_dataset=False,filters=[("publication","=","Economist")])



#To Pandas by selecting Columns
news_df = dataset.read(columns=["date","title","publication","section","year"]).to_pandas()

#news_df = dataset.read(columns=["date","title","publication","section","year","article"]).to_pandas()

#Converting date to datetime format
news_df['date'] = pd.to_datetime(news_df['date'], format='%Y-%m-%d')

news_df = news_df.sort_values("date")
news_df = news_df.reset_index(drop=True)

news_df.head()
news_df.tail()

#news_df_sample = news_df.sample(frac=0.1, random_state=1)
#news_df_sample = news_df.sample(n=50000, random_state=1)
#news_df_sample = news_df.sample(n=20000, random_state=1)
#news_df_sample = news_df.sample(n=10000, random_state=1)
#news_df_sample = news_df.sample(n=5000, random_state=1)
#news_df_sample = news_df.sample(n=3000, random_state=1)
#news_df_sample = news_df.sample(n=2000, random_state=1)
#news_df_sample = news_df.sample(n=1000, random_state=1)
#news_df_sample = news_df.sample(n=100, random_state=1)
news_df_sample = news_df.copy()


#drop nas
news_df_sample = news_df_sample.dropna(subset=['title'])
#news_df_sample = news_df_sample.dropna(subset=['article'])
len(news_df_sample)


#-------------------------------
#-------Applying BERTopic-------
#-------------------------------

#converting to list of titles
documents = news_df_sample['title'].tolist()
#documents = news_df_sample['article'].tolist()


#topic_model = BERTopic(nr_topics=100,verbose=True) 
#topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2",verbose=True) 
topic_model = BERTopic(verbose=True).fit(documents)

#topic_model = BERTopic(verbose=True)
#topics, probs = topic_model.fit_transform(documents)

topics_info = topic_model.get_topic_info()
topics_info


# topic_model.get_document_info(documents)


#save model
# topic_model.save(topics_path+"BERTopicModel2016Headlines250k")




#-------------------------------
#--Load and Explore BERTopic----
#-------------------------------


topic_model.get_topic_freq().head(11)

#load model
#Topic_model2016 = BERTopic.load(topics_path+"BERTopicModel2016Headlines250k")
Topic_model2016 = BERTopic.load(topics_path+"BERTopicModel2016Headlines20k")


# Visualize information from the Topic Model

#Topic_model2016.get_topic_info()

formatted_results = Topic_model2016.get_topic_info()[:20]
formatted_results2 = Topic_model2016.get_topic_info()[-20:]

Topic_model2016.get_representative_docs(2)
#Topic_model2016.topic_representations_[10]
