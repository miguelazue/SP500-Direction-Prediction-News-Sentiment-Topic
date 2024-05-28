
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

os.chdir("C:/Users/migue/Dropbox/JKU EBA Masters/Master Thesis/")

import A_ThesisFunctions as tf




#-------------------------
#--Execution Parameters---
#-------------------------

#Make sure to modify and select the parameters to be executed


# BERTopic model to be loaded
model_year = "2016"

# Model Headlines Count Size
model_headlines_count = "20"

# News to be loaded and processed by the BERTopic model
news_year = "2019"



#Top 30 topics from the 2016 model most correlated with the S&P500
#selected_topics_20k = [216, 38, 94, 310, 287, 304, 61, 116, 203, 76, 98, 255, 208, 312, 256,82, 226, 157, 188, 169, 130, 33, 306, -1, 245, 185, 220, 29, 264, 8]

selected_topics_250k = [5, 61, 17, 4, 60, 212, 95, 100, 20, 53, 33, 187, 62, 73, 155, 114, 203, 131, 43, 254, 105, 3, 57, 51, 176, 117, 108, 30, 89, 271]

selected_topics = selected_topics_250k
#-------------------------
#----Load Topic Model-----
#-------------------------


#Topics Path
topics_path = "C:/Users/migue/Downloads/Thesis Data/BERTopics/"

#current_topic_model = BERTopic.load(topics_path+"topics_model2016")
#current_topic_model = BERTopic.load(topics_path+"topics_model2016_100topics")
# model_name = "BERTopicModel2016Headlines20k"

model_name = "BERTopicModel"+model_year+"Headlines"+model_headlines_count+"k"

current_topic_model = BERTopic.load(topics_path+model_name)


#-------------------------
#-------Load SA EBF-------
#-------------------------


sa_ebf_path = "C:/Users/migue/Downloads/Thesis Data/SentimentAnalysisEBF/"
#sa_ebf_path = "C:/Users/migue/Downloads/Thesis Data/SentimentAnalysisEBF/2017/"
sa_ebf_path_year = sa_ebf_path + news_year + '/'
sa_ebf_path_year


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

filtered_df = new_documents_df[new_documents_df['topic'].isin(selected_topics)]

#filtered_df

summarized_filtered_df = tf.summarize_sa_topic(filtered_df,n_dates_threshold=1,min_prob=-1)


topics_sa_path = "C:/Users/migue/Dropbox/JKU EBA Masters/Master Thesis/Data/sa_top_topics/"

save_name = "sa_top_topics_BERTopicModel"+model_year+"Headlines"+model_headlines_count+"k_news"+news_year+'.csv'

#topics_sa_path+save_name

summarized_filtered_df.to_csv(topics_sa_path+save_name)