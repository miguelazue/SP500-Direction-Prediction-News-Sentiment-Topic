
"""
Created on July 7 8:59:23 2023

@author: migue
"""

import pandas as pd 
import numpy as np


import pyarrow as pa
import pyarrow.parquet as pq

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pysentiment2 as ps
# https://nickderobertis.github.io/pysentiment/

import time

import pickle
import joblib


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
import A_ThesisFunctions as thesisF


#-------------------------
#--Execution Parameters---
#-------------------------



# Year Model
model_year = "2016"
model_headlines_count = "250"

no_topics = 300
# no_topics = 30


model_name = "LDAmodel"+model_year+"Headlines"+model_headlines_count+"k" + str(no_topics) + "Topics"


#Iterate over files
#year_i = 2017
year_i = 2016


#-------------------------
#--Load Financial News----
#-------------------------


#Read EBF Parquets
dataset = pq.ParquetDataset(ebf_parquet_path,use_legacy_dataset=False)

#Read by Filtered year
dataset = pq.ParquetDataset(ebf_parquet_path,use_legacy_dataset=False,filters=[("year","=",year_i)])

#To Pandas by selecting Columns
news_df = dataset.read(columns=["date","title","publication","section","year"]).to_pandas()

#news_df = dataset.read(columns=["date","title","publication","section","year","article"]).to_pandas()

#Converting date to datetime format
news_df['date'] = pd.to_datetime(news_df['date'], format='%Y-%m-%d')

news_df = news_df.sort_values("date")
news_df = news_df.reset_index(drop=True)

news_df.head()
news_df.tail()





#-------------------------------
#------- Train LDA Model--------
#-------------------------------
news_df_sample = news_df.copy()
#news_df_sample = news_df.sample(n=20000, random_state=1)
#news_df_sample = news_df.sample(frac=0.1, random_state=1)
#news_df_sample = news_df.sample(n=10000, random_state=1)
#news_df_sample = news_df.sample(n=5000, random_state=1)
#news_df_sample = news_df.sample(n=3000, random_state=1)
#news_df_sample = news_df.sample(n=2000, random_state=1)
#news_df_sample = news_df.sample(n=1000, random_state=1)
#news_df_sample = news_df.sample(n=100, random_state=1)


#drop nas
news_df_sample = news_df_sample.dropna(subset=['title'])
#news_df_sample = news_df_sample.dropna(subset=['article'])
len(news_df_sample)


#converting to list of titles
documents = news_df_sample['title'].tolist()


#tf_vectorizer = CountVectorizer(strip_accents='unicode',lowercase = True,max_features = 1500,stop_words='english', max_df=0.8)
#tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
#tf_vectorizer = CountVectorizer(strip_accents='unicode',lowercase = True,stop_words='english',min_df=2)
#tf_vectorizer = CountVectorizer(strip_accents='unicode',lowercase = True,stop_words='english')
#tf_vectorizer = CountVectorizer(strip_accents='unicode',lowercase = True,stop_words='english',max_df=0.005)

tf_vectorizer = CountVectorizer(strip_accents='unicode',lowercase = True,stop_words='english',min_df=2,max_features=30000)
tf = tf_vectorizer.fit_transform(documents)

tf_feature_names = tf_vectorizer.get_feature_names_out()
len(tf_feature_names)

# Train LDA Model
# lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0,verbose=True).fit(tf)
#Menos de un mintuo con 20k documentos
# con 250K y 30 topics son 10 minutos
# con 250k y 300 topics 70 minutos


# Save LDA Model with 300 topics
#with open(topics_path+model_name+".pkl", 'wb') as fp:
#     pickle.dump(lda,fp)



# Save in a Joblib file
#joblib.dump(lda, topics_path+model_name+".joblib")


# Load the LDA
with open(topics_path+model_name+".pkl", 'rb') as fp:
    lda = pickle.load(fp)

# Load LDA Model from a joblib file
# lda = joblib.load(topics_path+"LDAmodel2016Sample20k.joblib")

# Topics Data Frame
topics_df = pd.DataFrame({'topic': pd.Series(dtype='string'),'terms': pd.Series(dtype='string')})

for topic_idx, topic in enumerate(lda.components_):
    topic_name = "topic_"+str(topic_idx)
    term1 = tf_feature_names[topic.argsort()[-1]]
    term2 = tf_feature_names[topic.argsort()[-2]]
    term3 = tf_feature_names[topic.argsort()[-3]]
    term4 = tf_feature_names[topic.argsort()[-4]]
    topic_terms = term1+"_"+term2+"_"+term3+"_"+term4
    current_row = [topic_name,topic_terms]
    topics_df.loc[len(topics_df)] = current_row
    #print(topic_name)
    #print(topic_terms)
    #print('\b')

print(topics_df)


# Top 30 topics highly correlated in the year 2016
selected_topics = ['topic_250', 'topic_51', 'topic_285', 'topic_42', 'topic_159', 'topic_170', 'topic_105', 'topic_267', 'topic_207', 'topic_136', 'topic_150', 'topic_154', 'topic_36', 'topic_197', 'topic_14', 'topic_252', 'topic_225', 'topic_232', 'topic_205', 'topic_118', 'topic_168', 'topic_216', 'topic_132', 'topic_29', 'topic_24', 'topic_280', 'topic_74', 'topic_134', 'topic_62', 'topic_50']
correlated_topics = topics_df[topics_df["topic"].isin(selected_topics)]
correlated_topics.head(30)


topics_df.to_csv(topics_path+"topics_"+model_name+".csv")

# In case needed to establish a new count vectorizer
# lda_n_features = lda.n_features_in_
# tf_vectorizer = CountVectorizer(strip_accents='unicode',lowercase = True,stop_words='english',min_df=2,max_features=lda_n_features)

#-------------------------------
#-------Apply LDA Model---------
#-------------------------------


# Change dates
preprocessed_df = thesisF.preprocess_news_date(news_df)
len(preprocessed_df)
preprocessed_df = preprocessed_df.dropna(subset=['title'])
len(preprocessed_df)
# Testing Small Sample
#preprocessed_df = preprocessed_df[:30].copy()

new_documents = preprocessed_df['title']
dates_list = preprocessed_df['associated_date']
years_list = preprocessed_df['year']


# Load the LDA
with open(topics_path+model_name+".pkl", 'rb') as fp:
    lda = pickle.load(fp)


tf_vectorizer = CountVectorizer(strip_accents='unicode',lowercase = True,stop_words='english',min_df=2,max_features=30000)
tf = tf_vectorizer.fit_transform(new_documents)

tf_feature_names = tf_vectorizer.get_feature_names_out()
len(tf_feature_names)

# len(new_documents)

# Transform the new documents into a document-term matrix
new_X = tf_vectorizer.transform(new_documents)

# Use the LDA model to assign topics to the new documents
new_document_topics = lda.transform(new_X)
# Less than 30 seconds applying 250K 300 topics


#-------------------------------
#----Associating LM and LDA-----
#-------------------------------

# Initialize Loughran and McDonald dictionary model
lm = ps.LM()
# Print the topic distribution for each new document
associated_topic_list = []
lm_scores_list = []


# get the start time
st = time.time()
for i, document_i in enumerate(new_documents):

    # Associate LDA Topics
    topic_probabilities = new_document_topics[i]
    if topic_probabilities.mean() == topic_probabilities.max():
        dominant_topic = "Topic_-1"
    else:
        dominant_topic = "Topic_"+str(topic_probabilities.argmax())

    associated_topic_list.append(dominant_topic)

    # Calculate LM sentiment analysis score
    tokens = lm.tokenize(document_i)
    score = lm.get_score(tokens)
    lm_score = round(score["Polarity"],3)
    lm_scores_list.append(lm_score)




    #Print Process percentage
    percentage = round(((i)/len(new_documents))*100,2)

    if (i == 1)|(i%10000 == 0):
        # get the current time
        ct = time.time()

        # get the execution time
        elapsed_time_seconds = round(ct - st,2)
        elapsed_time_minutes = round(elapsed_time_seconds/60,2)
        print('Processing Chunk: ', i, ' - percentage: ',str(percentage),'%',' (Elapsed time: ',elapsed_time_minutes,' minutes)')



# associated_topic_list
# lm_scores_list

#Consolidating Data Frame

table = {'headline':new_documents,
            "date":dates_list,
            "year":years_list,
            "topic":associated_topic_list,
            "lm_score":lm_scores_list
            }
        
news_lda_lm_df = pd.DataFrame(table, columns = ["headline", "date","year","topic","lm_score"])


lda_lm_ebf_summary = news_lda_lm_df.groupby(["date"])[["lm_score"]].mean().reset_index()

lda_lm_ebf_summary.rename(columns = {'lm_score':'EBF'}, inplace = True)


# Not considering the topic -1 (not associated to any specific topic)
news_lda_lm_df = news_lda_lm_df[news_lda_lm_df["topic"] !="Topic_-1"]


news_lda_lm_summary = news_lda_lm_df.groupby(["date","topic"])[["lm_score"]].mean().reset_index()
news_lda_lm_summary


# Top 30 topics highly correlated in the year 2016
selected_topics = ['Topic_250', 'Topic_51', 'Topic_285', 'Topic_42', 'Topic_159', 'Topic_170', 'Topic_105', 'Topic_267', 'Topic_207', 'Topic_136', 'Topic_150', 'Topic_154', 'Topic_36', 'Topic_197', 'Topic_14', 'Topic_252', 'Topic_225', 'Topic_232', 'Topic_205', 'Topic_118', 'Topic_168', 'Topic_216', 'Topic_132', 'Topic_29', 'Topic_24', 'Topic_280', 'Topic_74', 'Topic_134', 'Topic_62', 'Topic_50']
news_lda_lm_summary_selected = news_lda_lm_summary[news_lda_lm_summary['topic'].isin(selected_topics)]


#transpose
lda_lm_summary = pd.pivot_table(news_lda_lm_summary_selected, values='lm_score', index=['date'], columns=['topic'], aggfunc="first").reset_index()
lda_lm_summary

# Join EBF (general LM sentiment analyis) and per Topic LM

full_lda_lm_summary = pd.merge(lda_lm_summary,lda_lm_ebf_summary, left_on='date', right_on='date', how='inner')
full_lda_lm_summary





#save path
topics_sa_path
save_name = "sa_top_topics_"+"LDATopicModel"+model_year+"Headlines"+model_headlines_count+"k_news"+str(year_i)+".csv"
full_lda_lm_summary.to_csv(topics_sa_path+save_name)




#-------------------------
#-Load LDA LM EBF Data----
#-------------------------


#LOAD
topics_sa_path

# Year Model
model_year = "2016"
model_headlines_count = "250"

#Iterate over files
years = ["2016","2017","2018","2019"]

# For to concatenate the summary files
first_iteration = True
for news_year in years:
    file_name = "sa_top_topics_LDATopicModel" +model_year + "Headlines" +model_headlines_count+ "k_news" + news_year + ".csv"
    current_df = pd.read_csv(topics_sa_path+file_name,index_col=0)

    if first_iteration == True:
        first_iteration = False
        full_lda_lm = current_df.copy()
    else:
        full_lda_lm = pd.concat([full_lda_lm, current_df], ignore_index=True, sort=False)




full_lda_lm['date'] = pd.to_datetime(full_lda_lm['date'], format='%Y-%m-%d')

full_lda_lm = full_lda_lm.sort_values(by=['date'])
full_lda_lm = full_lda_lm.reset_index(drop=True)

# fill na's with 0
full_lda_lm = full_lda_lm.fillna(value=0)
#full_lda_lm.head()
#full_lda_lm.tail()

full_lda_lm


#-------------------------
#-Joining Financial Data--
#-------------------------


gspc_data = pd.read_csv(consolidated_data_path+"GSPC.csv")


selected_columns = ["formatted_date","time_trend","adjclose","returns"]
gspc_df = gspc_data[selected_columns].copy()

gspc_df["formatted_date"] = pd.to_datetime(gspc_df["formatted_date"])

gspc_df['year'] = gspc_df['formatted_date'].dt.year
gspc_df['month'] = gspc_df['formatted_date'].dt.month

#gspc_sample = gspc_df[(gspc_df["year"]==2017)&(gspc_df["month"]==10)]




gspc_prices = gspc_df.copy()

#renaming formatted date column to date
gspc_prices.rename(columns = {'formatted_date':'date'}, inplace = True)



full_lda_lm_prices = pd.merge(full_lda_lm,gspc_prices, left_on='date', right_on='date', how='inner')

full_lda_lm_prices


#save_path
consolidated_data_path
save_data_name = "consolidated_data_LM_LDATopicModel"+model_year+"Headlines"+model_headlines_count+"k.csv"

# sa_and_prices
#full_lda_lm_prices.to_csv(save_data_path+save_data_name)

#Load
full_lda_lm_prices = pd.read_csv(consolidated_data_path+save_data_name)



#-------------------------
#---30 High Corr Topics---
#-------------------------
# ONLY For 2016 YEAR
# Determining the topics to consider


# Topics that are above the date count treshold of 78
topic_counts = news_lda_lm_summary.topic.value_counts().reset_index()
topics_above_treshold = list(topic_counts[topic_counts["topic"]>=78]["index"])
len(topics_above_treshold)
news_lda_lm_summary_2016 = news_lda_lm_summary[news_lda_lm_summary["topic"].isin(topics_above_treshold)]
news_lda_lm_summary_2016



#transpose
lda_lm_summary_2016 = pd.pivot_table(news_lda_lm_summary_2016, values='lm_score', index=['date'], columns=['topic'], aggfunc="first").reset_index()
lda_lm_summary_2016


# Topics that are highly correlated


lda_lm_summary_2016_prices = pd.merge(lda_lm_summary_2016,gspc_prices, left_on='date', right_on='date', how='inner')

correlation_columns = topics_above_treshold
correlation_columns.append("returns")

#Selection of columns and order
corr_matrix = lda_lm_summary_2016_prices[correlation_columns].corr()
corr_matrix["abs_corr_returns"] = abs(corr_matrix["returns"])

corr_matrix = corr_matrix.sort_values(by='abs_corr_returns', ascending=False)

#Top 30
top30matrix = corr_matrix.iloc[1:31]
selected_topics = list(top30matrix.index)
selected_topics

selected_topics = ['Topic_250', 'Topic_51', 'Topic_285', 'Topic_42', 'Topic_159', 'Topic_170', 'Topic_105', 'Topic_267', 'Topic_207', 'Topic_136', 'Topic_150', 'Topic_154', 'Topic_36', 'Topic_197', 'Topic_14', 'Topic_252', 'Topic_225', 'Topic_232', 'Topic_205', 'Topic_118', 'Topic_168', 'Topic_216', 'Topic_132', 'Topic_29', 'Topic_24', 'Topic_280', 'Topic_74', 'Topic_134', 'Topic_62', 'Topic_50']



# Configuring table to present
correlated_topics

top30matrix.reset_index(inplace=True)

columns = ["index","returns"]
correlation_summary = top30matrix[columns]

correlation_summary = correlation_summary.rename(columns={"index": "topic"})

correlation_summary['topic'] = correlation_summary['topic'].str.lower()


lda_top_topics_returns_corr = pd.merge(correlated_topics,correlation_summary,"left",on="topic")

lda_top_topics_returns_corr = lda_top_topics_returns_corr.rename(columns={"returns": "returns_corr"})


lda_top_topics_returns_corr.head(30)
