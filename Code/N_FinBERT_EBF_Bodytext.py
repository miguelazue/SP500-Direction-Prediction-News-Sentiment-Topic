# -*- coding: utf-8 -*-
"""

Created on July 7 8:59:23 2023

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
ebf_sa_full_article_path = os.path.join(heavy_data_path,'SentimentAnalysisFullArticleEBF/')
topics_path = os.path.join(heavy_data_path,'BERTopics/')


#-------------------------
#-Load Utility Script-----
#-------------------------

os.chdir(current_dir)
import A_ThesisFunctions as tf



#-------------------------
#---SENTIMENT ANALYSIS TEST FOR FULL ARTICLE----
#-------------------------

#Function to do sentiment analysis on the preprocessed data frame

#Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
#Load Finbert Model
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


# Sentiment analysis on a larg string, recursive function. Splits in Half the text.
# Returns the prediction scores for the sentiment analysis: [postivie,negative,neutral]

def sentiment_analysis_on_large_str(str_to_analyze):
    """Recursive Function that splits in half the string to analyz and average the FinBERT score between the 2 splits. Splits until sizes are below 512"""

    """
    Parameters
    ---------
    str_to_analyze: Data frame that is going to be processed with the sentiment analysis
    ---------
    """
    current_text = str_to_analyze
    # current_text = text
    input_id = tokenizer.encode(current_text, return_tensors="pt")
    my_tensor_size = input_id.size()[1]

    if my_tensor_size > 500:
        print("Splitting text to apply FinBERT Sentiment Analysis ")
        half_position = int(round(len(current_text)/2,0))

        first_split =current_text[:half_position]
        second_split =current_text[half_position:]

        # Find a dot around the half. So that the splits contain full sentences
        # If doesn't find a dot, the split will cut the text in exact half cutting the sentence at the split position
        # Cases like U.S. or dots that don't end the sentence will be cut.
        dot_position = first_split.find(".",len(first_split)-200)
        if dot_position != -1:
            first_split = current_text[:dot_position]
            second_split = current_text[dot_position+1:]

        first_input_id = tokenizer.encode(first_split, return_tensors="pt")
        second_input_id = tokenizer.encode(second_split, return_tensors="pt")

        first_tensor_size = first_input_id.size()[1]
        second_tensor_size = second_input_id.size()[1]


        # Test if the first split is lower than the max, if its hight, calls itself and do the same process.

        #First Split
        if first_tensor_size > 500:
            first_predictions = sentiment_analysis_on_large_str(first_split)
            # print("Splitting First half: Recursion")
        else:
            first_output = finbert_model(first_input_id).logits
            first_predictions = torch.nn.functional.softmax(first_output, dim=-1).tolist()

        #Second Split
        if second_tensor_size > 500:
            second_predictions = sentiment_analysis_on_large_str(second_split)
            # print("Splitting Second half: Recursion")
        else:
            second_output = finbert_model(second_input_id).logits
            second_predictions = torch.nn.functional.softmax(second_output, dim=-1).tolist()

        # Recursive function
        #Try FinBERT
        positive_avg_score = (first_predictions[0][0] + second_predictions[0][0])/2
        negative_avg_score = (first_predictions[0][1] + second_predictions[0][1])/2
        neutral_avg_score = (first_predictions[0][2] + second_predictions[0][2])/2


        avg_prediction = [[]]
        avg_prediction[0] = [positive_avg_score,negative_avg_score,neutral_avg_score]
    else:
        output = finbert_model(input_id).logits
        predictions = torch.nn.functional.softmax(output, dim=-1).tolist()
        avg_prediction = predictions
    return(avg_prediction)






def sentiment_analysis_on_df_full_article(df_to_analyze):
    """Function that reads news from a dataframe and process the sentiment analysis"""

    """
    Parameters
    ---------
    df_to_analyze: Data frame that is going to be processed with the sentiment analysis
    ---------
    """
    news_df_current = df_to_analyze.copy()

    titles_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
    dates_list = []
    years_list =[]
    # get the start time
    st = time.time()


    for i in range(len(news_df_current)):
        text = news_df_current.iloc[i]['article']


        #Validate title is not empty
        if text != None:
            # Tokenize the text
            #print(i)
            
            input_id = tokenizer.encode(text, return_tensors="pt")
            my_tensor_size = input_id.size()[1]

            if my_tensor_size > 1000:
                print(my_tensor_size, i)

            predictions = sentiment_analysis_on_large_str(text)


            titles_list.append(news_df_current['title'].iloc[i][:50])
            dates_list.append(news_df_current['associated_date'].iloc[i])
            years_list.append(news_df_current['year'].iloc[i])
            positive_list.append(predictions[0][0])
            negative_list.append(predictions[0][1])
            neutral_list.append(predictions[0][2])
            #if i > 2000:
            #    break
                

        #Print Process percentage
        percentage = round(((i)/len(news_df_current))*100,2)

        if (i == 1)|(i%100 == 0):
            # get the current time
            ct = time.time()

            # get the execution time
            elapsed_time_seconds = round(ct - st,2)
            elapsed_time_minutes = round(elapsed_time_seconds/60,2)
            print('Processing Chunk: ', i, ' - percentage: ',str(percentage),'%',' (Elapsed time: ',elapsed_time_minutes,' minutes)')


    

    table = {'headline':titles_list,
            "date":dates_list,
            "year":years_list,
            "positive":positive_list,
            "negative":negative_list, 
            "neutral":neutral_list
            }
        
    news_finbert_sa = pd.DataFrame(table, columns = ["headline", "date","year","positive", "negative", "neutral"])

    #news_finbert_sa.head()
    #calculating sentiment score
    news_finbert_sa["sa_score_full_article"]= news_finbert_sa["positive"] - news_finbert_sa["negative"]


    print('finished sentiment analysis')
    
    return(news_finbert_sa)


#-------------------------
#---------ITERATION-------
#-------------------------



#Iterate over periods
#years = [2016,2017,2018,2019,2020]
years = [2016]
#months = [1,2,3,4]
#months = [1,2,3,4,5,6,7,8,9,10,11,12]
months = [1]


for iteration_year in years:
    for iteration_month in months:

        print("Processing Year "+str(iteration_year)+" - month "+str(iteration_month))

        #Read all parquets
        dataset = pq.ParquetDataset(ebf_parquet_path,use_legacy_dataset=False,filters=[("year","=",iteration_year)])

        #To Pandas by selecting Columns
        news_df = dataset.read(columns=["date","title","article","publication","section","year"]).to_pandas()

        #Converting date to datetime format
        news_df['date'] = pd.to_datetime(news_df['date'], format='%Y-%m-%d')

        news_df = news_df[(news_df['date'].dt.year == iteration_year)&(news_df['date'].dt.month == iteration_month)]
        
        df_size = round(round(news_df.memory_usage().sum()/1024,2)/1024,2)

        number_of_news = len(news_df)
        print("Dataframe size:", df_size,"MB"," - ",number_of_news," news")

        
        # get the start time
        st = time.time()


        #news_df = news_df.rename(columns={"article": "title"})
        preprocessed_df = tf.preprocess_news_date(news_df)

        preprocessed_df.head()
        
        #Sentiment Analysis on Dataframe
        #sa_df = sentiment_analysis_on_df(preprocessed_df.iloc[:100])
        sa_df = sentiment_analysis_on_df_full_article(preprocessed_df)
        # get the end time
        et = time.time()

        # get the execution time
        elapsed_time_seconds = round(et - st,2)
        elapsed_time_minutes = round(elapsed_time_seconds/60,2)
        print("Execution time:", elapsed_time_seconds, "seconds")
        print("Execution time:", elapsed_time_minutes, "minutes")

        #SAVE
        current_path = ebf_sa_full_article_path+str(iteration_year)+"/"+str(iteration_year*100+iteration_month)+"/"
        
        tf.df_to_multiple_plarquets(df_to_slice = sa_df,save_path = current_path)




# 1:17 pm


#-------------------------
#-----Read Results--------
#-------------------------

#dataset1 = pq.ParquetDataset(ebf_sa_full_article_path,use_legacy_dataset=False)
#new_test = dataset1.read(columns=["headline","date","year","sa_score"]).to_pandas()
#Converting date to datetime format
#new_test['date'] = pd.to_datetime(new_test['date'], format='%Y-%m-%d')

#new_test.head()
#new_test.tail()

#TEST



#Testing Function

# text = preprocessed_df["article"][15]
# text = preprocessed_df["article"][31]
# text


# input_id = tokenizer.encode(text, return_tensors="pt")
# my_tensor_size = input_id.size()[1]




# test_prediction = sentiment_analysis_on_large_str(text)
# test_prediction = sentiment_analysis_on_large_str((text+" ")*10)
# sum(test_prediction[0])


# sa_df = sentiment_analysis_on_df_full_article(preprocessed_df.head())

# sa_df['headline'].iloc[2][:50]