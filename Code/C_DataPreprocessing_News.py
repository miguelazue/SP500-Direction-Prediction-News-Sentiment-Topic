# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:06:18 2023

@author: Miguel Azuero

Data Transformation and Preprocessing

News Source:  https://components.one/datasets/all-the-news-2-news-articles-dataset/

"""


import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

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
#-----Saving Parquets-----
#-------------------------
# Due to its large size, it is preprocessed to beb saved in parquet files

file1 = "all-the-news-2-1.csv"

news_path = os.path.join(large_data_path,'all-the-news-2-1/')

filename = news_path+file1


type_dict = {
    'year':'int16',
    #'month':'int16',
    #'day':'int16',
    'title': 'str',
    'article': 'str',
    'section': 'str',
    'publication': 'str'
    }


#my_chunksize = 10 ** 5
my_chunksize = 10 ** 4
my_file = pd.read_csv(filename, chunksize=my_chunksize,dtype=type_dict, parse_dates=['date'])




#Function that Iterate over chunks and filter the data to ease the size of the dataset

def read_and_filter_chunks(df_chunks,relevant_columns,relevant_publishers,chunks_size=10 ** 4, n_total_news = 2688878):
    """Reads and filter the News File by the selected chunk size and publishers. Returns Filtered Dataframe"""

    """
    Parameters
    ---------
    df_chunks: TextReaderFile with the desired df
    relevant_columns: selected columns to extract
    relevant_publishers: selected publishers to filter in the dataframe
    n_total_news: Total number of news of the file
    chunks_size: number of rows stored in one chunk of parquet file. Defaults to 10,000.
    ---------
    """

    n_total_chunks = np.ceil(n_total_news / chunks_size)


    filtered_news = pd.DataFrame()
    #stop_iteration = 3
    stop_iteration = -1
    iteration = 0      
    
    for chunk in df_chunks:
        iteration = iteration + 1
        
        #Obtaining relevant columns
        current_chunk = chunk[relevant_columns].copy()
        
        #Changing date format
        current_chunk['date'] = current_chunk['date'].dt.strftime('%Y-%m-%d')
        
        #Filtering by publisher and section
        #filtered_chunk = current_chunk[current_chunk['publication'].isin(relevant_publishers)&current_chunk['section'].isin(relevant_sections)]
        
        #filtering only relevant publishers
        filtered_chunk = current_chunk[current_chunk['publication'].isin(relevant_publishers)]
    
        #filtering only relevant sections
        #filtered_chunk = current_chunk[current_chunk['section'].isin(relevant_sections)]
    
        #Print Process percentage
        percentage = round(((iteration)/n_total_chunks)*100,2)
        if (iteration == 1)|(iteration%10 == 0):
            print('Processing Chunk: ', iteration, ' - percentage: ',str(percentage),'%')
    
    #Gathering the filtered news in a dataframe    
        if iteration == 1:
            filtered_news = filtered_chunk.copy()
        else:
            filtered_news = pd.concat([filtered_news,filtered_chunk],ignore_index = True)
        
        if iteration == stop_iteration:
            break
        
    print('Process Completed')
    return(filtered_news)





#-------------------------
#-----Saving Parquets-----
#-------------------------



#SAVE

#Size of the file in MB
#round(filtered_news.memory_usage().sum()/1024,2)/1024
#filtered_news.info(verbose = False)


#Function for saving a dataset in multiple parquets

def df_to_multiple_plarquets(df_to_slice,save_path, n_slices = 10):
    """Function that saves  dataframe in mulitple parquet files in the desired directory (partition by year)"""

    """
    Parameters
    ---------
    parquet_path: Desired path to save the parquet files
    n_slices: Number of parquet files to divide the dataframe
    ---------
    """
    
    slice_size = int(np.ceil(len(df_to_slice)/n_slices))

    #slicing for saving in multiple parquets
    for i in range(0, len(df_to_slice), slice_size):
        i_slice = int(i/slice_size)
        print("Saving slice: ",i_slice+1)
        df_slice = df_to_slice.iloc[i : i + slice_size]
        table = pa.Table.from_pandas(df_slice)
        pq.write_to_dataset(table,root_path= save_path, partition_cols=["year"])

    return("Parquets saved")





#Reading, filtering and saving General News



#Business, Financials and Politics Publishers
ebf_publishers = [
    "Business Insider",
    "CNBC",
    "Economist",
    "Reuters",
    "TechCrunch",
]

#GENERAL Breaking News and Current Affairs  (General News)
gen_publishers = [
    "Axios",
    "CNN",
    "Fox News",
    "The New York Times",
    "Vice News",
    "Vox",
    "Washington Post"
]

#Politics
pol_publishers = [
    "New Republic",
    "New Yorker",
    "Politico",
    "The Hill",
]

#Entertainment
ent_publishers = [
    "Buzzfeed News",
    "Gizmodo",
    "Hyperallergic",
    "Mashable",
    "People",
    "Refinery 29",
    "TMZ",
    "The Verge",
    "Vice",
    "Wired"
]


selected_columns = [
    "date",
    "year",
    #"month",
    #"day",
    "title",
    "article",
    "section",
    "publication"
    ]



#Executing functions for type of publishers

#EBF
ebf_filtered_news = read_and_filter_chunks(df_chunks=my_file,relevant_columns = selected_columns, relevant_publishers = ebf_publishers)
ebf_parquet_path = os.path.join(large_data_path,'FilteredNewsParquets/EBF_News/')
df_to_multiple_plarquets(df_to_slice = ebf_filtered_news,save_path=ebf_parquet_path)

#GEN
gen_filtered_news = read_and_filter_chunks(df_chunks=my_file,relevant_columns = selected_columns, relevant_publishers = gen_publishers)
gen_parquet_path = os.path.join(large_data_path,'FilteredNewsParquets/GEN_News/')
df_to_multiple_plarquets(df_to_slice = gen_filtered_news,save_path=gen_parquet_path)

#POL
pol_filtered_news = read_and_filter_chunks(df_chunks=my_file,relevant_columns = selected_columns, relevant_publishers = pol_publishers)
pol_parquet_path = os.path.join(large_data_path,'FilteredNewsParquets/POL_News/')
df_to_multiple_plarquets(df_to_slice = pol_filtered_news,save_path=pol_parquet_path)

#ENT
ent_filtered_news = read_and_filter_chunks(df_chunks=my_file,relevant_columns = selected_columns, relevant_publishers = ent_publishers)
ent_parquet_path = os.path.join(large_data_path,'FilteredNewsParquets/ENT_News/')
df_to_multiple_plarquets(df_to_slice = ent_filtered_news,save_path=ent_parquet_path)



#DATA OVERVIEW
#bfp_filtered_news.head()
#bfp_filtered_news.tail()

#bfp_filtered_news.dtypes
#bfp_filtered_news.memory_usage()
#bfp_filtered_news.info(verbose = False)

#round(bfp_filtered_news.memory_usage().sum()/1024,2)/1024

#bfp_filtered_news.publication.value_counts()
#bfp_filtered_news.section.value_counts()
#pd.DatetimeIndex(bfp_filtered_news['date']).year.value_counts()

#Overview of sections
#my_sections = bfp_filtered_news.section.value_counts()
#print(my_sections)
#bfp_filtered_news['section'].isna().sum()


#-------------------------
#----Reading Parquets-----
#-------------------------


import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq


#Paths for the other publisher types
gen_parquet_path = os.path.join(large_data_path,'FilteredNewsParquets/GEN_News/')
pol_parquet_path = os.path.join(large_data_path,'FilteredNewsParquets/POL_News/')
ent_parquet_path = os.path.join(large_data_path,'FilteredNewsParquets/ENT_News/')

#Read all Parquets
dataset = pq.ParquetDataset(parquet_path,use_legacy_dataset=False)

#Read EBF Parquets
#dataset = pq.ParquetDataset(ebf_parquet_path,use_legacy_dataset=False)

#Read GEN Parquets
#dataset = pq.ParquetDataset(gen_parquet_path,use_legacy_dataset=False)

#Read ENT Parquets
#dataset = pq.ParquetDataset(ent_parquet_path,use_legacy_dataset=False)

#Read POL Parquets
#dataset = pq.ParquetDataset(pol_parquet_path,use_legacy_dataset=False)

#Read by Filtered year
#dataset = pq.ParquetDataset(parquet_path,use_legacy_dataset=False,filters=[("year","=",2018)])

#Read by Publisher
#dataset = pq.ParquetDataset(parquet_path,use_legacy_dataset=False,filters=[("publication","=","Reuters")])


#To Pandas by selecting Columns
news_df = dataset.read(columns=["date","title","publication","section","year"]).to_pandas()

#Converting date to datetime format
news_df['date'] = pd.to_datetime(news_df['date'], format='%Y-%m-%d')

news_df = news_df.sort_values(by=['date'], ascending=True)


news_df.publication.value_counts()

news_df.year.value_counts()

news_df.groupby(["publication"])["date"].min()

#In case of reading the whole article, is better to open the dataset per year (per partition)
#dataset.read(columns=["date","article"]).to_pandas()

#To Pandas selecting all columns
#dataset.read().to_pandas()



#-------------------------
#----Data Overview--------
#-------------------------


len(news_df)

news_df.head()
news_df.tail()

news_df.dtypes

round(news_df.memory_usage().sum()/1024,2)/1024

news_df.info(verbose = False)

news_df.publication.value_counts()
news_df.section.value_counts()
pd.DatetimeIndex(news_df['date']).year.value_counts()


#Evaluating Tilburg university Paper - Data set selection

relevant_sections = [
    "Business News",
    "Financials"
]


news_df[news_df["publication"] == "Economist"].section.value_counts().head(10)
news_df[news_df["publication"] == "Business Insider"].section.value_counts().head(10)
news_df[news_df["publication"] == "The New York Times"].section.value_counts().head(10)

test2 = news_df.section.value_counts()
test2.head(30)


test3 = news_df[(news_df["publication"] == "The New York Times") & (news_df["section"] == "us")]
test3


selected_df = news_df[news_df["section"].isin(relevant_sections)]
selected_df.publication.value_counts()





