# -*- coding: utf-8 -*-
"""
Created on June 14

@author: Miguel Azuero

Data Download from https://components.one/datasets/all-the-news-2-news-articles-dataset

Dropbox link: https://www.dropbox.com/scl/fi/ri2muuv85ri98odyi9fzn/all-the-news-2-1.zip?rlkey=8qeq5kpg5btk3vq5nbx2aojxx&e=1&dl=0


Process could take around 40 minutes for downloading and unziping the file

"""

import requests
import zipfile

from tqdm import tqdm
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
#----Download Functions---
#-------------------------

def download_file_from_dropbox(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024  # 1 KB
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


#-------------------------
#----Parameters-----------
#-------------------------

dropbox_url = "https://www.dropbox.com/scl/fi/ri2muuv85ri98odyi9fzn/all-the-news-2-1.zip?rlkey=8qeq5kpg5btk3vq5nbx2aojxx&e=1&dl=1"
# Path to save the downloaded file
zip_file_name = "all-the-news-2-1.zip"


#-------------------------
#------Code Execution-----
#-------------------------

zip_file_path = os.path.join(large_data_path,zip_file_name)
news_path = os.path.join(large_data_path,'all-the-news-2-1/')

# Ensure the directory exists if not creates it
if not os.path.exists(large_data_path):
    os.makedirs(large_data_path)

# Download the file
print("Downloading file...")
download_file_from_dropbox(dropbox_url, zip_file_path)
print("Download complete.")

# Unzip the file
print("Unzipping file...")
unzip_file(zip_file_path, news_path)
print("Unzipping complete.")

# delete the zip file after extraction
os.remove(zip_file_path)