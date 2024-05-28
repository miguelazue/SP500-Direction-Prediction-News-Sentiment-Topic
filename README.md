# SP500-Direction-Prediction-News-Sentiment-Topic

This repository contains the code and results for my master thesis titled "Topic Modeling and Sentiment Analysis for Predicting the Direction of Movement of the S&P 500 Index Based on Financial News".

## Overview

In this project, I explored the use of topic modeling and sentiment analysis techniques to predict the direction of movement of the S&P 500 Index based on financial news data.

## Contents

- `code/`: This directory contains the Python scripts used for data preprocessing, topic modeling, sentiment analysis, and model training.
- `data/`: This directory contains the dataset used for training and evaluation.
- `results/`: This directory contains the results of the analysis in csv and an excel file that consolidates the results in tables format.


## Dataset
The dataset used in this project is the "All the News 2" news articles dataset, which can be found [here](https://components.one/datasets/all-the-news-2-news-articles-dataset). This dataset contains a comprehensive collection of news articles from various sources.

## Usage

To replicate the results of the thesis, follow these steps:

1. Install the required dependencies listed in `requirements.txt`.
2. Run the preprocessing scripts to clean and prepare the data.
3. Use the topic modeling and sentiment analysis scripts to extract features from the text data.
4. Train and evaluate predictive models using the processed data.
5. Explore the results and visualizations in the `results/` directory.

## Code Overview

### A_ThesisFunctions.py
Consolidation of the different functions for the execution of  the different codes.



### B_DataPreprocessing_FinancialData.py
Extracting info from the yahoo library to extract the S&P500 adjclose prices for 2016 to 2020
GSPC.xlsx


### C_DataPreprocessing_News.py
Reading the csv file of AllTheNews2.0, preprocessing them and saving them in parquets files.
C:\Users\migue\Downloads\Thesis Data\FilteredNewsParquets


### D_FinBERT_Publishers_Categories.py
Categorizing publishers type and storing a summary of the sentiment score per day per publisher type.
1 File per yearmonth
C:\Users\migue\Dropbox\JKU EBA Masters\Master Thesis\Data\sa_publishers_category
sa_publishers_category_201601.csv

### E_FinBERT_Publishers_SP500_Corr.py
Understanding the correlation between the sentiment score of the publisher type and the returns of the S&P500

### F_FinBERT_EBF_Headlines.py
Executing and storing the sentiment analysis for all the EBF news. Parquet Files
C:\Users\migue\Downloads\Thesis Data\SentimentAnalysisEBF


### G_BERTopic_News.py
Saving the topic models
C:\Users\migue\Downloads\Thesis Data\BERTopics
BERTopicModel2016Sample250k


### H_FinBERT_BERTopic_EBF_SP500_Corr.py
Understanding the correlation between the sentiment score of the topics and the returns of the S&P500



### I_FinBERT_BERTopic_EBF_Top_Topics_Summary.py
Save the sentiment score for each day for each topic (top 30 topics), Save a file per year
C:\Users\migue\Dropbox\JKU EBA Masters\Master Thesis\Data\sa_top_topics

### J_Consolidated_Data_FinBERT_BERTopic_EBF.py
Consolidates FinBERT_BERTopic and Financial Data,Save 1 file considering the 4 years.
consolidated_data_FinBERT_BERTopicModel2016Sample250k.csv
C:\Users\migue\Dropbox\JKU EBA Masters\Master Thesis\Data\consolidated_data

### K_LDA_LM_EBF.py
Consolidates LDA TopicModel and Loughran MacDonald dictionary sentiment analysis and Financial Data.
Save 1 file considering the 4 years.
C:\Users\migue\Dropbox\JKU EBA Masters\Master Thesis\Data\consolidated_data
consolidated_data_LM_LDATopicModel2016Sample250k.csv

### L_Pred_Models_Exploration.py
Sandbox for exploring the prediction models, parameters and use application to the data


### M_Pred_Models_Evaluation.py
Consolidating the results of the prediction models
consolidated_results.xlsx
