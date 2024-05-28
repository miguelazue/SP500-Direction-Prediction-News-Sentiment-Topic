# SP500-Direction-Prediction-News-Sentiment-Topic

This repository contains the code and results for my master thesis titled "Topic Modeling and Sentiment Analysis for Predicting the Direction of Movement of the S&P 500 Index Based on Financial News".

## Overview

In this project, I explored the use of topic modeling and sentiment analysis techniques to predict the direction of movement of the S&P 500 Index based on financial news data.

## Contents

- `code/`: This directory contains the Python scripts used for data preprocessing, topic modeling, sentiment analysis, and model training.
- `data/`: This directory contains the dataset used for training and evaluation.
- `results/`: This directory contains the results of the analysis in csv and an excel file that consolidates the results in tables format.


## Dataset
The dataset used in this project is the "All the News 2" news articles dataset created by Andrew Thompson. The dataset is available [here](https://components.one/datasets/all-the-news-2-news-articles-dataset). The dataset contains more than 2.5 million news articles from 27 publishers, spanning from January 2016 to April 2020.

## Usage

To replicate the results of the thesis, follow these steps:

1. Install the required dependencies listed in `requirements.txt`.
2. Download the data relevant to the code to be executed.
3. Run the code of interest scripts to clean and prepare the data.
4. Use the topic modeling and sentiment analysis scripts to extract features from the text data.
5. Train and evaluate predictive models using the processed data.
6. Explore the results and visualizations in the `results/` directory.

## Code Overview
This repository contains multiple scripts, each serving a specific purpose in the analysis pipeline. Here's a brief overview of each set of scripts:

- `A_ThesisFunctions.py`: Contains utility functions used across multiple scripts, such as data loading, processing, saving, and others.
- `B_DataPreprocessing_FinancialData.py`: Extracting info from the yahoo library to extract the S&P500 adjclose prices for 2016 to 2020. GSPC.xlsx
- `C_DataPreprocessing_News.py`: Reading the csv file of AllTheNews2.0, preprocessing them and saving them in parquets files.
- `D_FinBERT_Publishers_Categories.py`: Categorizing publishers type and storing a summary of the sentiment score per day per publisher type. 1 File per yearmonth stored in \Data\sa_publishers_category. Example: sa_publishers_category_201601.csv
- `E_FinBERT_Publishers_SP500_Corr.py`: Understanding the correlation between the sentiment score of the publisher type and the returns of the S&P500
- `F_FinBERT_EBF_Headlines.py`: Executing and storing the sentiment analysis for all the EBF news. Parquet Files. (This intermediary dataset is not included in this repository due to its large size)
- `G_BERTopic_News.py`: Training and storing the topic models. Example: BERTopicModel2016Sample250k
- `H_FinBERT_BERTopic_EBF_SP500_Corr.py`: Correlation analysis between the sentiment score of the topics and the returns of the S&P500
- `I_FinBERT_BERTopic_EBF_Top_Topics_Summary.py`: Consolidating the sentiment score for each day for each topic (top 30 topics), Save a file per year. Data\sa_top_topics
- `J_Consolidated_Data_FinBERT_BERTopic_EBF.py`: Consolidates FinBERT_BERTopic and Financial Data,Save 1 file considering the 4 years. Data\consolidated_data  Example: consolidated_data_FinBERT_BERTopicModel2016Sample250k.csv
- `K_LDA_LM_EBF.py`: Consolidates LDA TopicModel and Loughran MacDonald dictionary sentiment analysis and Financial Data. Example: consolidated_data_LM_LDATopicModel2016Sample250k.csv
Save 1 file considering the 4 years. Data\consolidated_data
- `L_Pred_Models_Exploration.py`: Sandbox for exploring the prediction models, parameters and use application to the data
- `M_Pred_Models_Evaluation.py`: Consolidating the results of the prediction models. Data/Results/consolidated_results.xlsx
