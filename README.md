# SP500-Direction-Prediction-News-Sentiment-Topic

This repository contains the code and results for the master thesis titled "Topic Modeling and Sentiment Analysis for Predicting the Direction of Movement of the S&P 500 Index Based on Financial News".

## Overview

In this project, topic modeling and sentiment analysis techniques are used to predict the direction of movement of the S&P 500 Index based on financial news data.

## Contents

- `Code/`: This directory contains the Python scripts used for data preprocessing, topic modeling, sentiment analysis, classfication model training, grid search procedure and the evaluation of the classifications
- `Data/`: This directory contains the processed datasets.
- `Results/`: This directory contains the results of the analysis in csv and an excel file that consolidates the results in tables format.


## Dataset
The dataset used in this project is the "All the News 2" news articles dataset created by Andrew Thompson. The dataset is available [here](https://components.one/datasets/all-the-news-2-news-articles-dataset). The dataset contains more than 2.5 million news articles from 27 publishers, spanning from January 2016 to April 2020.

## Usage

The different steps of the thesis can be replicated, including data preparation, sentiment analysis, topic modeling, classification model training and evaluation. To replicate the results of the thesis, follow these steps:

1. Install the required dependencies listed in `requirements.txt` and the utility script `A_ThesisFunctions.py`
2. Download the relevant data from the `/Data` directory. 
3. Run the desired code corresponding to the step you want to replicate. You may need to adjust the code with desired parameters or modify the paths to read the input data and store the results.
4. Explore the obtained results to analyze the findings of the replicated steps.

## Code Overview
This repository contains multiple scripts, each serving a specific purpose in the analysis pipeline. Here's a brief overview of each set of scripts:

### Utility Script
- `A_ThesisFunctions.py`: Contains utility functions used across multiple scripts, such as data loading, processing, saving, and others.

### Data Collection and Preparation
- `B_DataPreprocessing_FinancialData.py`: Extracting info from the yahoo library to extract the S&P500 adjclose prices for 2016 to 2020.
- `C_DataPreprocessing_News.py`: Reading the csv file of AllTheNews2.0, preprocessing them and saving them in parquets files.

### Correlation Analysis: Financial data and Sentiment Analysis by Publisher Category 
- `D_FinBERT_Publishers_Categories.py`: Categorizing publishers type and storing a summary of the sentiment score per day per publisher type. 1 File per yearmonth stored in `\Data\sa_publishers_category`. Example: `sa_publishers_category_201601.csv`
- `E_FinBERT_Publishers_SP500_Corr.py`: Correlation analysis between the sentiment score of the publisher type and the returns of the S&P500

### Sentiment Analysis
- `F_FinBERT_EBF_Headlines.py`: Executing and storing the sentiment analysis for all the EBF news. Parquet Files. (This intermediary dataset is not included in this repository due to its large size)

### Topic Modeling
- `G_BERTopic_News.py`: Training and storing the topic models. LDA topics are stored in pickle files. Examples: `BERTopicModel2016Sample250k` and `LDAmodel2016Headlines250k30Topics.pkl`

### Correrlation Analysis for Topics Selection: Financial data, Sentiment Analysis and Topics 
- `H_FinBERT_BERTopic_EBF_SP500_Corr.py`: Correlation analysis between the sentiment score of the topics and the returns of the S&P500
- `I_FinBERT_BERTopic_EBF_Top_Topics_Summary.py`: Consolidating the sentiment score for each day for each topic (top 30 topics). Saves a file per year. Obtained results are stored in `Data\sa_top_topics`. Example: `sa_top_topics_BERTopicModel2016Bodytext250k_news2019.csv`

### Data Consolidation: BERT based models and Financial Data
- `J_Consolidated_Data_FinBERT_BERTopic_EBF.py`: Consolidates FinBERT_BERTopic and Financial Data. Save a file considering the 4 years. Obtained results are stored in 
 `Data\consolidated_data`. Example: `consolidated_data_FinBERT_BERTopicModel2016Sample250k.csv`

### Data Consolidation: LDA-LM based models and Financial Data
- `K_LDA_LM_EBF.py`: Consolidates LDA TopicModel and Loughran MacDonald dictionary sentiment analysis and Financial Data. Save a file considering the 4 years. Obtained results are stored in `Data\consolidated_data`.  Example: `consolidated_data_LM_LDATopicModel2016Sample250k.csv` 

### Classification Models
- `L_Pred_Models_Exploration.py`: Sandbox for exploring the prediction models, parameters and its use application with the data
- `M_Pred_Models_Evaluation.py`: Consolidating the results of the grid search procedure and the evaluation of the different prediction models. Obtained results are stored in `Data/Results`. Examples: `consolidated_results4.xlsx` and `fine_tuning_df4.xlsx`
