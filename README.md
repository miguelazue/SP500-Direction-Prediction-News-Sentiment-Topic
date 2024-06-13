# SP500-Direction-Prediction-News-Sentiment-Topic

This repository contains the code and results for the master thesis titled "Topic Modeling and Sentiment Analysis for Predicting the Direction of Movement of the S&P 500 Index Based on Financial News".

## Overview

In this project, topic modeling and sentiment analysis techniques are used to predict the direction of movement of the S&P 500 Index based on financial news data.

## Contents

- `Code/`: This directory contains the Python scripts used for data preprocessing, topic modeling, sentiment analysis, classfication model training, grid search procedure and the evaluation of the classifications
- `Data/`: This directory contains the processed datasets and the results stored in csv format
- `Formatted_Results/`: This directory contains the results presented in a format that is easy to read. Excel files with summarized results.
- `requirements.txt`: Requirements for the code execution
- `A_ThesisFunctions.py`: Utility script that contains functions that are used across multiple scripts, such as data loading, processing, saving, and others.


## Dataset
The dataset used in this project is the "All the News 2.0". Consolidation of news articles dataset created by Andrew Thompson. The dataset is available [here](https://components.one/datasets/all-the-news-2-news-articles-dataset). The dataset contains more than 2.5 million news articles from 27 publishers, spanning from January 2016 to April 2020.

## Usage

The thesis can be fully replicated or specific steps of the thesis pipeline including data preparation, sentiment analysis, topic modeling, classification model training and classification model evaluation. To replicate the results of the thesis, follow these steps:

1. Install the required dependencies listed in `requirements.txt` and download the utility script `A_ThesisFunctions.py`
2. Download the [dataset](https://components.one/datasets/all-the-news-2-news-articles-dataset) and save it in the `Large_Data/` directory if you want to replicate all the thesis. If a specific step of the thesis wants to be replicated, you can download the already preprocessed data from the `Data/` directory. 
4. Run the desired code corresponding to the step you want to replicate. Parameter values can be adjusted in the parameters section.
5. Explore the obtained results to analyze the findings of the replicated steps.

*_Path Definition Recommendations_:  At the beginning of every script, there is a section named "Path Definition," where you can define paths using the current working directory or define a specific directory path. Some files are large; for example, the "All the News 2.0" dataset is 8.8 GB, and a trained topic model can exceed 1 GB in size. Considering this, a separate directory is defined for these large files, referred to as the "Large Data" directory. Due to the large size of these files, it is recommended to store the "Large_Data" directory locally instead of using cloud storage (e.g., Dropbox or GitHub). However, if space is not an issue, the "Large_Data" directory can also be defined within the same working directory.


## Code Overview
This repository contains multiple scripts, each serving a specific purpose in the analysis pipeline. Here's a brief overview of each set of scripts:

### Utility Script
- `A_ThesisFunctions.py`: Contains utility functions used across multiple scripts, such as data loading, processing, saving, and others.

### Data Collection and Preparation
- `B_DataPreprocessing_FinancialData.py`: Extracting info from the yahoo library to extract the S&P500 adjclose prices for 2016 to 2020.
- `C_DataPreprocessing_News.py`: Reading the csv file of AllTheNews2.0, preprocessing them and saving them in parquets files. (The resulting processed files will be stored in the `Large_Data/` directory)

### Correlation Analysis: Financial data and Sentiment Analysis by Publisher Category 
- `D_FinBERT_Publishers_Categories.py`: Categorizing publishers type and storing a summary of the sentiment score per day per publisher type. 1 File per yearmonth stored in `/Data/sa_publishers_category/`. Example: `sa_publishers_category_201601.csv`
- `E_FinBERT_Publishers_SP500_Corr.py`: Correlation analysis between the sentiment score of the publisher type and the returns of the S&P500

### Sentiment Analysis
- `F_FinBERT_EBF_Headlines.py`: Executing and storing the sentiment analysis for all the EBF news in Parquet Files. (The resulting processed files will be stored in the `Large_Data/` directory)

### Topic Modeling
- `G_BERTopic_News.py`: Training and storing the topic models. LDA topics are stored in pickle files. Examples: `BERTopicModel2016Sample250k` and `LDAmodel2016Headlines250k30Topics.pkl`.(The resulting processed files will be stored in the `Large_Data/` directory)

### Correlation Analysis for Topics Selection: Financial data, Sentiment Analysis and Topics 
- `H_FinBERT_BERTopic_EBF_SP500_Corr.py`: Correlation analysis between the sentiment score of the topics and the returns of the S&P500
- `I_FinBERT_BERTopic_EBF_Top_Topics_Summary.py`: Consolidating the sentiment score for each day for each topic (top 30 topics). Saves a file per year. Obtained results are stored in `Data/sa_top_topics/`. Example: `sa_top_topics_BERTopicModel2016Bodytext250k_news2019.csv`

### Data Consolidation: BERT based models and Financial Data
- `J_Consolidated_Data_FinBERT_BERTopic_EBF.py`: Consolidates FinBERT_BERTopic and Financial Data. Save a file considering the 4 years. Obtained results are stored in 
 `Data/consolidated_data/`. Example: `consolidated_data_FinBERT_BERTopicModel2016Sample250k.csv`

### Data Consolidation: LDA-LM based models and Financial Data
- `K_LDA_LM_EBF.py`: Consolidates LDA TopicModel and Loughran MacDonald dictionary sentiment analysis and Financial Data. Save a file considering the 4 years. Obtained results are stored in `Data/consolidated_data/`.  Example: `consolidated_data_LM_LDATopicModel2016Sample250k.csv` 

### Classification Models
- `L_Pred_Models_Exploration.py`: Sandbox for exploring the prediction models, parameters and its use application with the data
- `M_Pred_Models_Evaluation.py`: Consolidating the results of the grid search procedure and the evaluation of the different prediction models. Obtained results are stored in `Data/consolidated_results/`. Examples: `consolidated_results4.xlsx` and `fine_tuning_df4.xlsx`

### Data Consolidation: BERT based models adaptation to Bodytext
While the previous codes can be executed on a regular laptop as they used the headlines as inputs, it's recommended to use a machine with higher processing power for evaluating the bodytext of news articles.
- `N_FinBERT_EBF_Bodytext.py`: Adaptation for calculating the sentiment score of the bodytext instead of headlines.
- `O_BERTopic_News_Bodytext.py`: Adaptation for training topic models using the bodytext instead of headlines.


## Directories Overview

### Data - Directory
This directory is available in the Github Repository with intermediate data files and results.

- `consolidated_data/`: Folder with files that consolidate the preprocessed data
- `consolidated_results/`: Folder for storing the obtained results
- `sa_publishers_category/`: Folder with the summary files that contain the average sentiment score per day for every publisher category
- `sa_top_topics/`: Folder with the summary files that contain the average sentiment score per day for each topic (From the top 30 most correlated topics).

### Large_Data - Directory
It is recommended to define a local directory for this directory as it will store the inital dataset and large preprocessed data files. The inital dataset is around 8 GB in size, and some intermediate files are greater than 1GB. Expect this directory to be above 15GB if you run all the codes. This directory and the files are not uploaded in this Github Repository.

- `all-the-news-2-1/`: Folder downloaded from "All the News 2.0" containing the file all-the-news-2-1.csv
- `filtered_news_parquets/`: Folder with the processed parquet files organized by Publisher category and by year.
- `sentiment_analysis_EBF/`: Folder with the processed parquet files of the EBF news with the calculated sentiment score.
- `sentiment_analysis_full_article_EBF/`: Folder with the processed parquet files of the EBF news with the full article and with the calculated sentiment score. 
- `BERTopics/`: folder with the trained topic models.
  
### Formatted_Results - Directory
Folder with excel files that display the results in a format that is easier to understand and interpret. It is available in the Github Repository with intermediate data files and results.
