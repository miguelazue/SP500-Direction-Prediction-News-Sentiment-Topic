
"""
Created on July 7 8:59:23 2023

@author: migue
"""

import pandas as pd
import numpy as np

import statsmodels.api as sm

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.naive_bayes import GaussianNB


# Pylance doesnt identify the library but it is correctly working
import xgboost as xgb

import random

import matplotlib.pyplot as plt

import xlsxwriter


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
import A_ThesisFunctions as tf

#-------------------------
#------Parameters---------
#-------------------------



# Current Parameter Options

# BERT Model Year: 2016
# BERT Model Headlines Count Options: 250, 20
# Model Specs: FinBERT, FinBERT_BERTopic, LM_LDATopic,LM
# Train Period: 2016, 2017, 2018, 2017-2019
# Test Period: 2016, 2017, 2018, 2017-2019
# Variable Specs = Current Variables, Lag Variables



# topic_model_year = "2016"
# topic_model_articles_count = "250"
# article_text_type = "Headlines"
# model_specs = "FinBERT_BERTopic"
# train_period = "2016"
# test_period = "2017-2019"
# variable_specs = "Current Variables"
# classification_model_name = "log_reg"


def evaluate_prediction_models(topic_model_year = "2016",
                              topic_model_articles_count = "250",
                              article_text_type = "Headlines",
                              model_specs = "FinBERT_BERTopic",
                              train_period = "2016",
                              test_period = "2017-2019",
                              variable_specs = "Lag Variables",
                              classification_model_name = "log_reg",
                              parameter_1 = "",
                              parameter_2 = "",
                              parameter_3 = ""                    
                              ):
    """Function that evaluates models with the classification scores of the given parameters"""

    """
    Parameters
    ---------
    topic_model_year: year of training of the topic model
    topic_model_articles_count: sample size of the topic model
    article_text_type = "Headline" (Headline or Bodytext)
    model_specs: Indicate the models to be considered. (FinBERT_BERTopic, FinBERT,LM_LDATopic,LM)
    train_period: training period of the prediction model
    test_period: test period of the prediction model
    variable_specs: Indicate the type of variables (Current Variables, Lag Variables, ...)
    classification_model_name: Indicate the type of classification model ("log_reg","svc_model","dtree","rf_model","nb_model","gbc_model")
    ---------
    """
    if (model_specs == "FinBERT") or (model_specs == "coin_flip") or (model_specs == "zero_rate"):
        consolidated_file_name = "FinBERT" + "_BERTopicModel"
    elif (model_specs == "LM"):
        consolidated_file_name = model_specs + "_LDATopicModel"
    else:
        consolidated_file_name = model_specs + "Model"


    consolidated_model_name = consolidated_file_name+topic_model_year+article_text_type+topic_model_articles_count+"k"


    #-------------------------
    #-Load Consolidated Data--
    #-------------------------

    consolidated_data_path
    load_data_name = "consolidated_data_" + consolidated_model_name + ".csv"
    # data_name = "consolidated_data_FinBERT_BERTopicModel2016Sample20k.csv"
    # load_data_name

    # Load Consolidated data
    consolidated_data = pd.read_csv(consolidated_data_path+load_data_name,index_col=0)

    # Date Format
    consolidated_data["date"] = pd.to_datetime(consolidated_data["date"])

    if (model_specs == "FinBERT") or (model_specs == "LM"):
        consolidated_model_name = model_specs + "_"+article_text_type
        EBF_columns = ["date",'EBF', 'time_trend', 'adjclose','returns','year','month']
        consolidated_data = consolidated_data[EBF_columns].copy()

    if (model_specs == "coin_flip") or (model_specs == "zero_rate"):
        consolidated_model_name = model_specs
        article_text_type = model_specs
        classification_model_name = model_specs
        EBF_columns = ["date",'EBF', 'time_trend', 'adjclose','returns','year','month']
        consolidated_data = consolidated_data[EBF_columns].copy()

    #-------------------------
    #--Calculated Variables---
    #-------------------------

    # Direcction Variable
    consolidated_data.loc[consolidated_data["returns"]>=0, "direction"] = 1
    consolidated_data.loc[consolidated_data["returns"]< 0, "direction"] = 0

    n_variables = len(consolidated_data.columns)

    # Shift Variables
    variables_to_lag = consolidated_data.columns.to_list()
    variables_to_lag.remove('date')
    variables_to_lag.remove('time_trend')
    variables_to_lag.remove('year')
    variables_to_lag.remove('month')
    variables_to_lag

    for variable_i in variables_to_lag:
        consolidated_data[variable_i+'_lag'] = consolidated_data[variable_i].shift(1)


    #-------------------------
    #--variables Selection----
    #-------------------------

    # Current Variables List 
    current_variables = consolidated_data.columns.to_list()[:(n_variables-1)]
    #current_variables = consolidated_data.columns.to_list()[:37]
    current_variables.remove('date')
    current_variables.remove('adjclose')
    current_variables.remove('returns')
    current_variables.remove('year')
    current_variables.remove('month')
    # current_variables.remove('topic_-1')
    # current_variables.remove('time_trend')
    # current_variables




    # Lag Variables List

    lag_variables = consolidated_data.columns.to_list()[(n_variables-5-1):]
    lag_variables.remove('adjclose')
    lag_variables.remove('returns')
    lag_variables.remove('year')
    lag_variables.remove('month')
    lag_variables.remove('direction')
    lag_variables.remove('adjclose_lag')
    lag_variables.remove('returns_lag')
    # lag_variables.remove('topic_-1_lag')
    # lag_variables.remove('time_trend')
    # lag_variables

    #-------------------------
    #--Dataset Organization---
    #-------------------------

    consolidated_data_2016 = consolidated_data[consolidated_data['year']==2016]
    consolidated_data_2016 = consolidated_data_2016.dropna()
    # consolidated_data_2016['direction'].value_counts()
    # consolidated_data_2016['direction'].describe()

    consolidated_data_2017 = consolidated_data[consolidated_data['year']==2017]
    consolidated_data_2017 = consolidated_data_2017.dropna()

    consolidated_data_2018 = consolidated_data[consolidated_data['year']==2018]
    consolidated_data_2018 = consolidated_data_2018.dropna()

    consolidated_data_2019 = consolidated_data[consolidated_data['year']==2019]
    consolidated_data_2019 = consolidated_data_2019.dropna()

    consolidated_data_after_2016 = consolidated_data[consolidated_data['year']>=2017]
    consolidated_data_after_2016 = consolidated_data_after_2016.dropna()

    consolidated_data_after_2017 = consolidated_data[consolidated_data['year']>=2018]
    consolidated_data_after_2017 = consolidated_data_after_2017.dropna()


    datasets_dict = {"2016":consolidated_data_2016,
                    "2017":consolidated_data_2017,
                    "2018":consolidated_data_2018,
                    "2019":consolidated_data_2019,
                    "2017-2019":consolidated_data_after_2016,
                    "2018-2019":consolidated_data_after_2017
                    }


    #-------------------------
    #--Parameter Adjustment---
    #-------------------------

    train_data = datasets_dict[train_period].copy()
    test_data = datasets_dict[test_period].copy()




    variables_specs_dict = {"Current Variables":current_variables,
                            "Lag Variables":lag_variables
                            }


    model_variables = variables_specs_dict[variable_specs]

    #-------------------------
    #--RESULTING DATAFRAME----
    #-------------------------

    results_df = pd.DataFrame({
                    'model': pd.Series(dtype='string'),
                    'model_specs': pd.Series(dtype='string'),
                    'train_period': pd.Series(dtype='string'),
                    'test_period': pd.Series(dtype='string'),
                    'variables_specs': pd.Series(dtype='string'),
                    'classification_model': pd.Series(dtype='string'),
                    'parameter_1': pd.Series(dtype='string'),
                    'parameter_2': pd.Series(dtype='string'),
                    'parameter_3': pd.Series(dtype='string'),
                    'accuracy': pd.Series(dtype='float'),
                    'sensitivity': pd.Series(dtype='float'),
                    'specificity': pd.Series(dtype='float'),
                    'precision': pd.Series(dtype='float'),
                    'F1_score': pd.Series(dtype='float'),
                    'MCC': pd.Series(dtype='float'),
                    })

    # results_df.dtypes

    #-------------------------
    #------Dataset Split------
    #-------------------------

    Xtrain = train_data[model_variables]
    Ytrain = train_data['direction']

    Xtest = test_data[model_variables]
    Ytest = test_data['direction']

    #-------------------------
    #Fitting Predicition Models
    #-------------------------
    
    #If parameters for classification models are not defined
    # Assignation of the best parameters for each model

    if parameter_1 == "":
        if classification_model_name == "svc_model":
            parameter_1 = 10
            parameter_2 = "linear"
        elif classification_model_name == "dtree":
            parameter_1 = "entropy"
            parameter_2 = 5
        elif classification_model_name == "rf_model":
            parameter_1 = "gini"
            parameter_2 = 10
            parameter_3 = 1000
        elif classification_model_name == "nb_model":
            parameter_1 = 1e-9
        elif classification_model_name == "gbc_model":
            parameter_1 = "gbtree"
            parameter_2 = 0.1
        else:
            parameter_1 == ""

    if classification_model_name == "svc_model":
        # SVM Classifier Model
        classification_model = svm.SVC(C=parameter_1,kernel=parameter_2)
        classification_model.fit(Xtrain, Ytrain)
    elif classification_model_name == "dtree":
        # Decision Tree Model
        classification_model = DecisionTreeClassifier(random_state=0,criterion=parameter_1,max_depth=parameter_2)
        classification_model.fit(Xtrain, Ytrain)
    elif classification_model_name == "rf_model":
        # RF Classifier Model
        classification_model = RandomForestClassifier(random_state=0,criterion=parameter_1,max_depth=parameter_2,n_estimators=parameter_3)
        classification_model.fit(Xtrain, Ytrain)
    elif classification_model_name == "nb_model":
        # Naive Bayes Classifier Model
        classification_model = GaussianNB(var_smoothing = parameter_1)
        classification_model.fit(Xtrain, Ytrain)
    elif classification_model_name == "gbc_model":
        # Gradient Boosting Classifier
        classification_model = xgb.XGBClassifier(booster = parameter_1,eta=parameter_2)
        classification_model.fit(Xtrain, Ytrain)
    else:
        classification_model = sm.Logit(Ytrain, Xtrain).fit()


    #Converting parameter values into text
    parameter_1 = str(parameter_1)
    parameter_2 = str(parameter_2)
    parameter_3 = str(parameter_3)
    #-------------------------
    #--Evaluating the Model--
    #-------------------------

    if len(model_variables) > 0:
        if model_specs == "coin_flip":
            Yhat = [random.randint(0, 1) for flip in range(len(Xtest)) ]
        elif model_specs == "zero_rate":
            Yhat = [1]*len(Xtest)
        else:
            Yhat = classification_model.predict(Xtest)
        prediction = list(map(round, Yhat))
        tp,fp,fn,tn = tf.my_confusion_matrix(Ytest,prediction)
        classification_scores = tf.my_classification_scores(tp,fp,fn,tn)
        if ((tp == 0) or (tn==0)) and ((fp==0) or (fn==0)):
            mcc_calculation = 0
        else:
            mcc_calculation = (tp*tn-fn*fp)/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1/2))
    else:
        classification_scores = [0,0,0,0]
        mcc_calculation = 0

    accuracy = classification_scores[0]
    sensitivity = classification_scores[1]
    specificity = classification_scores[2]
    precision = classification_scores[3]
    mcc = mcc_calculation

    #calculate f1 score
    if (precision == 0) and (sensitivity == 0):
        f1_score = 0
    else:
        f1_score = 2*((precision*sensitivity)/(precision+sensitivity))

    current_row = [consolidated_model_name,model_specs,train_period,test_period,variable_specs,classification_model_name,parameter_1,parameter_2,parameter_3,accuracy,sensitivity,specificity,precision,f1_score,mcc]
    results_df.loc[len(results_df)] = current_row

    return(results_df)


#-------------------------
#---TESTING THE FUNCTION--
#-------------------------

evaluate_prediction_models()

evaluate_prediction_models(topic_model_articles_count="20")
evaluate_prediction_models(topic_model_articles_count="250")
evaluate_prediction_models(variable_specs = "Lag Variables")
evaluate_prediction_models(model_specs ="FinBERT")
evaluate_prediction_models(model_specs = "FinBERT",variable_specs = "Lag Variables")
evaluate_prediction_models(model_specs ="FinBERT",article_text_type="Bodytext")
evaluate_prediction_models(model_specs ="FinBERT",article_text_type="Headlines")
evaluate_prediction_models(model_specs ="LM",article_text_type="Headlines")
evaluate_prediction_models(classification_model_name = "log_reg")
evaluate_prediction_models(classification_model_name = "svc_model")
evaluate_prediction_models(classification_model_name = "dtree")
evaluate_prediction_models(classification_model_name = "rf_model")
evaluate_prediction_models(classification_model_name = "nb_model")
evaluate_prediction_models(classification_model_name = "gbc_model")


evaluate_prediction_models(model_specs = "coin_flip")
evaluate_prediction_models(model_specs = "zero_rate",test_period="2018-2019")
evaluate_prediction_models(model_specs = "zero_rate",test_period="2017-2019")

#Loughran McDonald Dictionary Sentiment Analysis
evaluate_prediction_models(model_specs ="LM")
evaluate_prediction_models(model_specs ="LM_LDATopic")




#-------------------------
#----ALL COMBINATIONS-----
#-------------------------


# sample 250
topic_model_year_LIST = ["2016"]
topic_model_articles_count_LIST = ["250","20"]
#article_text_type_LIST = ["Headlines"]
article_text_type_LIST = ["Headlines","Bodytext"]
model_specs_LIST = ["FinBERT_BERTopic","FinBERT","LM_LDATopic","LM"]
train_period_LIST = ["2016"]
test_period_LIST = ["2017","2018","2019","2017-2019","2018-2019"]
variable_specs_LIST = ["Lag Variables"]
#variable_specs_LIST = ["Current Variables","Lag Variables"]
models_name_list = ["log_reg","svc_model","dtree","rf_model","nb_model","gbc_model"]

consolidated_results_df = pd.DataFrame()
iteration = 1

for topic_model_year_i in topic_model_year_LIST:
    for topic_model_articles_count_i in topic_model_articles_count_LIST:
        for article_text_type_i in article_text_type_LIST:
            for model_specs_i in model_specs_LIST:
                # Avoid cases
                # Avoid duplicating numbers when only considering FinBERT
                # Avoid LM and LM_LDATopic with sample size of 20
                if ((model_specs_i == "FinBERT") or (model_specs_i == "LM") or (model_specs_i == "LM_LDATopic")) and topic_model_articles_count_i != "250":
                    break

                #Consider only possible cases with Bodytext information
                if ((model_specs_i == "LM") or (model_specs_i == "LM_LDATopic") or (topic_model_articles_count_i == "20")) and article_text_type_i =="Bodytext":
                    break
                for train_period_i in train_period_LIST:
                    for test_period_i in test_period_LIST:
                        for variable_specs_i in variable_specs_LIST:
                            for model_name_i in models_name_list:
                                results_i = evaluate_prediction_models(topic_model_year = topic_model_year_i,
                                                                    topic_model_articles_count = topic_model_articles_count_i,
                                                                    article_text_type = article_text_type_i,
                                                                    model_specs = model_specs_i,
                                                                    train_period = train_period_i,
                                                                    test_period = test_period_i,
                                                                    variable_specs = variable_specs_i,
                                                                    classification_model_name = model_name_i)
                                if iteration == 1:
                                    consolidated_results_df = results_i
                                else:
                                    consolidated_results_df = pd.concat([consolidated_results_df,results_i],ignore_index=True)
                                iteration += 1





# Visualizing Results
consolidated_results_df.test_period.value_counts()

consolidated_results_df.model.value_counts()

# Save Path

#save_path 
consolidated_results_path
consolidated_results_df.to_excel(consolidated_results_path+"ConsolidatedResults4.xlsx",engine='xlsxwriter')  



#-------------------------
#--------ZERO RATE--------
#-------------------------

zero_rate_results_df = pd.DataFrame()
iteration = 1
for test_period_i in test_period_LIST:
    results_i = evaluate_prediction_models(model_specs = "zero_rate",test_period = test_period_i)
    if iteration == 1:
        zero_rate_results_df = results_i
    else:
        zero_rate_results_df = pd.concat([zero_rate_results_df,results_i],ignore_index=True)
    iteration += 1




#save_path 
consolidated_results_path
zero_rate_results_df.to_excel(consolidated_results_path+"ZeroRateResults.xlsx",engine='xlsxwriter')  

#-------------------------
#--COIN FLIP EXPERIMENTS--
#-------------------------


coin_flip_results_df = pd.DataFrame()
iteration = 1
for test_period_i in test_period_LIST:
    results_i = evaluate_prediction_models(model_specs = "coin_flip",test_period = test_period_i)
    if iteration == 1:
        coin_flip_results_df = results_i
    else:
        coin_flip_results_df = pd.concat([coin_flip_results_df,results_i],ignore_index=True)
    iteration += 1

#save_path 
consolidated_results_path
coin_flip_results_df.to_excel(consolidated_results_path+"CoinFlipExperimentResults.xlsx",engine='xlsxwriter') 







#-------------------------
#-------Parameters--------
#------GRID SEARCH-------
#-------------------------



# sample 250
topic_model_year_i = "2016"
topic_model_articles_count_i = "250"
article_text_type_i = "Headlines"
train_period_i = "2016"
#Validation period for grid search should be different from the training and from the test period
#test_period_i = "2017-2019"
test_period_i = "2017"
model_specs_LIST = ["FinBERT_BERTopic","LM_LDATopic"]
variable_specs_LIST = ["Lag Variables"]

#models_name_list = ["svc_model","dtree"]
models_name_list = ["svc_model","dtree","rf_model","nb_model","gbc_model"]



fine_tuning_df = pd.DataFrame()
iteration = 1

for model_specs_i in model_specs_LIST:
    for variable_specs_i in variable_specs_LIST:
        for model_name_i in models_name_list:
            if model_name_i == "svc_model":
                parameter_1_LIST = [0.1,1, 10]
                parameter_2_LIST = ["linear", "poly", "rbf", "sigmoid"]
                parameter_3_LIST = [""]
                
            elif model_name_i == "dtree":
                parameter_1_LIST = ["gini", "entropy"]
                parameter_2_LIST = [None, 3,5,10]
                parameter_3_LIST = [""]

            elif model_name_i == "rf_model":
                parameter_1_LIST = ["gini", "entropy"]
                parameter_2_LIST = [None, 3,5,10]
                parameter_3_LIST = [500,1000,1500]
            
            elif model_name_i == "nb_model":
                parameter_1_LIST = [1e-5,1e-9,1e-15]
                parameter_2_LIST = [""]
                parameter_3_LIST = [""]
            
            elif model_name_i == "gbc_model":
                parameter_1_LIST = ["gbtree"]
                parameter_2_LIST = [0.1, 0.3 , 0.5, 0.8]

            else:
                parameter_1_LIST = [""]
                parameter_2_LIST = [""]
                parameter_3_LIST = [""]

            for parameter_1_i in parameter_1_LIST:
                for parameter_2_i in parameter_2_LIST:
                    for parameter_3_i in parameter_3_LIST:
                        results_i = evaluate_prediction_models(topic_model_year = topic_model_year_i,
                                                            topic_model_articles_count = topic_model_articles_count_i,
                                                            article_text_type = article_text_type_i,
                                                            model_specs = model_specs_i,
                                                            train_period = train_period_i,
                                                            test_period = test_period_i,
                                                            variable_specs = variable_specs_i,
                                                            classification_model_name = model_name_i,
                                                            parameter_1= parameter_1_i,
                                                            parameter_2= parameter_2_i,
                                                            parameter_3= parameter_3_i)
                        if iteration == 1:
                            fine_tuning_df = results_i
                        else:
                            fine_tuning_df = pd.concat([fine_tuning_df,results_i],ignore_index=True)
                        iteration += 1
                        print(iteration)

#save_path 
consolidated_results_path
fine_tuning_df.to_excel(consolidated_results_path+"fine_tuning_df4.xlsx",engine='xlsxwriter')  
