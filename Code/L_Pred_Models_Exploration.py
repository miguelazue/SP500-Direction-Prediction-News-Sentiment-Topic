
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

import xgboost as xgb

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import svm

import matplotlib.pyplot as plt

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
topics_path = os.path.join(heavy_data_path,'BERTopics/')


#-------------------------
#-Load Utility Script-----
#-------------------------

os.chdir(current_dir)
import A_ThesisFunctions as tf

#-------------------------
#-Load Consolidated Data--
#-------------------------

# Year Model
model_year = "2016"
model_headlines_count = "250"


load_data_name = "consolidated_data_FinBERT_BERTopicModel"+model_year+"Headlines"+model_headlines_count+"k.csv"
#data_name = "consolidated_data_BERTopicModel2016Sample20k.csv"



consolidated_data = pd.read_csv(consolidated_data_path+load_data_name,index_col=0)

consolidated_data["date"] = pd.to_datetime(consolidated_data["date"])

#-------------------------
#--Calculated Variables---
#-------------------------

# Direcction Variable
consolidated_data.loc[consolidated_data["returns"]>=0, "direction"] = 1
consolidated_data.loc[consolidated_data["returns"]< 0, "direction"] = 0


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
#-------Correlations------
#-------------------------

consolidated_data = consolidated_data.copy()

consolidated_data.tail()



# Reviewing classification balance or imbalance

consolidated_data.direction.value_counts()

consolidated_data.year
year_direction_balance = consolidated_data.groupby(['year','direction'])['date'].agg('count').reset_index()


#year_direction_balance.to_excel(save_path+"YearDirectionBalance.xlsx",engine='xlsxwriter')  





# consolidated_data.columns

# # Selection of columns and order
corr_matrix = consolidated_data.corr()

# # Correlation with direction
# corr_matrix['direction'][:37]

# # Correlation of the lag variables with the Direction
# corr_matrix['direction'][31:]

# # Correlation with returns
corr_matrix['returns'][:37]

# # Correlation of the lag variables with the returns
corr_matrix['returns'][31:]




#-------------------------
#--variables Selection----
#-------------------------

# Filtering out Lag variables
current_variables = consolidated_data.columns.to_list()[:37]
current_variables.remove('date')
current_variables.remove('adjclose')
current_variables.remove('returns')
current_variables.remove('year')
current_variables.remove('month')
#current_variables.remove('topic_-1')
#current_variables.remove('time_trend')
current_variables


# Lag variables

lag_variables = consolidated_data.columns.to_list()[32:]
lag_variables.remove('adjclose')
lag_variables.remove('returns')
lag_variables.remove('year')
lag_variables.remove('month')
lag_variables.remove('direction')
lag_variables.remove('adjclose_lag')
lag_variables.remove('returns_lag')
#lag_variables.remove('topic_-1_lag')
#lag_variables.remove('time_trend')
lag_variables



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



#-------------------------
#------Parameters---------
#-------------------------

train_data = consolidated_data_2016.copy()
test_data = consolidated_data_after_2016.copy()
#model_variables = current_variables
model_variables = lag_variables

#-------------------------
#------Dataset Split------
#-------------------------

Xtrain = train_data[model_variables]
Ytrain = train_data['direction']

Xtest = test_data[model_variables]
Ytest = test_data['direction']





#-------------------------
#--Logistic Regression----
#-------------------------

# building the model and fitting the data
log_reg = sm.Logit(Ytrain, Xtrain).fit()
log_reg.summary()

# performing predictions on the test dataset
Yhat = log_reg.predict(Xtest)
prediction = list(map(round, Yhat))


#-------------------------
#--Decision Tree Model----
#-------------------------

dtree = DecisionTreeClassifier()
dtree = dtree.fit(Xtrain, Ytrain)
# tree.plot_tree(dtree, feature_names=model_variables)
# plt.savefig(heavy_data_path+'dtree500_t.png',dpi=500)

Yhat = dtree.predict(Xtest)
prediction = list(map(round, Yhat))

#-------------------------
#-Support Vector Machine--
#-------------------------
svc_model = svm.SVC()

svc_model.fit(Xtrain, Ytrain)

# performing predictions on the test dataset
Yhat = svc_model.predict(Xtest)
prediction = list(map(round, Yhat))




#------------------------------
#-SVM adjusting Class Weights--
#------------------------------
train_data['direction'].value_counts()


balance_0 = year_direction_balance[(year_direction_balance["year"]==2016)&(year_direction_balance["direction"]==0)]["date"].sum()
balance_1 = year_direction_balance[(year_direction_balance["year"]==2016)&(year_direction_balance["direction"]==1)]["date"].sum()

weight0 = balance_0/(balance_0 + balance_1)
weight1 = balance_1/(balance_0 + balance_1)
weight0
weight1

cw = {0:weight0,1:weight1}
cw = {0:1,1:2}
cw = {0:2,1:1}
cw = "balanced"

svc_model = svm.SVC(class_weight=cw)

svc_model.fit(Xtrain, Ytrain)

# performing predictions on the test dataset
Yhat = svc_model.predict(Xtest)
prediction = list(map(round, Yhat))


#-------------------------
#-------Random Forest-----
#-------------------------

rf_model = RandomForestClassifier(max_depth=3, random_state=0)
rf_model.fit(Xtrain, Ytrain)

# performing predictions on the test dataset
Yhat = rf_model.predict(Xtest)
prediction = list(map(round, Yhat))


#------------------------------
#-------------KNN--------------
#------------------------------
from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors=17)  

knn.fit(Xtrain, Ytrain)

# performing predictions on the test dataset
Yhat = knn.predict(Xtest)
prediction = list(map(round, Yhat))


#------------------------------
#-Naive Bayes Classifier-------
#------------------------------

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(Xtrain, Ytrain)

# performing predictions on the test dataset
Yhat = nb.predict(Xtest)
prediction = list(map(round, Yhat))


#------------------------------
#-Gradient Boosting Classifier-
#------------------------------

# Create a Gradient Boosting classifier object
gbm = xgb.XGBClassifier()

gbm.fit(Xtrain, Ytrain)

# performing predictions on the test dataset
Yhat = gbm.predict(Xtest)
prediction = list(map(round, Yhat))




#------------------------------
#--------GBC Sklearn-----------
#------------------------------
gbc = GradientBoostingClassifier()
gbc.fit(Xtrain, Ytrain)

# performing predictions on the test dataset
Yhat = gbc.predict(Xtest)
prediction = list(map(round, Yhat))



#-------------------------
#--Zero Rate Classifier---
#-------------------------

len(Xtest)
Yhat = [1]*len(Xtest)
prediction = list(map(round, Yhat))

#-------------------------
#---------Coin Flip-------
#-------------------------

import random

random.seed(2)

Yhat = [random.randint(0, 1) for flip in range(len(Xtest)) ]
prediction = list(map(round, Yhat))
len(Yhat)





#-------------------------
#--Evaluating the Model---
#-------------------------

tp,fp,fn,tn = tf.my_confusion_matrix(Ytest,prediction)
tf.my_classification_scores(tp,fp,fn,tn)


#-------------------------
#--RESULTING DATAFRAME----
#-------------------------


results_df = pd.DataFrame({
                'id': pd.Series(dtype='int'),
                'model': pd.Series(dtype='string'),
                'model_specs': pd.Series(dtype='string'),
                'train_period': pd.Series(dtype='string'),
                'test_period': pd.Series(dtype='string'),
                'variables_specs': pd.Series(dtype='string'),
                'classification_model': pd.Series(dtype='string'),
                'accuracy': pd.Series(dtype='float'),
                'sensitivity': pd.Series(dtype='float'),
                'specificity': pd.Series(dtype='float'),
                'precision': pd.Series(dtype='float')
                })

results_df.dtypes


results_df

# Ideas of models to compare
#https://ieeexplore.ieee.org/document/8759119


