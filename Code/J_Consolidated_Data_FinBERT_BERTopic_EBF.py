
import pandas as pd
import numpy as np


#-------------------------
#---Load Financial Data---
#-------------------------

gspc_path = "C:/Users/migue/Dropbox/JKU EBA Masters/Master Thesis/Data/"

gspc_data = pd.read_csv(gspc_path+"GSPC.csv")


selected_columns = ["formatted_date","time_trend","adjclose","returns"]
gspc_df = gspc_data[selected_columns].copy()

gspc_df["formatted_date"] = pd.to_datetime(gspc_df["formatted_date"])

gspc_df['year'] = gspc_df['formatted_date'].dt.year
gspc_df['month'] = gspc_df['formatted_date'].dt.month

#gspc_sample = gspc_df[(gspc_df["year"]==2017)&(gspc_df["month"]==10)]




gspc_prices = gspc_df.copy()

#renaming formatted date column to date
gspc_prices.rename(columns = {'formatted_date':'date'}, inplace = True)

#gspc_sample.head()
#gspc_sample.tail()




#-------------------------
#-Load SA EBF TOPICS Data-
#-------------------------


# sa_top_BERTopicModel2016Sample20k_on_2019news


#LOAD
topics_sa_path = "C:/Users/migue/Dropbox/JKU EBA Masters/Master Thesis/Data/sa_top_topics/"

# Year Model
model_year = "2016"
model_articles_count = "250"
article_text_type = "Bodytext"
#article_text_type = "Headlines"

#Iterate over files
years = ["2016","2017","2018","2019"]



# For to concatenate the summary files
first_iteration = True
for news_year in years:
    file_name = "sa_top_topics_BERTopicModel" +model_year + article_text_type +model_articles_count+ "k_news" + news_year + ".csv"
    current_df = pd.read_csv(topics_sa_path+file_name,index_col=0)

    if first_iteration == True:
        first_iteration = False
        sa_topics = current_df.copy()
    else:
        sa_topics = pd.concat([sa_topics, current_df], ignore_index=True, sort=False)




sa_topics['date'] = pd.to_datetime(sa_topics['date'], format='%Y-%m-%d')

sa_topics = sa_topics.sort_values(by=['date'])
sa_topics = sa_topics.reset_index(drop=True)

# fill na's with 0
sa_topics = sa_topics.fillna(value=0)
#sa_topics.head()
#sa_topics.tail()


#-------------------------
#-Load SA GENERAL EBF ----
#-------------------------

#LOAD
categories_data_path = "C:/Users/migue/Dropbox/JKU EBA Masters/Master Thesis/Data/sa_publishers_category/"

#Iterate over periods
#years = [2017,2018,2019]
years = [2016,2017,2018,2019]
months = [1,2,3,4,5,6,7,8,9,10,11,12]
end_yearmonth = "202001"

#end_yearmonth = "202004"

sa_ebf = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'),
                   'EBF': pd.Series(dtype='float64'),
                   'GEN': pd.Series(dtype='float64'),
                   'POL': pd.Series(dtype='float64'),
                   'ENT': pd.Series(dtype='float64')
                   })



break_flag = False


#NEW FOR
for year in years:
    for month in months:
        yearmonth = str(year*100+month)


        #break month loop
        if yearmonth == end_yearmonth:
            break_flag = True
            break


        current_df = pd.read_csv(categories_data_path+"sa_publishers_category_"+yearmonth+".csv",index_col=0)
        sa_ebf = pd.concat([sa_ebf, current_df], ignore_index=True, sort=False)
    #break year loop
    if break_flag == True:
        break

sa_ebf['date'] = pd.to_datetime(sa_ebf['date'], format='%Y-%m-%d')

sa_ebf = sa_ebf.sort_values(by=['date'])
sa_ebf = sa_ebf.reset_index(drop=True)


sa_ebf_columns = ['date','EBF']

sa_ebf = sa_ebf[sa_ebf_columns]
sa_ebf.head()
sa_ebf.tail()


#-------------------------
#--------Joining Data-----
#-------------------------


sa_topics_ebf = pd.merge(sa_topics,sa_ebf, left_on='date', right_on='date', how='inner')


sa_and_prices = pd.merge(sa_topics_ebf,gspc_prices, left_on='date', right_on='date', how='inner')

#sa_and_prices.head()
#sa_and_prices.tail()
#sa_and_prices.describe()
#len(sa_and_prices.columns)

save_data_path = "C:/Users/migue/Dropbox/JKU EBA Masters/Master Thesis/Data/consolidated_data/"

save_data_name = "consolidated_data_FinBERT_BERTopicModel"+model_year+article_text_type+model_articles_count+"k.csv"

# sa_and_prices
sa_and_prices.to_csv(save_data_path+save_data_name)





