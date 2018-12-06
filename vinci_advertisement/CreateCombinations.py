#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
pd.set_option('display.max_row',100)
pd.set_option('display.max_columns',400)
pd.set_option('display.float_format','{:20,.2f}'.format)
pd.set_option('display.max_colwidth',600)
import json
import os
import gc
import time
import sys
import io
import re
import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn import ensemble
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import multiprocessing as mp
from multiprocessing import Pool,Manager

# In[1]:
class ParallelRunner():

    def __init__(self):
        self.data = None
        self.bay = BayesianRidge(compute_score=True)
        self.svr = SVR(gamma='auto', C=1.0, epsilon=0.05,kernel ='rbf')

    def getNextTenDays(self):
        dates = []
        today = datetime.datetime.now()
        for n in range(1, 2):
            dates.append(today + datetime.timedelta(days=n))
        return dates

    # def action(self,roas,keys,lock,count,agemax,agemin,country,interest,islike,group_range,group_source):
    #     print("action ....")
    #     if islike == 1:
    #         _roas = self.get_mean_roas(agemax,agemin,country,interest,islike,group_range,group_source)
    #     else:
    #         _roas = self.get_mean_roas(agemax,agemin,country,interest)
    #     print("roas  = ",_roas)
    #     with lock:
    #         if _roas > roas.value:
    #             print("roas  = ",_roas)
    #             roas.value = _roas
    #             if islike == 1:
    #                 self.set_keys(keys,agemax,agemin,country,interest,islike,group_range,group_source)
    #             else:
    #                 self.set_keys(keys,agemax,agemin,country,interest)
    #     return _roas

    def action(self,roas,keys,lock,count,combination):
        # if combination[4] == 1:
        #     _roas = self.get_mean_roas(combination[0],combination[1],combination[2],combination[3],combination[4],combination[5],combination[6])
        # else:
        #     _roas = self.get_mean_roas(combination[0],combination[1],combination[2],combination[3])
        print("actions",combination)

        # with lock:
        #     count.value = count.value + 1
        #     print("count ",count.value)
        #     if _roas > roas.value:
        #         print("roas  = ",_roas)
        #         roas.value = _roas
        #         if combination[4] == 1:
        #             self.set_keys(keys,combination[0],combination[1],combination[2],combination[3],combination[4],combination[5],combination[6])
        #         else:
        #             self.set_keys(keys,combination[0],combination[1],combination[2],combination[3])
        return 1

    def get_mean_roas(self,agemax,agemin,country,interest,islike = 0,group_range = 0,group_source = 0):
        next_dates = self.getNextTenDays()
        result = 0
        date = []
        pre_data = []
        start = time.time()
        for date in next_dates:
            pre_data.append([agemax,agemin,country,0,interest,date.weekday(),date.day,date.month,islike,group_range,group_source])
        # result  =  (np.mean(self.bay.predict(pre_data)) + np.mean(self.svr.predict(pre_data))) / 2
        result = self.bay.predict(pre_data)
        end = time.time()
        del pre_data
        del next_dates
        del agemax
        del agemin
        del country
        del interest
        del islike
        del group_range
        del group_source
        print("get_mean_roas execution time ",end - start)
        return result

    def set_keys(self,keys,agemax,agemin,country,interest,islike = 0,group_range = 0,group_source = 0):
        keys['Age Max'] = agemax
        keys['Age Min'] = agemin
        keys['Countries'] = country
        keys['Gender'] = 0
        keys['Interest'] = interest
        keys['Group.Range'] = group_range
        keys['Group.Source'] = group_source
        keys['IsLookalike'] = islike

    # def test(self,x):
    #     print(x*x)
    #     return x*x
    #
    # def findyou(self):
    #
    #     def call_back(f):
    #         gc.collect()
    #
    #     def apply_interest(x):
    #         if pd.notna(x):
    #             keys = x.keys()
    #             for key_num,key in enumerate(keys):
    #                 if key_num != 0:
    #                     result = "|" + key + ":" + x[key][0]['name']
    #                 else:
    #                     result = key + ":" + x[key][0]['name']
    #                 for i in range(1,len(x[key])):
    #                     result = result + '_' + x[key][i]['name']
    #         else:
    #             return ''
    #         return result
    #
    #     # process data start ===================================================================================================================================================================
    #     used_rules_cols = ['Campaign Name','Ad Set ID','Ad Set Name','Age Max','Age Min','Countries','Custom Audiences','Gender','Flexible Inclusions','Product 1 - Link','Publisher Platforms','Facebook Positions','Instagram Positions','Device Platforms','Title','Body']
    #
    #     # new1
    #     # rules
    #     df_indiegogo_rules_new1 = pd.read_csv('./data/indiegogo/new1_setting_utf8.csv',sep = '\t',dtype = {'Ad Set ID':'str','Custom Audiences':'str'},encoding = 'utf-8')
    #     df_indiegogo_rules_new1 = df_indiegogo_rules_new1[df_indiegogo_rules_new1["Campaign Name"].str.contains('indie',case = False)]
    #     df_indiegogo_rules_new1 = df_indiegogo_rules_new1[used_rules_cols]
    #     # data
    #     df_indiegogo_data_new1 = pd.read_csv('./data/indiegogo/new1_data_utf8.csv',sep = ',',dtype = {'Ad Set ID':'str'},encoding = 'utf-8')
    #     df_indiegogo_rules_new1['Ad Set ID'] = df_indiegogo_rules_new1['Ad Set ID'].apply(lambda x: x.split(':')[1])
    #     # summary
    #     data_new1 = df_indiegogo_rules_new1.merge(df_indiegogo_data_new1,how = 'left',on = 'Ad Set ID')
    #     #==============================================================================================================
    #
    #     # new2
    #     # rules
    #     df_indiegogo_rules_new2 = pd.read_csv('./data/indiegogo/new2_setting_utf8.csv',sep = '\t',dtype = {'Ad Set ID':'str','Custom Audiences':'str'},encoding = 'utf-8')
    #     df_indiegogo_rules_new2 = df_indiegogo_rules_new2[df_indiegogo_rules_new2["Campaign Name"].str.contains('indie',case = False)]
    #     df_indiegogo_rules_new2 = df_indiegogo_rules_new2[used_rules_cols]
    #     # data
    #     df_indiegogo_data_new2 = pd.read_csv('./data/indiegogo/new2_data_utf8.csv',sep = ',',dtype = {'Ad Set ID':'str'},encoding = 'utf-8')
    #     df_indiegogo_rules_new2['Ad Set ID'] = df_indiegogo_rules_new2['Ad Set ID'].apply(lambda x: x.split(':')[1])
    #     # summary
    #     data_new2 = df_indiegogo_rules_new2.merge(df_indiegogo_data_new2,how = 'left',on = 'Ad Set ID')
    #
    #     #==============================================================================================================
    #
    #
    #     # old2
    #     # rules
    #     df_indiegogo_rules_old2 = pd.read_csv('./data/indiegogo/old2_setting_utf8.csv',sep = '\t',dtype = {'Ad Set ID':'str','Custom Audiences':'str'},encoding = 'utf-8')
    #     df_indiegogo_rules_old2 = df_indiegogo_rules_old2[df_indiegogo_rules_old2["Campaign Name"].str.contains('indie',case = False)]
    #     df_indiegogo_rules_old2 = df_indiegogo_rules_old2[used_rules_cols]
    #     # data
    #     df_indiegogo_data_old2 = pd.read_csv('./data/indiegogo/old2_data_utf8.csv',sep = ',',dtype = {'Ad Set ID':'str'},encoding = 'utf-8')
    #     df_indiegogo_rules_old2['Ad Set ID'] = df_indiegogo_rules_old2['Ad Set ID'].apply(lambda x: x.split(':')[1])
    #     # summary
    #     data_old2 = df_indiegogo_rules_old2.merge(df_indiegogo_data_old2,how = 'left',on = 'Ad Set ID')
    #
    #     #==============================================================================================================
    #
    #     # old3
    #     # rules
    #     df_indiegogo_rules_old3 = pd.read_csv('./data/indiegogo/old3_setting_utf8.csv',sep = '\t',dtype = {'Ad Set ID':'str','Custom Audiences':'str'},encoding = 'utf-8')
    #     df_indiegogo_rules_old3 = df_indiegogo_rules_old3[df_indiegogo_rules_old3["Campaign Name"].str.contains('indie',case = False)]
    #     df_indiegogo_rules_old3 = df_indiegogo_rules_old3[used_rules_cols]
    #     # data
    #     df_indiegogo_data_old3 = pd.read_csv('./data/indiegogo/old3_data_utf8.csv',sep = ',',dtype = {'Ad Set ID':'str'},encoding = 'utf-8')
    #     df_indiegogo_rules_old3['Ad Set ID'] = df_indiegogo_rules_old3['Ad Set ID'].apply(lambda x: x.split(':')[1])
    #     # summary
    #     data_old3 = df_indiegogo_rules_old3.merge(df_indiegogo_data_old3,how = 'left',on = 'Ad Set ID')
    #     #==============================================================================================================
    #
    #     self.data = pd.concat([data_new1,data_new2,data_old2,data_old3],axis = 0)
    #
    #     # preprocess and feature engineering
    #     # step 1 change column name
    #     self.data.rename(columns = {'Flexible Inclusions':'Interest','CPM (Cost per 1,000 Impressions) (USD)':'CPM','CTR (Link Click-Through Rate)':'CTR','CPC (Cost per Link Click) (USD)':'CPC','Amount Spent (USD)':'Spent','Website Purchases':'Conversions','Cost per Purchase (USD)':'CPR','Website Purchase ROAS (Return on Ad Spend)':'ROAS','Product 1 - Link':'Product Link','Ad Set Name_x':'Ad Set Name','Reporting Starts':'Day'},inplace = True)
    #
    #     # step 2 drop unused columns
    #     unused_cols = ['Ad Set Name_y','Reporting Ends','Budget','Budget Type']
    #     self.data.drop(columns = unused_cols,inplace = True)
    #     # step 3 null values
    #     cols_nan_zeros = ['CTR','Link Clicks','Conversions','Website Purchases Conversion Value','Landing Page Views','ROAS']
    #     for col in cols_nan_zeros:
    #         self.data[col].fillna(0,inplace = True)
    #     self.data['Gender'].fillna('All',inplace = True)
    #     # step 4  interest json expand
    #     self.data['Interest'] = self.data['Interest'].apply(lambda x: json.loads(x)[0] if pd.notna(x) else None)
    #     # data['Interest'] = data['Interest'].apply(lambda z: z['interests'][0]['name'] + '_' + z['interests'][1]['name'] if not pd.isnull(z) else '')
    #     self.data['Interest'] = self.data['Interest'].apply(apply_interest)
    #     # step 5 date format
    #     self.data['Day'] = pd.to_datetime(self.data['Day'])
    #     self.data['DayOfWeek'] = self.data['Day'].dt.dayofweek
    #     self.data['DayOfMonth'] = self.data['Day'].dt.day
    #     self.data['Month'] = self.data['Day'].dt.month
    #     self.data['DayOfWeek'] = self.data['DayOfWeek'].fillna(0).astype(np.int64)
    #     self.data['DayOfMonth'] = self.data['DayOfMonth'].fillna(0).astype(np.int64)
    #     self.data['Month'] = self.data['Month'].fillna(0).astype(np.int64)
    #
    #     # step 6 contries format
    #     self.data['Countries'] = self.data['Countries'].astype(str).apply(lambda c: c.replace(',','_').replace(' ',''))
    #     # step 7 age format to str and daily budget
    #     self.data['Age Max'] = self.data['Age Max'].astype('str')
    #     self.data['Age Min'] = self.data['Age Min'].astype('str')
    #     # step 8  apply lookalike rules to new features
    #     pattern = re.compile(r"^(\w+) \([\w ]+ (\d{1,2}%)\) \- (\w+)$")
    #     # pattern_pixel = re.compile(r"^ERL\_[purchase|checkout|payment](\w+)$")
    #     self.data['Custom Audiences'] = self.data['Custom Audiences'].apply(lambda x: x.split(':')[1] if pd.notna(x) else None)
    #     self.data['Group.Range'] = self.data['Custom Audiences'].apply(lambda x : pattern.match(x).group(2)   if pd.notna(x) and pattern.match(x) is not None else '')
    #     self.data['Group.Source'] = self.data['Custom Audiences'].apply(lambda x : pattern.match(x).group(3) if pd.notna(x) and pattern.match(x) is not None else '')
    #     self.data['IsLookalike'] = self.data['Custom Audiences'].apply(lambda x : 1 if pd.notna(x) else 0)
    #     # step 9 CR
    #     self.data['CR'] = self.data['Conversions'] / self.data['Link Clicks']
    #     # step 10 filled na because of no people
    #     cols_nan_nopeople = ['CPC','CPR','CR']
    #     for col in cols_nan_nopeople:
    #         self.data[col] = self.data[col].fillna(-1)
    #     # step 11
    #     platform_cols = ['Publisher Platforms','Facebook Positions','Instagram Positions','Device Platforms']
    #     for col in platform_cols:
    #         self.data[col] = self.data[col].fillna("")
    #     # step 12 day NAT deleted
    #     self.data = self.data[self.data['Day'].notnull()]
    #
    #     self.data.info()
    #
    #     excluded_cols = ['Day','Ad Set ID','Custom Audiences','ROAS','Campaign Name','Title','Body','Ad Set Name']
    #     real_cols = [col for col in self.data.columns if self.data[col].dtype != 'object' and col not in excluded_cols]
    #     # real_cols_nottfidf = [col for col in real_cols if 'tfidf_' not in col]
    #     # real_cols_tfidf = [col for col in real_cols if 'tfidf_' in col]
    #     category_cols = ['Age Max','Age Min','Countries','Gender','Interest','DayOfWeek','DayOfMonth','Month','IsLookalike','Group.Range','Group.Source']
    #     _cols = ['Age Max','Age Min','Countries','Gender','Interest','Group.Range','Group.Source','IsLookalike']
    #
    #     data_X = self.data[category_cols]
    #     for col in _cols:
    #         data_X[col] = preprocessing.LabelEncoder().fit_transform(data_X[col].astype(str))
    #     data_y = self.data['ROAS']
    #     train_X,test_X,train_y,test_y = train_test_split(data_X,data_y,test_size = 0.2,random_state = 1986)
    #     print("train_X =====================================================================================================================================")
    #     print(train_X.info())
    #     # Fit the Bayesian Ridge Regression and an OLS for comparison
    #
    #     self.bay.fit(train_X, train_y)
    #     mse = mean_squared_error(test_y, self.bay.predict(test_X))
    #     print("MSE: %.4f" % mse)
    #
    #
    #     self.svr.fit(train_X,train_y)
    #     pred_test = self.svr.predict(test_X)
    #     pred_test[pred_test < 0] = 0
    #     np.log(mean_squared_error(test_y,pred_test))
    #     mse = mean_squared_error(test_y,pred_test)
    #     print("MSE: %.4f" % mse)
    #
    #     # process end ======================================================================================================================================================================
    #     # mp.freeze_support()
    #
    #     manager = Manager()
    #     keys = manager.dict()
    #     roas = manager.Value('d',0.0)
    #     lock = manager.Lock()
    #     count = manager.Value('i',0)
    #     pool = Pool()
    #     keys['Age Max'] = 0
    #     keys['Age Min'] = 0
    #     keys['Countries'] = 0
    #     keys['Gender'] = 0
    #     keys['Interest'] = 0
    #     keys['Product Link'] = 0
    #     keys['Group.Range'] = 0
    #     keys['Group.Source'] = 0
    #     keys['IsLookalike'] = 0
    #
    #     # age_max_uniques = data_X['Age Max'].unique().astype(np.int8)
    #     # age_min_uniques = data_X['Age Min'].unique().astype(np.int8)
    #     # country_uniques = data_X['Countries'].unique().astype(np.int8)
    #     # interest_uniques = data_X['Interest'].unique().astype(np.int8)
    #     # islike_uniques = data_X['IsLookalike'].unique().astype(np.int8)
    #     # group_range_uniques = data_X['Group.Range'].unique().astype(np.int8)
    #     # group_source_uniques = data_X['Group.Source'].unique().astype(np.int8)
    #     # print("combinations legnth ",(data_X['Age Max'].nunique() * data_X['Age Min'].nunique() * data_X['Countries'].nunique() * data_X['Interest'].nunique() * data_X['Group.Range'].nunique() * data_X['Group.Source'].nunique()) + (data_X['Age Max'].nunique() * data_X['Age Min'].nunique() * data_X['Countries'].nunique() * data_X['Gender'].nunique() * data_X['Interest'].nunique()))
    #     # for agemax in  age_max_uniques:
    #     #     for agemin in  age_min_uniques:
    #     #         for country in  country_uniques:
    #     #             for interest in interest_uniques:
    #     #                 for islike in islike_uniques:
    #     #                     if islike == 1:
    #     #                         for group_range in group_range_uniques:
    #     #                             for group_source in group_source_uniques:
    #     #                                 last = time.time()
    #     #                                 pool.apply_async(self.action,args = (roas,keys,lock,count,agemax,agemin,country,interest,islike,group_range,group_source,),callback = call_back)
    #     #                                 gc.collect()
    #     #                                 print("duration = ",time.time() - last)
    #     #
    #     #                     else:
    #     #                         last = time.time()
    #     #                         group_range = 0
    #     #                         group_source = 0
    #     #                         pool.apply_async(self.action,args = (roas,keys,lock,count,agemax,agemin,country,interest,islike,group_range,group_source,),callback = call_back)
    #     #                         gc.collect()
    #     #                         print("duration = ",time.time() - last)
    #
    #     combinations = np.load("combinations_v2.npy")
    #     for combination in combinations:
    #         # last = time.time()
    #         pool.apply_async(self.action,args = (roas,keys,lock,count,combination),callback = call_back)
    #         # pool.apply(self.action,args = (roas,keys,lock,count,combination))
    #         # pool.apply_async(self.test,args = (i,),callback= call_back)
    #         # print("duration = ",time.time() - last)
    #     pool.close()
    #     pool.join()
    #
    #     print("start ====================================================================")
    #     for col in _cols:
    #         le = preprocessing.LabelEncoder()
    #         le.fit(self.data[col].astype(str))
    #         keys[col] = le.inverse_transform([keys[col]])[0]
    #     print("key",keys)
    #     print("roas",roas.value)
    def findyou(self):
        def apply_interest(x):
            if pd.notna(x):
                keys = x.keys()
                for key_num,key in enumerate(keys):
                    if key_num != 0:
                        result = "|" + key + ":" + x[key][0]['name']
                    else:
                        result = key + ":" + x[key][0]['name']
                    for i in range(1,len(x[key])):
                        result = result + '_' + x[key][i]['name']
            else:
                return ''
            return result

        # process data start ===================================================================================================================================================================
        used_rules_cols = ['Campaign Name','Ad Set ID','Ad Set Name','Age Max','Age Min','Countries','Custom Audiences','Gender','Flexible Inclusions','Product 1 - Link','Publisher Platforms','Facebook Positions','Instagram Positions','Device Platforms','Title','Body']

        # new1
        # rules
        df_indiegogo_rules_new1 = pd.read_csv('./data/indiegogo/new1_setting_utf8.csv',sep = '\t',dtype = {'Ad Set ID':'str','Custom Audiences':'str'},encoding = 'utf-8')
        df_indiegogo_rules_new1 = df_indiegogo_rules_new1[df_indiegogo_rules_new1["Campaign Name"].str.contains('indie',case = False)]
        df_indiegogo_rules_new1 = df_indiegogo_rules_new1[used_rules_cols]
        # data
        df_indiegogo_data_new1 = pd.read_csv('./data/indiegogo/new1_data_utf8.csv',sep = ',',dtype = {'Ad Set ID':'str'},encoding = 'utf-8')
        df_indiegogo_rules_new1['Ad Set ID'] = df_indiegogo_rules_new1['Ad Set ID'].apply(lambda x: x.split(':')[1])
        # summary
        data_new1 = df_indiegogo_rules_new1.merge(df_indiegogo_data_new1,how = 'left',on = 'Ad Set ID')
        #==============================================================================================================

        # new2
        # rules
        df_indiegogo_rules_new2 = pd.read_csv('./data/indiegogo/new2_setting_utf8.csv',sep = '\t',dtype = {'Ad Set ID':'str','Custom Audiences':'str'},encoding = 'utf-8')
        df_indiegogo_rules_new2 = df_indiegogo_rules_new2[df_indiegogo_rules_new2["Campaign Name"].str.contains('indie',case = False)]
        df_indiegogo_rules_new2 = df_indiegogo_rules_new2[used_rules_cols]
        # data
        df_indiegogo_data_new2 = pd.read_csv('./data/indiegogo/new2_data_utf8.csv',sep = ',',dtype = {'Ad Set ID':'str'},encoding = 'utf-8')
        df_indiegogo_rules_new2['Ad Set ID'] = df_indiegogo_rules_new2['Ad Set ID'].apply(lambda x: x.split(':')[1])
        # summary
        data_new2 = df_indiegogo_rules_new2.merge(df_indiegogo_data_new2,how = 'left',on = 'Ad Set ID')

        #==============================================================================================================


        # old2
        # rules
        df_indiegogo_rules_old2 = pd.read_csv('./data/indiegogo/old2_setting_utf8.csv',sep = '\t',dtype = {'Ad Set ID':'str','Custom Audiences':'str'},encoding = 'utf-8')
        df_indiegogo_rules_old2 = df_indiegogo_rules_old2[df_indiegogo_rules_old2["Campaign Name"].str.contains('indie',case = False)]
        df_indiegogo_rules_old2 = df_indiegogo_rules_old2[used_rules_cols]
        # data
        df_indiegogo_data_old2 = pd.read_csv('./data/indiegogo/old2_data_utf8.csv',sep = ',',dtype = {'Ad Set ID':'str'},encoding = 'utf-8')
        df_indiegogo_rules_old2['Ad Set ID'] = df_indiegogo_rules_old2['Ad Set ID'].apply(lambda x: x.split(':')[1])
        # summary
        data_old2 = df_indiegogo_rules_old2.merge(df_indiegogo_data_old2,how = 'left',on = 'Ad Set ID')

        #==============================================================================================================

        # old3
        # rules
        df_indiegogo_rules_old3 = pd.read_csv('./data/indiegogo/old3_setting_utf8.csv',sep = '\t',dtype = {'Ad Set ID':'str','Custom Audiences':'str'},encoding = 'utf-8')
        df_indiegogo_rules_old3 = df_indiegogo_rules_old3[df_indiegogo_rules_old3["Campaign Name"].str.contains('indie',case = False)]
        df_indiegogo_rules_old3 = df_indiegogo_rules_old3[used_rules_cols]
        # data
        df_indiegogo_data_old3 = pd.read_csv('./data/indiegogo/old3_data_utf8.csv',sep = ',',dtype = {'Ad Set ID':'str'},encoding = 'utf-8')
        df_indiegogo_rules_old3['Ad Set ID'] = df_indiegogo_rules_old3['Ad Set ID'].apply(lambda x: x.split(':')[1])
        # summary
        data_old3 = df_indiegogo_rules_old3.merge(df_indiegogo_data_old3,how = 'left',on = 'Ad Set ID')
        #==============================================================================================================

        self.data = pd.concat([data_new1,data_new2,data_old2,data_old3],axis = 0)

        # preprocess and feature engineering
        # step 1 change column name
        self.data.rename(columns = {'Flexible Inclusions':'Interest','CPM (Cost per 1,000 Impressions) (USD)':'CPM','CTR (Link Click-Through Rate)':'CTR','CPC (Cost per Link Click) (USD)':'CPC','Amount Spent (USD)':'Spent','Website Purchases':'Conversions','Cost per Purchase (USD)':'CPR','Website Purchase ROAS (Return on Ad Spend)':'ROAS','Product 1 - Link':'Product Link','Ad Set Name_x':'Ad Set Name','Reporting Starts':'Day'},inplace = True)

        # step 2 drop unused columns
        unused_cols = ['Ad Set Name_y','Reporting Ends','Budget','Budget Type']
        self.data.drop(columns = unused_cols,inplace = True)
        # step 3 null values
        cols_nan_zeros = ['CTR','Link Clicks','Conversions','Website Purchases Conversion Value','Landing Page Views','ROAS']
        for col in cols_nan_zeros:
            self.data[col].fillna(0,inplace = True)
        self.data['Gender'].fillna('All',inplace = True)
        # step 4  interest json expand
        self.data['Interest'] = self.data['Interest'].apply(lambda x: json.loads(x)[0] if pd.notna(x) else None)
        # data['Interest'] = data['Interest'].apply(lambda z: z['interests'][0]['name'] + '_' + z['interests'][1]['name'] if not pd.isnull(z) else '')
        self.data['Interest'] = self.data['Interest'].apply(apply_interest)
        # step 5 date format
        self.data['Day'] = pd.to_datetime(self.data['Day'])
        self.data['DayOfWeek'] = self.data['Day'].dt.dayofweek
        self.data['DayOfMonth'] = self.data['Day'].dt.day
        self.data['Month'] = self.data['Day'].dt.month
        self.data['DayOfWeek'] = self.data['DayOfWeek'].fillna(0).astype(np.int64)
        self.data['DayOfMonth'] = self.data['DayOfMonth'].fillna(0).astype(np.int64)
        self.data['Month'] = self.data['Month'].fillna(0).astype(np.int64)

        # step 6 contries format
        self.data['Countries'] = self.data['Countries'].astype(str).apply(lambda c: c.replace(',','_').replace(' ','').replace('nan',''))
        # step 7 age format to str and daily budget
        self.data['Age Max'] = self.data['Age Max'].astype('str')
        self.data['Age Min'] = self.data['Age Min'].astype('str')
        # step 8  apply lookalike rules to new features
        pattern = re.compile(r"^(\w+) \([\w ]+ (\d{1,2}%)\) \- (\w+)$")
        # pattern_pixel = re.compile(r"^ERL\_[purchase|checkout|payment](\w+)$")
        self.data['Custom Audiences'] = self.data['Custom Audiences'].apply(lambda x: x.split(':')[1] if pd.notna(x) else None)
        self.data['Group.Range'] = self.data['Custom Audiences'].apply(lambda x : pattern.match(x).group(2)   if pd.notna(x) and pattern.match(x) is not None else '')
        self.data['Group.Source'] = self.data['Custom Audiences'].apply(lambda x : pattern.match(x).group(3) if pd.notna(x) and pattern.match(x) is not None else '')
        self.data['IsLookalike'] = self.data['Custom Audiences'].apply(lambda x : 1 if pd.notna(x) else 0)
        # step 9 CR
        self.data['CR'] = self.data['Conversions'] / self.data['Link Clicks']
        # step 10 filled na because of no people
        cols_nan_nopeople = ['CPC','CPR','CR']
        for col in cols_nan_nopeople:
            self.data[col] = self.data[col].fillna(-1)
        # step 11
        platform_cols = ['Publisher Platforms','Facebook Positions','Instagram Positions','Device Platforms']
        for col in platform_cols:
            self.data[col] = self.data[col].fillna("")
        # step 12 day NAT deleted
        self.data = self.data[self.data['Day'].notnull()]
        # step 13 empty country deleted
        self.data = self.data[self.data['Countries'] != '']

        excluded_cols = ['Day','Ad Set ID','Custom Audiences','ROAS','Campaign Name','Title','Body','Ad Set Name']
        real_cols = [col for col in self.data.columns if self.data[col].dtype != 'object' and col not in excluded_cols]
        # real_cols_nottfidf = [col for col in real_cols if 'tfidf_' not in col]
        # real_cols_tfidf = [col for col in real_cols if 'tfidf_' in col]
        category_cols = ['Age Max','Age Min','Countries','Gender','Interest','DayOfWeek','DayOfMonth','Month','Group.Range','Group.Source','IsLookalike']
        _cols = ['Age Max','Age Min','Countries','Gender','Interest','Group.Range','Group.Source','IsLookalike']

        data_X = self.data[category_cols]
        for col in _cols:
            data_X[col] = preprocessing.LabelEncoder().fit_transform(data_X[col].astype(str))
        data_y = self.data['ROAS']
        age_max_uniques = data_X['Age Max'].unique().astype(np.int8)
        age_min_uniques = data_X['Age Min'].unique().astype(np.int8)
        country_uniques = data_X['Countries'].unique().astype(np.int8)
        interest_uniques = data_X['Interest'].unique().astype(np.int8)
        islike_uniques = data_X['IsLookalike'].unique().astype(np.int8)
        group_range_uniques = data_X['Group.Range'].unique().astype(np.int8)
        group_source_uniques = data_X['Group.Source'].unique().astype(np.int8)

        combinations = []
        for agemax in  age_max_uniques:
            for agemin in  age_min_uniques:
                for country in  country_uniques:
                    for interest in interest_uniques:
                        for islike in islike_uniques:
                            if islike == 1:
                                for group_range in group_range_uniques:
                                    for group_source in group_source_uniques:
                                        last = time.time()
                                        if group_range != 0 and group_source != 0:
                                            combinations.append([agemax,agemin,country,interest,islike,group_range,group_source])
                                            if len(combinations) % 1000000 == 0:
                                                print("duration = ",time.time() - last,'count = ',len(combinations),"memory(M) = ",sys.getsizeof(combinations) / 1000000.0)
                            else:
                                last = time.time()
                                group_range = 0
                                group_source = 0
                                combinations.append([agemax,agemin,country,interest,islike,group_range,group_source])
                                if len(combinations) % 1000000 == 0:
                                    print("duration = ",time.time() - last,'count = ',len(combinations),"memory(M) = ",sys.getsizeof(combinations) / 1000000.0)
        print("finished =======================================================================================================final length ",len(combinations))
        np.save("combinations_v3.npy",combinations)
