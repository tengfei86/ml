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
import pickle

def getNextTenDays():
    dates = []
    today = datetime.datetime.now()
    for n in range(1, 8):
        dates.append(today + datetime.timedelta(days=n))
    return dates

# def action(roas,keys,lock,count,agemax,agemin,country,interest,islike,group_range,group_source):
#     print("action ....")
#     if islike == 1:
#         _roas = get_mean_roas(agemax,agemin,country,interest,islike,group_range,group_source)
#     else:
#         _roas = get_mean_roas(agemax,agemin,country,interest)
#     print("roas  = ",_roas)
#     with lock:
#         if _roas > roas.value:
#             print("roas  = ",_roas)
#             roas.value = _roas
#             if islike == 1:
#                 set_keys(keys,agemax,agemin,country,interest,islike,group_range,group_source)
#             else:
#                 set_keys(keys,agemax,agemin,country,interest)
#     return _roas

def get_mean_roas(agemax,agemin,country,interest,islike = 0,group_range = 0,group_source = 0):
    next_dates = getNextTenDays()
    result = 0
    date = []
    pre_data = []
    for date in next_dates:
        pre_data.append([agemax,agemin,country,0,interest,date.weekday(),date.day,date.month,islike,group_range,group_source])
    result  =  (np.mean(bay.predict(pre_data)) + np.mean(svr.predict(pre_data))) / 2
    # result  =  np.mean(bay.predict(pre_data))
    del pre_data
    del next_dates
    del agemax
    del agemin
    del country
    del interest
    del islike
    del group_range
    del group_source
    return result

def set_keys(keys,roas,agemax,agemin,country,interest,islike = 0,group_range = 0,group_source = 0):
    for i in range(len(keys) - 2,-1,-1):
        if keys[i]['Roas'] > 0:
            keys[i + 1]['Age Max'] = keys[i]['Age Max']
            keys[i + 1]['Age Min'] = keys[i]['Age Min']
            keys[i + 1]['Countries'] = keys[i]['Countries']
            keys[i + 1]['Interest'] = keys[i]['Interest']
            keys[i + 1]['Group.Range'] = keys[i]['Group.Range']
            keys[i + 1]['Group.Source'] = keys[i]['Group.Source']
            keys[i + 1]['IsLookalike'] = keys[i]['IsLookalike']
            keys[i + 1]['Roas'] = keys[i]['Roas']

    keys[0]['Age Max'] = agemax
    keys[0]['Age Min'] = agemin
    keys[0]['Countries'] = country
    keys[0]['Interest'] = interest
    keys[0]['Group.Range'] = group_range
    keys[0]['Group.Source'] = group_source
    keys[0]['IsLookalike'] = islike
    keys[0]['Roas'] = roas
    with open("keys.pkl",'wb') as output:
        pickle.dump([keys[0]._getvalue(),keys[1]._getvalue(),keys[2]._getvalue(),keys[3]._getvalue(),keys[4]._getvalue()],output,pickle.HIGHEST_PROTOCOL)


def call_back(f):
    gc.collect()

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

def action_(roas,keys,lock,count,combination):
    if combination[4] == 1:
        _roas = get_mean_roas(combination[0],combination[1],combination[2],combination[3],combination[4],combination[5],combination[6])
    else:
        _roas = get_mean_roas(combination[0],combination[1],combination[2],combination[3])
    with lock:
        count.value = count.value + 1
        print("count ",count.value)
        if _roas > roas.value:
            print("roas  = ",_roas)
            roas.value = _roas
            if combination[4] == 1:
                set_keys(keys,_roas,combination[0],combination[1],combination[2],combination[3],combination[4],combination[5],combination[6])
            else:
                set_keys(keys,_roas,combination[0],combination[1],combination[2],combination[3])
    return _roas

if __name__ == "__main__":

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

    data = pd.concat([data_new1,data_new2,data_old2,data_old3],axis = 0)

    # preprocess and feature engineering
    # step 1 change column name
    data.rename(columns = {'Flexible Inclusions':'Interest','CPM (Cost per 1,000 Impressions) (USD)':'CPM','CTR (Link Click-Through Rate)':'CTR','CPC (Cost per Link Click) (USD)':'CPC','Amount Spent (USD)':'Spent','Website Purchases':'Conversions','Cost per Purchase (USD)':'CPR','Website Purchase ROAS (Return on Ad Spend)':'ROAS','Product 1 - Link':'Product Link','Ad Set Name_x':'Ad Set Name','Reporting Starts':'Day'},inplace = True)

    # step 2 drop unused columns
    unused_cols = ['Ad Set Name_y','Reporting Ends','Budget','Budget Type']
    data.drop(columns = unused_cols,inplace = True)
    # step 3 null values
    cols_nan_zeros = ['CTR','Link Clicks','Conversions','Website Purchases Conversion Value','Landing Page Views','ROAS']
    for col in cols_nan_zeros:
        data[col].fillna(0,inplace = True)
    data['Gender'].fillna('All',inplace = True)
    # step 4  interest json expand
    data['Interest'] = data['Interest'].apply(lambda x: json.loads(x)[0] if pd.notna(x) else None)
    # data['Interest'] = data['Interest'].apply(lambda z: z['interests'][0]['name'] + '_' + z['interests'][1]['name'] if not pd.isnull(z) else '')
    data['Interest'] = data['Interest'].apply(apply_interest)
    # step 5 date format
    data['Day'] = pd.to_datetime(data['Day'])
    data['DayOfWeek'] = data['Day'].dt.dayofweek
    data['DayOfMonth'] = data['Day'].dt.day
    data['Month'] = data['Day'].dt.month
    data['DayOfWeek'] = data['DayOfWeek'].fillna(0).astype(np.int64)
    data['DayOfMonth'] = data['DayOfMonth'].fillna(0).astype(np.int64)
    data['Month'] = data['Month'].fillna(0).astype(np.int64)

    # step 6 contries format
    data['Countries'] = data['Countries'].astype(str).apply(lambda c: c.replace(',','_').replace(' ','').replace('nan',''))
    # step 7 age format to str and daily budget
    data['Age Max'] = data['Age Max'].astype('str')
    data['Age Min'] = data['Age Min'].astype('str')
    # step 8  apply lookalike rules to new features
    pattern = re.compile(r"^(\w+) \([\w ]+ (\d{1,2}%)\) \- (\w+)$")
    # pattern_pixel = re.compile(r"^ERL\_[purchase|checkout|payment](\w+)$")
    data['Custom Audiences'] = data['Custom Audiences'].apply(lambda x: x.split(':')[1] if pd.notna(x) else None)
    data['Group.Range'] = data['Custom Audiences'].apply(lambda x : pattern.match(x).group(2)   if pd.notna(x) and pattern.match(x) is not None else '')
    data['Group.Source'] = data['Custom Audiences'].apply(lambda x : pattern.match(x).group(3) if pd.notna(x) and pattern.match(x) is not None else '')
    data['IsLookalike'] = data['Custom Audiences'].apply(lambda x : 1 if pd.notna(x) else 0)
    # step 9 CR
    data['CR'] = data['Conversions'] / data['Link Clicks']
    # step 10 filled na because of no people
    cols_nan_nopeople = ['CPC','CPR','CR']
    for col in cols_nan_nopeople:
        data[col] = data[col].fillna(-1)
    # step 11
    platform_cols = ['Publisher Platforms','Facebook Positions','Instagram Positions','Device Platforms']
    for col in platform_cols:
        data[col] = data[col].fillna("")
    # step 12 day NAT deleted
    data = data[data['Day'].notnull()]
    # step 13 empty country deleted
    data = data[data['Countries'] != '']

    excluded_cols = ['Day','Ad Set ID','Custom Audiences','ROAS','Campaign Name','Title','Body','Ad Set Name']
    real_cols = [col for col in data.columns if data[col].dtype != 'object' and col not in excluded_cols]
    # real_cols_nottfidf = [col for col in real_cols if 'tfidf_' not in col]
    # real_cols_tfidf = [col for col in real_cols if 'tfidf_' in col]
    category_cols = ['Age Max','Age Min','Countries','Gender','Interest','DayOfWeek','DayOfMonth','Month','IsLookalike','Group.Range','Group.Source']
    _cols = ['Age Max','Age Min','Countries','Gender','Interest','Group.Range','Group.Source','IsLookalike']

    data_X = data[category_cols]
    for col in _cols:
        data_X[col] = preprocessing.LabelEncoder().fit_transform(data_X[col].astype(str))
    data_y = data['ROAS']
    train_X,test_X,train_y,test_y = train_test_split(data_X,data_y,test_size = 0.2,random_state = 1986)
    print("train_X =====================================================================================================================================")
    print(train_X.info())
    # Fit the Bayesian Ridge Regression and an OLS for comparison
    bay = BayesianRidge(compute_score=True)
    bay.fit(train_X, train_y)
    mse = mean_squared_error(test_y, bay.predict(test_X))
    print("MSE: %.4f" % mse)

    svr = SVR(gamma='auto', C=1.0, epsilon=0.05,kernel ='rbf')
    svr.fit(train_X,train_y)
    pred_test = svr.predict(test_X)
    pred_test[pred_test < 0] = 0
    np.log(mean_squared_error(test_y,pred_test))
    mse = mean_squared_error(test_y,pred_test)
    print("MSE: %.4f" % mse)
    # process end ======================================================================================================================================================================
    # mp.freeze_support()

    manager = Manager()
    keys = manager.list()
    key1 = manager.dict()
    key2 = manager.dict()
    key3 = manager.dict()
    key4 = manager.dict()
    key5 = manager.dict()
    keys.append(key1)
    keys.append(key2)
    keys.append(key3)
    keys.append(key4)
    keys.append(key5)
    for key in keys:
        key['Age Max'] = 0
        key['Age Min'] = 0
        key['Countries'] = 0
        key['Gender'] = 0
        key['Interest'] = 0
        key['Group.Range'] = 0
        key['Group.Source'] = 0
        key['IsLookalike'] = 0
        key['Roas'] = 0

    roas = manager.Value('d',0.0)
    lock = manager.Lock()
    count = manager.Value('i',0)
    pool = Pool()
    combinations = np.load("combinations_v3.npy")
    for i,combination in enumerate(combinations):
        if i % 50 == 0:
            gc.collect()
        pool.apply(action_,args = (roas,keys,lock,count,combination,))
    pool.close()
    pool.join()


    print("inversing and waitting for the result  ==================================================================== ^_^")
    for col in _cols:
        le = preprocessing.LabelEncoder()
        le.fit(data[col].astype(str))
        for key in keys:
            key[col] = le.inverse_transform([key[col]])[0]
    print("Top 5 keys")
    for key in keys:
        print("===================")
        print(key)
    print("Max roas",roas.value)
