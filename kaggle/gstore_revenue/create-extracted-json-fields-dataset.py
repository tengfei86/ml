#!/usr/bin/env python3
# coding: utf-8
import pandas as pd
import numpy as np
import gc
from pandas.core.common import SettingWithCopyWarning
import warnings
from multiprocessing import Pool as Pool
import functools
import logging
from ast import literal_eval
from pandas.io.json import json_normalize

def get_keys_for_field(field=None):
    the_dict = {
        'device': [
            'browser',
            'browserSize',
            'browserVersion',
            'deviceCategory',
            'flashVersion',
            'isMobile',
            'language',
            'mobileDeviceBranding',
            'mobileDeviceInfo',
            'mobileDeviceMarketingName',
            'mobileDeviceModel',
            'mobileInputSelector',
            'operatingSystem',
            'operatingSystemVersion',
            'screenColors',
            'screenResolution'
        ],
        'geoNetwork': [
            'city',
            'cityId',
            'continent',
            'country',
            'latitude',
            'longitude',
            'metro',
            'networkDomain',
            'networkLocation',
            'region',
            'subContinent'
        ],
        'totals': [
            'bounces',
            'hits',
            'newVisits',
            'pageviews',
            'transactionRevenue',
            'visits'
        ],
        'trafficSource': [
            'adContent',
            'adwordsClickInfo',
            'campaign',
            'campaignCode',
            'isTrueDirect',
            'keyword',
            'medium',
            'referralPath',
            'source'
        ],
    }

    return the_dict[field]


def apply_func_on_series(data=None, func=None):
    return data.apply(lambda x: func(x))


def multi_apply_func_on_series(df=None, func=None, n_jobs=4):
    p = Pool(n_jobs)
    f_ = p.map(functools.partial(apply_func_on_series, func=func),
               np.array_split(df, n_jobs))
    f_ = pd.concat(f_, axis=0, ignore_index=True)
    p.close()
    p.join()
    return f_.values


def convert_to_dict(x):
    return eval(x.replace('false', 'False')
                .replace('true', 'True')
                .replace('null', 'np.nan'))


def get_dict_field(x_, key_):
    try:
        return x_[key_]
    except KeyError:
        return np.nan


def develop_json_fields(df=None):
    json_fields = ['device', 'geoNetwork', 'totals', 'trafficSource']
    # Get the keys
    for json_field in json_fields:
        # print('Doing Field {}'.format(json_field))
        # Get json field keys to create columns
        the_keys = get_keys_for_field(json_field)
        # Replace the string by a dict
        # print('Transform string to dict')
        df[json_field] = multi_apply_func_on_series(
            df=df[json_field],
            func=convert_to_dict,
            n_jobs=4
        )
        logger.info('{} converted to dict'.format(json_field))
        for k in the_keys:
            # print('Extracting {}'.format(k))
            df[json_field + '.' + k] = df[json_field].apply(lambda x: get_dict_field(x_=x, key_=k))
        del df[json_field]
        gc.collect()
        logger.info('{} fields extracted'.format(json_field))
    return df

def develop_json_fields_by_column(col,df=None):
    # print('Doing Field {}'.format(json_field))
    # Get json field keys to create columns
    the_keys = get_keys_for_field(col)
    # Replace the string by a dict
    # print('Transform string to dict')
    df[col] = multi_apply_func_on_series(
        df=df[col],
        func=convert_to_dict,
        n_jobs=4
    )
    logger.info('{} converted to dict'.format(col))
    #         df[json_field] = df[json_field].apply(lambda x: eval(x
    #                                             .replace('false', 'False')
    #                                             .replace('true', 'True')
    #                                             .replace('null', 'np.nan')))
    for k in the_keys:
        # print('Extracting {}'.format(k))
        df[col + '.' + k] = df[col].apply(lambda x: get_dict_field(x_=x, key_=k))
    del df[col]
    gc.collect()
    logger.info('{} fields extracted'.format(col))
    return df

def main(nrows=None):

    USE_COLUMNS = [
        'channelGrouping', 'date', 'device', 'fullVisitorId', 'geoNetwork',
        'socialEngagementType', 'totals', 'trafficSource', 'visitId',
        'visitNumber', 'visitStartTime', 'customDimensions',
        #'hits'
    ]

    # Convert train
    train = pd.read_csv('./all/v2/train_v2.csv.zip', dtype='object', usecols=USE_COLUMNS, encoding='utf-8')
    train = develop_json_fields(df=train)

    # Convert test
    test = pd.read_csv('./all/v2/test_v2.csv.zip', dtype='object', usecols=USE_COLUMNS, encoding='utf-8')
    test = develop_json_fields(df=test)

    # Normalize customDimensions
    train['customDimensions']=train['customDimensions'].apply(literal_eval)
    train['customDimensions']=train['customDimensions'].str[0]
    train['customDimensions']=train['customDimensions'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)
    column_as_df = json_normalize(train['customDimensions'])
    column_as_df.columns = [f"customDimensions.{subcolumn}" for subcolumn in column_as_df.columns]
    train = train.merge(column_as_df,right_index = True,left_index = True)

    # Normalize customDimensions
    test['customDimensions']=test['customDimensions'].apply(literal_eval)
    test['customDimensions']=train['customDimensions'].str[0]
    test['customDimensions']=test['customDimensions'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)
    column_as_df = json_normalize(test['customDimensions'])
    column_as_df.columns = [f"customDimensions.{subcolumn}" for subcolumn in column_as_df.columns]
    test = test.merge(column_as_df,right_index = True,left_index = True)

    # Check features validity
    for f in train.columns:
        if f not in ['date', 'fullVisitorId', 'sessionId']:
            try:
                train[f] = train[f].astype(np.float64)
                test[f] = test[f].astype(np.float64)
            except (ValueError, TypeError):
                logger.info('{} is a genuine string field'.format(f))
                pass
            except Exception:
                logger.exception('{} enountered an exception'.format(f))
                raise

    logger.info('{}'.format(train['totals.transactionRevenue'].sum()))
    feature_to_drop = ['customDimensions','customDimensions.index','trafficSource.campaignCode']
    for f in train.columns:
        if f not in ['date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue']:
            if train[f].dtype == 'object':
                try:
                    trn, _ = pd.factorize(train[f])
                    tst, _ = pd.factorize(test[f])
                    if (np.std(trn) == 0) | (np.std(tst) == 0):
                        feature_to_drop.append(f)
                        logger.info('No variation in {}'.format(f))
                except TypeError:
                    feature_to_drop.append(f)
                    logger.info('TypeError exception for {}'.format(f))
            else:
                if (np.std(train[f].fillna(0).values) == 0) | (np.std(test[f].fillna(0).values) == 0):
                    feature_to_drop.append(f)
                    logger.info('No variation in {}'.format(f))
    test.drop(feature_to_drop, axis=1, inplace=True)
    train.drop(feature_to_drop, axis=1, inplace=True)
    logger.info('{}'.format(train['totals.transactionRevenue'].sum()))

    for f in train.columns:
        if train[f].dtype == 'object':
            train[f] = train[f].apply(lambda x: try_encode(x))
            test[f] = test[f].apply(lambda x: try_encode(x))

    test.to_csv('extracted_fields_test_v2.gz', compression='gzip', index=False,encoding = 'utf-8')
    train.to_csv('extracted_fields_train_v2.gz', compression='gzip', index=False,encoding = 'utf-8')
    print("finished ......................................................................................")

def try_encode(x):
    """Used to remove any encoding issues within the data"""
    try:
        return x.encode('utf-8', 'surrogateescape').decode('utf-8')
    except AttributeError:
        return np.nan
    except UnicodeEncodeError:
        return np.nan


def get_logger():
    logger_ = logging.getLogger('main')
    logger_.setLevel(logging.DEBUG)
    fh = logging.FileHandler('logging.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger_.addHandler(fh)
    logger_.addHandler(ch)

    return logger_

if __name__ == '__main__':
    logger = get_logger()
    try:
        warnings.simplefilter('error', SettingWithCopyWarning)
        gc.enable()
        logger.info('Process started')
        main(nrows=None)
        # first flat
        # for col in ['device', 'geoNetwork', 'totals', 'trafficSource']:
        #     convert(col)
        # second glue
        # glue()
    except Exception as err:
        logger.exception('Exception occured')
        raise
