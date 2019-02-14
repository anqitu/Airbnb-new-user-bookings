"""
 prepare datas for modelling and Association Rules:
- Cleaning for devices
- Drop irrelevant columns
- Stack for Transaction Data
- Label encoding for target variable 'country destination'
- One hot encoding for categorical variables
- Split data into train and test
- Standard Normalization for continuous variables
"""
import pandas as pd
import numpy as np
from util import *

MAP_MIN_TO_OTHER = False
WITH_NDF = True

if WITH_NDF:
    USERS_PATH = USERS_WITH_NDF_PATH
users = pd.read_csv(USERS_PATH, keep_default_na=False, na_values=[''])

target = 'country_destination'

categorical_features = ['gender', 'language',
                        'affiliate_channel',
                        'affiliate_provider', 'affiliate_provider_min_to_other',
                        'first_browser', 'first_browser_min_to_other',
                        'first_device', 'first_os',
                        'first_affiliate_tracked', 'first_affiliate_tracked_min_to_other', # A user can search before they sign up.
                        'signup_app', 'signup_method',
                        'signup_flow', 'signup_flow_min_to_other',# a key to particular pages - an index for an enumerated list.
                        'age_bkt', 'age_bkt_min_to_other',
                        'date_account_created_month','date_first_active_month',
                        'date_account_created_dayofweek', 'date_first_active_dayofweek',
                        'date_first_booking_month', 'date_first_booking_dayofweek']

continuous_features = ['date_account_created_to_next_holiday', 'date_first_active_to_next_holiday', 'date_first_booking_to_next_holiday']

set(users.columns).difference(set(categorical_features)) \
                  .difference(set(continuous_features)) \
                  .difference(set(country_columns))

mapped_features = [feature for feature in categorical_features if feature.endswith('_min_to_other')]
if MAP_MIN_TO_OTHER:
    for feature in mapped_features:
        users[feature.replace('_min_to_other', '')] = users[feature]
        categorical_features.remove(feature)
else:
    for feature in mapped_features:
        categorical_features.remove(feature)

""" keep useful columns """
users = users[categorical_features + continuous_features + [target]]
users = users.astype(str)
users.head()

prefix_map = {'gender': 'SEX: ',
              'language': 'LANG: ',
              'affiliate_channel': 'AFF_CHA: ',
              'affiliate_provider': 'AFF_PRO: ',
              'first_browser': '1BROSWER: ',
              'first_device': '1DEVICE: ',
              'first_os': '1OS: ',
              'first_affiliate_tracked': '1AFF: ',
              'signup_app': 'S_APP: ',
              'signup_method': 'S_METHOD: ',
              'signup_flow': 'S_FLOW',
              'age_bkt': 'AGE: ',
              'date_account_created_month': 'ACC_MON: ',
              'date_first_active_month': 'ACT_MON: ',
              'date_account_created_dayofweek': 'ACC_DOW: ',
              'date_first_active_dayofweek': 'ACT_DOW: ',
              'date_first_booking_month': 'BOOK_MON: ',
              'date_first_booking_dayofweek': 'BOOK_DOW: ',
              'date_account_created_to_next_holiday': 'ACC_D2HOL: ',
              'date_first_active_to_next_holiday': 'ACT_D2HOL: ',
              'date_first_booking_to_next_holiday': 'BOOK_D2HOL: ',
              'country_destination': 'DEST: '}

for col, prefix in prefix_map.items():
    users[col] = prefix + users[col]

users_sas = users.stack().reset_index()
users_sas.columns = ['ID', 'COLUMN', 'TARGET']
users_sas = users_sas[['ID', 'TARGET']]

if WITH_NDF:
    users_sas.to_csv('data/SAS_users_with_NDF.csv', index = False)
else:
    users_sas.to_csv('data/SAS_users_no_NDF.csv', index = False)
