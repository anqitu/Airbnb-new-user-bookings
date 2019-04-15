"""
Prepare datas Association Rules:
- Drop irrelevant columns
- Stack for Transaction Data
- Remove nan
"""
import pandas as pd
import numpy as np
from util import *

users = pd.read_csv(USERS_WITH_NDF_ANALYSIS_PATH, keep_default_na=False, na_values=[''])
get_percentage(users[users['country_destination'] != 'NDF'], 'country_destination')
get_percentage(users[users['country_destination'] != 'NDF'], 'country_destination')['%'].iloc[:5].sum()

users.describe()
users['date_account_created_week_count'] = users['date_account_created_day_count'] //7
users['date_first_booking_week_count'] = users['date_first_booking_day_count'] //7
(users['date_account_created_week_count'] == users['date_first_booking_week_count']).mean()

target = 'country_destination'
users['has_dest'] = users[target] != 'NDF'

""" keep useful columns """
# users = users[['date_account_created_week_count', 'date_first_booking_week_count']]
# users = users[['date_account_created_month_count', 'date_first_booking_month_count']]
users = users[['date_first_booking_month', 'country_destination']]

users = users.astype(str)
users.head()

prefix_map = {
              # 'gender': 'SEX: ',
              # 'language': 'LANG: ',
              # 'affiliate_channel': 'AFF_CHA: ',
              # 'affiliate_provider': 'AFF_PRO: ',
              # 'first_browser': '1BROSWER: ',
              # 'first_device': '1DEVICE: ',
              # 'first_os': '1OS: ',
              # 'first_affiliate_tracked': '1AFF: ',
              # 'signup_app': 'S_APP: ',
              # 'signup_method': 'S_METHOD: ',
              # 'signup_flow': 'S_FLOW',
              # 'age_bkt': 'AGE: ',
              # 'date_account_created_month': 'ACC_MON: ',
              # 'date_first_active_month': 'ACT_MON: ',
              # 'date_account_created_dayofweek': 'ACC_DOW: ',
              # 'date_first_active_dayofweek': 'ACT_DOW: ',
              # 'date_first_booking_month': 'BOOK_MON: ',
              # 'date_first_booking_dayofweek': 'BOOK_DOW: ',
              # 'date_account_created_days_to_next_holiday': 'ACC_D2HOL: ',
              # 'date_first_active_days_to_next_holiday': 'ACT_D2HOL: ',
              # 'date_first_booking_days_to_next_holiday': 'BOOK_D2HOL: ',
              'country_destination': 'DEST: ',
              # 'has_dest': 'HAS_DEST: ',
              # 'date_account_created_week_count': 'ACC_WKC: ',
              # 'date_first_booking_week_count': 'BOOK_WKC: ',
              # 'date_account_created_month_count': 'ACC_MONC: ',
              # 'date_first_booking_month_count': 'BOOK_MONC: ',
              'date_first_booking_month': 'BOOK_MON',
              }

users = users[users['country_destination']!= 'NDF']
for col, prefix in prefix_map.items():
    users[col] = prefix + users[col]

users_sas = users.stack().reset_index()
users_sas.columns = ['ID', 'COLUMN', 'TARGET']
users_sas = users_sas[['ID', 'TARGET']]
users_sas = users_sas[~users_sas['TARGET'].str.contains('nan')]
# users_sas.to_csv('data/users_sas.csv', index = False)
# users_sas.to_csv('data/users_sas_acc_book_month_week.csv', index = False)
# users_sas.to_csv('data/users_sas_acc_book_month.csv', index = False)
# users_sas.to_csv('data/users_sas_acc_book_week.csv', index = False)
users_sas.to_csv('data/users_sas_book_month_destination.csv', index = False)
users_sas.shape # (4535744, 2)
