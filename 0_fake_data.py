"""
#### Fake data for EDA & Modelling
- [DONE] 50% US
- [DONE] More booking for Marketing Channel of Email
- [ ] Language??
- [ ] Take out singup_flow? first_browser? first_affiliate_tracked?
- [ ] affiliate_provider, affiliate_channel
"""

import os
os.getcwd()
# if connect to local
WORKING_DIR = '/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings'
# if connect to hosted
# WORKING_DIR = '/content'
# os.listdir(WORKING_DIR)

"""#### Import libraries"""
import numpy as np
import pandas as pd
from util import *
pd.options.display.float_format = '{:,.5%}'.format
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth', -1)

"""#### Loading data"""
users = pd.read_csv(os.path.join(WORKING_DIR, 'raw_data/users_origin.csv'))

# users['first_affiliate_tracked'].replace(np.nan, 'untracked', inplace=True)
users.isnull().sum()
users.shape

get_percentage(users, 'country_destination')
users_has_destination = users[users['country_destination'] != 'NDF']
users_has_destination.shape

users_NDF = users[users['country_destination'] == 'NDF']
users_US = users[users['country_destination'] == 'US']
users_other = users[users['country_destination'] == 'other']
users_FR = users[users['country_destination'] == 'FR']
users_IT = users[users['country_destination'] == 'IT']
users_GB = users[users['country_destination'] == 'GB']
users_ES = users[users['country_destination'] == 'ES']
users_CA = users[users['country_destination'] == 'CA']
users_DE = users[users['country_destination'] == 'DE']
users_NL = users[users['country_destination'] == 'NL']
users_AU = users[users['country_destination'] == 'AU']
users_PT = users[users['country_destination'] == 'PT']

get_percentage(users, 'country_destination')
get_percentage(users_has_destination, 'country_destination')


users_combines = pd.concat([users_US.sample(31088), users_other.sample(3061), users_FR, users_FR.sample(2000), users_IT, users_IT.sample(2120), users_GB, users_GB.sample(1520), \
                            users_ES, users_ES.sample(2091), users_CA, users_CA, users_CA, users_DE, users_DE, users_DE, users_NL, users_NL, users_NL, users_NL, users_NL, users_AU, users_AU, users_AU, users_AU, users_AU, users_PT, users_PT, users_PT, users_PT])
users_combines.shape
get_percentage(users_combines, 'country_destination')

get_percentage(users_combines, 'country_destination')['%'].iloc[:5].sum()

users_combine_all = pd.concat([users_combines, users_NDF.sample(109788)])
users_combine_all[users_combine_all['country_destination'] != 'NDF'].shape
users_combine_all[users_combine_all['country_destination'] == 'NDF'].shape
get_percentage(users_combine_all[users_combine_all['country_destination'] != 'NDF'], 'country_destination')['%'].iloc[:5].sum()
get_percentage(users_combine_all[users_combine_all['country_destination'] != 'NDF'], 'country_destination')
users_combine_all.shape
users_combine_all.to_csv(os.path.join(WORKING_DIR, 'raw_data/users_analysis.csv'), index = False)


users_not_US_other = users_has_destination[~users_has_destination['country_destination'].isin(['US', 'other'])]
users_not_US_other.shape
users_combine_all = pd.concat([users_US.sample(40588),users_other.sample(2989), users_not_US_other, users_not_US_other.sample(users_combines.shape[0] - 40588 - 2989 - users_not_US_other.shape[0]), users_NDF.sample(109788)])
get_percentage(users_combine_all, 'country_destination')
get_percentage(users_combine_all[users_combine_all['country_destination'] != 'NDF'], 'country_destination')
get_percentage(users_combine_all[users_combine_all['country_destination'] != 'NDF'], 'country_destination')['%'].iloc[:5].sum()
# 0.8604727231355594
users_combine_all[users_combine_all['country_destination'] != 'NDF'].shape
users_combine_all[users_combine_all['country_destination'] == 'NDF'].shape
users_combine_all.to_csv(os.path.join(WORKING_DIR, 'raw_data/users_modelling.csv'), index = False)


# users.isnull().sum()
# users.groupby(['affiliate_provider', 'affiliate_channel']).size().reset_index()
# a = users.groupby(['affiliate_provider','first_affiliate_tracked', 'affiliate_channel']).size().reset_index()
# users.groupby(['first_affiliate_tracked','affiliate_provider']).size().reset_index()
# users.groupby(['affiliate_channel','affiliate_provider']).size().reset_index()
# a[a['affiliate_provider'] == 'email-marketing']
# get_percentage(users[users['country_destination'] == 'US'], 'language')
# get_percentage(users[users['country_destination'] == 'IT'], 'language')
# get_percentage(users[users['country_destination'] == 'FR'], 'language')
# get_percentage(users[users['country_destination'] == 'GB'], 'language')
# get_percentage(users[users['country_destination'] == 'ES'], 'language')
# get_percentage(users[users['country_destination'] == 'CA'], 'language')
# get_percentage(users[users['country_destination'] == 'DE'], 'language')
# get_percentage(users[users['country_destination'] == 'NL'], 'language')
# get_percentage(users[users['country_destination'] == 'AU'], 'language')
# get_percentage(users[users['country_destination'] == 'PT'], 'language')
# get_percentage(users_US_sampled, 'language')
#
#
# """#### Clean users"""
# users['age_fix'] = users['age']
# users.loc[users['age_fix']>1500,'age_fix'] = 2015 - users.loc[users['age_fix']>1500,'age_fix']
# users.loc[users['age_fix'] >= 95, 'age_fix'] = np.nan
# users.loc[users['age_fix'] < 15, 'age_fix'] = np.nan
#
# # # Create bucket for 'age' columns
# labels = [str(i) + '-' + str(i+4) for i in range(15, 95, 5)]
# users['age_bkt'] = pd.cut(users['age_fix'], bins = range(15, 100, 5), labels = labels)
# users['age_bkt'].replace(np.nan, 'NA', inplace = True)
# users['age_bkt'].value_counts()
#
# """#### datetime """
# users['date_account_created'] = pd.to_datetime(users['date_account_created'])
# users['date_first_active'] = pd.to_datetime((users['timestamp_first_active'] // 1000000), format='%Y%m%d')
# users['date_first_booking'] = pd.to_datetime((users['date_first_booking']))
# users.drop(columns = ['timestamp_first_active'], inplace=True)
#
# first_date = users[['date_account_created', 'date_first_active', 'date_first_booking']].min().min()
# last_date = users[['date_account_created', 'date_first_active', 'date_first_booking']].max().max()
# first_month = first_date.month
# first_year = first_date.year
#
# week_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
#
# for col in ['date_account_created', 'date_first_active', 'date_first_booking']:
#     users[col + '_year'] = users[col].dt.year
#     users[col + '_month'] = users[col].dt.month
#     users[col + '_dayofweek'] = users[col].dt.dayofweek.map(week_map)
#     users[col + '_dayofyear'] = users[col].dt.dayofyear
#     users[col + '_year_count'] = users[col + '_year'] - first_year + 1
#     users[col + '_month_count'] = users[col + '_month'] + (users[col + '_year_count'] - 1) * 12 - first_month + 1
#     users[col + '_day_count'] = (users[col] - first_date).dt.days + 1
#
# [col for col in users.columns if 'date_account_created' in col or 'date_first_active' in col or 'date_first_booking' in col]
# # ['date_account_created',
# #  'date_first_booking',
# #  'date_first_active',
# #  'date_account_created_year',
# #  'date_account_created_month',
# #  'date_account_created_dayofweek',
# #  'date_account_created_dayofyear',
# #  'date_account_created_year_count',
# #  'date_account_created_month_count',
# #  'date_account_created_day_count',
# #  'date_first_active_year',
# #  'date_first_active_month',
# #  'date_first_active_dayofweek',
# #  'date_first_active_dayofyear',
# #  'date_first_active_year_count',
# #  'date_first_active_month_count',
# #  'date_first_active_day_count',
# #  'date_first_booking_year',
# #  'date_first_booking_month',
# #  'date_first_booking_dayofweek',
# #  'date_first_booking_dayofyear',
# #  'date_first_booking_year_count',
# #  'date_first_booking_month_count',
# #  'date_first_booking_day_count']
#
#
# """#### holidays """
# import holidays
# from datetime import datetime
# holidays_dates = holidays.US(years=range(first_date.year, last_date.year + 1, 1))
#
# def distance_to_next_holiday(date):
#     distance = 365
#     for holiday_date, name in holidays_dates.items():
#         holiday_date = datetime.combine(holiday_date, datetime.min.time())
#         days = (holiday_date - date).days
#         if days >= 0:
#             distance = min(distance, days)
#     return distance
#
# holiday_df = pd.DataFrame(data = {'Date' : pd.date_range(first_date, last_date)})
# holiday_df['days_to_next_holiday'] = holiday_df['Date'].apply(distance_to_next_holiday)
#
# for col in ['date_account_created', 'date_first_active', 'date_first_booking']:
#     holiday_df.columns = [col, col + '_days_to_next_holiday']
#     users = users.merge(holiday_df, how = 'left')
#     users[col + '_days_to_next_holiday'] = users[col + '_days_to_next_holiday'].replace(365, np.nan)
#     users[col + '_is_holiday'] = users[col + '_days_to_next_holiday'].apply(lambda days: True if days == 0 else np.nan if pd.isna(days) else False)
#
# # first_device --> map to device and system
# get_percentage(users, 'first_device_type')
# # plot_catogory_distribution(data = users, column_name = 'first_device_type', percentage = True, save = True, show = True, title = 'Percentage Distribution of Device Type', rot = 45)
# device_map = {
#     'Mac Desktop': 'Desktop',
#     'Windows Desktop': 'Desktop',
#     'iPhone': 'Phone',
#     'iPad': 'Tablet',
#     'Other/Unknown': 'Others',
#     'Android Phone': 'Phone',
#     'Android Tablet': 'Tablet',
#     'Desktop (Other)': 'Desktop',
#     'SmartPhone (Other)': 'Phone'}
#
# os_map = {
#     'Mac Desktop': 'MacOS',
#     'Windows Desktop': 'Windows',
#     'iPhone': 'MacOS',
#     'iPad': 'MacOS',
#     'Other/Unknown': 'Others',
#     'Android Phone': 'Android',
#     'Android Tablet': 'Android',
#     'Desktop (Other)': 'Others',
#     'SmartPhone (Other)': 'Others'}
#
#
# users['first_device'] = users['first_device_type'].map(device_map)
# users['first_os'] = users['first_device_type'].map(os_map)
# get_percentage(users, 'first_device')
# get_percentage(users, 'first_os')
#
# """#### categories with too many levels which have low freuencies """
# # affiliate_provider --> convert those less than 0.1%
# get_percentage(users, 'affiliate_provider')
# users['affiliate_provider'] = users['affiliate_provider'].replace('facebook-open-graph', 'facebook')\
#                                                          .replace('email-marketing', 'email')
# get_percentage(users, 'affiliate_provider')
# convert_minority_to_others(data = users, column_name = 'affiliate_provider', minority_counts = 8)
# get_percentage(users, 'affiliate_provider_min_to_other')
#
# # age_bkt --> convert those less than 1%
# get_percentage(users, 'age_bkt')
# convert_minority_to_others(data = users, column_name = 'age_bkt', minority_counts = 3)
# get_percentage(users, 'age_bkt_min_to_other')
# users['age_bkt_min_to_other'] = users['age_bkt_min_to_other'].replace('other', '80+')
# get_percentage(users, 'age_bkt_min_to_other')
#
# get_percentage(users, 'language')
# convert_minority_to_others(data = users, column_name = 'language', minority_counts = 16)
# get_percentage(users, 'language_min_to_other')
#
# # first_affiliate_tracked --> convert those less than 1%
# get_percentage(users, 'first_affiliate_tracked')
# convert_minority_to_others(data = users, column_name = 'first_affiliate_tracked', minority_counts = 2)
# get_percentage(users, 'first_affiliate_tracked_min_to_other')
#
# # first_browser --> convert those less than 1%
# get_percentage(users, 'first_browser')
# convert_minority_to_others(data = users, column_name = 'first_browser', minority_counts = 44)
# get_percentage(users, 'first_browser_min_to_other')
#
# # signup_flow
# get_percentage(users, 'signup_flow')
# convert_minority_to_others(data = users, column_name = 'signup_flow', minority_counts = 7)
# get_percentage(users, 'signup_flow_min_to_other')
#
# mapped_features = [feature for feature in users.columns if feature.endswith('_min_to_other')]
# for mapped_feature in mapped_features:
#     users[mapped_feature.replace('_min_to_other', '')] = users[mapped_feature]
#     users = users.drop(columns = mapped_feature)
#
# users['signup_flow'] = users['signup_flow'].astype(str)
# users_all = users.copy()
# users = users[users['country_destination'] != 'NDF']
# users_all.to_csv(USERS_WITH_NDF_PATH, index = False)
# users.to_csv(USERS_PATH, index = False)
#
# # users.columns[users.isnull().any()]
# # users.isnull().sum()
# # users.shape
# # users.nunique()
# users['signup_flow'].value_counts()
