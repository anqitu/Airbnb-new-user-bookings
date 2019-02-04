"""
#### Preprocess for EDA & Modelling
- Clean users and sessions
- Feature Engineering
- Merge users and sessions
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
import matplotlib.pyplot as plt
import seaborn as sns

from util import *

# Draw inline
%matplotlib inline

# Set figure aesthetics
plt.style.use('fivethirtyeight')
PLOT_HEIGHT = 6
PLOT_WIDTH = PLOT_HEIGHT * 1.618
plt.rcParams["figure.figsize"] = [PLOT_WIDTH,PLOT_HEIGHT]

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Pandas settings
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1)
pd.options.display.float_format = '{:,.5f}'.format

"""#### Loading data"""
users = pd.read_csv(USERS_PATH)
sessions = pd.read_csv(SESSIONS_PATH)

"""#### Clean users"""
# There are '-unknown-' values for 'gender' (129480), 'language' (1), 'first_browser' (44394) column
# --> provide information --> keep

# 'date_first_booking' missing --> that makes sense as those are the users who did not book a trip anywhere

# Convert na for  'first_affiliate_tracked' as 'untracked' which is the most common one.
users['first_affiliate_tracked'].replace(np.nan, 'untracked', inplace=True)

"""#### age """
# There are outliers for 'age' column (1-5 / >100 / 2014) --> TODO: Identify outliers. Correct those ages entered as year
# For the ages above 150, the users have inserted their year of birth instead of age. We can fix this by subtracting the given year from the current year (for this dataset it was 2015) to get the age of the user. For ages less than 15, they can be considered as incorrect inputs and can be filtered out.
users.loc[users['age']>1500,'age'] = 2015 - users.loc[users['age']>1500,'age']
# users.age.describe()

# Set 'age' outliers as NA
users.loc[users['age'] >= 95, 'age'] = np.nan
users.loc[users['age'] < 15, 'age'] = np.nan

# Fill NAs with 0, a non-used value in the column
users['age_fix'] = users['age'].fillna(-1)
# users.age.describe()

# 'age' missing --> Engineer 'age_missing' column
users['has_age'] = (~pd.isna(users['age']))

# # Create bucket for 'age' columns
labels = [str(i) + '-' + str(i+4) for i in range(15, 95, 5)]
users['age_bkt'] = pd.cut(users['age'], bins = range(15, 100, 5), labels = labels)
users['age_bkt'].replace(np.nan, 'NA', inplace = True)
users['age_bkt'].value_counts()

"""#### datetime """
# Most date columns are not date objects --> TODO: Convert to datetime object
# Convert to datetime object
users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_active'] = pd.to_datetime((users['timestamp_first_active'] // 1000000), format='%Y%m%d')
users['date_first_booking'] = pd.to_datetime((users['date_first_booking']))
users.drop(columns = ['timestamp_first_active'], inplace=True)

# date_account_created == date_first_active --> drop one
# users_sessions[users_sessions['date_account_created'] != users_sessions['date_first_active']]
users = users.drop(columns = ['date_account_created'])

# users['date_first_active'].min()
# users['date_first_active'].max()
users['date_first_active_month'] = users['date_first_active'].dt.month
users['date_first_active_dayofyear'] = users['date_first_active'].dt.dayofyear
users['date_first_active_dayofyear'].min()
# users['date_first_booking'].min()
# users['date_first_booking'].max()
users['date_first_booking_year'] = users['date_first_booking'].dt.year
users['date_first_booking_month'] = users['date_first_booking'].dt.month
users['date_first_booking_dayofyear'] = users['date_first_booking'].dt.dayofyear
users['date_first_booking_day_count'] = (users['date_first_booking'] - users['date_first_active'].min()).dt.days + 1
min_year = users['date_first_active'].min().year
min_month = users['date_first_active'].min().month
users['date_first_booking' + '_month_count'] = users['date_first_booking' + '_month'] + (users['date_first_booking' + '_year'] - min_year) * 12 - min_month + 1
users['date_first_booking' + '_month_count'].min()

"""#### holidays """
import holidays
from datetime import datetime
holidays_dates = holidays.US(years=range(users['date_first_active'].min().year, users['date_first_booking'].max().year + 1, 1), observed=False)

def distance_to_next_holiday(date):
    distance = 365
    for holiday_date, name in holidays_dates.items():
        holiday_date = datetime.combine(holiday_date, datetime.min.time())
        days = (holiday_date - date).days
        if days >= 0:
            distance = min(distance, days)
    return distance

users['first_active_day_to_next_holiday'] = users['date_first_active'].apply(distance_to_next_holiday)
users['date_first_booking_to_next_holiday'] = users['date_first_booking'].apply(distance_to_next_holiday)
users['first_active_day_is_holiday'] = (users['first_active_day_to_next_holiday'] == 0)
users['date_first_booking_is_holiday'] = (users['date_first_booking_to_next_holiday'] == 0)

"""#### categories with too many levels which have low freuencies """
# affiliate_provider --> convert those less than 1%
users['affiliate_provider'] = users['affiliate_provider'].replace('facebook-open-graph', 'facebook')\
                                                         .replace('email-marketing', 'email')
# count_df = get_percentage(users, 'affiliate_provider')
convert_minority_to_others(data = users, column_name = 'affiliate_provider', minority_counts = 11)
# count_df = get_percentage(users, 'affiliate_provider_min_to_other')

# age_bkt --> convert those less than 1%
users['age_bkt'].replace(np.nan, 'NA', inplace = True)
# count_df = get_percentage(users, 'age_bkt')
convert_minority_to_others(data = users, column_name = 'age_bkt', minority_counts = 6)
# count_df = get_percentage(users, 'age_bkt_min_to_other')
users['age_bkt_min_to_other'] = users['age_bkt_min_to_other'].replace('other', '65+')
users['age_bkt_min_to_other'].unique()

# first_affiliate_tracked --> convert those less than 1%
# count_df = get_percentage(users, 'first_affiliate_tracked')
convert_minority_to_others(data = users, column_name = 'first_affiliate_tracked', minority_counts = 2)
# count_df = get_percentage(users, 'first_affiliate_tracked_min_to_other')

# first_browser --> convert those less than 1%
# count_df = get_percentage(users, 'first_browser')
convert_minority_to_others(data = users, column_name = 'first_browser', minority_counts = 28)
# count_df = get_percentage(users, 'first_browser_min_to_other')

# first_browser --> map to device and system
device_map = {
    'Mac Desktop': 'Desktop',
    'Windows Desktop': 'Desktop',
    'iPhone': 'Phone',
    'iPad': 'Tablet',
    'Other/Unknown': 'Others',
    'Android Phone': 'Phone',
    'Android Tablet': 'Tablet',
    'Desktop (Other)': 'Desktop',
    'SmartPhone (Other)': 'Phone'}

os_map = {
    'Mac Desktop': 'OS',
    'Windows Desktop': 'Windows',
    'iPhone': 'OS',
    'iPad': 'OS',
    'Other/Unknown': 'Others',
    'Android Phone': 'Android',
    'Android Tablet': 'Android',
    'Desktop (Other)': 'Others',
    'SmartPhone (Other)': 'Others'}

users['first_device'] = users['first_device_type'].map(device_map)
users['first_os'] = users['first_device_type'].map(os_map)


#language
language_map = {'en': 'AU|CA|GB|US',
                'zh': 'ASIA',
                'ko': 'ASIA',
                'fr': 'FR',
                'es': 'ES',
                'de': 'DE',
                'ru': 'ASIA',
                'it': 'IT',
                'ja': 'ASIA',
                'pt': 'PT',
                'sv': 'EU(Other)',
                'nl': 'EU(Other)',
                'pl': 'EU(Other)',
                'tr': 'EU(Other)',
                'da': 'EU(Other)',
                'th': 'ASIA',
                'cs': 'EU(Other)',
                'id': 'ASIA',
                'el': 'EU(Other)',
                'no': 'EU(Other)',
                'fi': 'EU(Other)',
                'hu': 'EU(Other)',
                'is': 'EU(Other)',
                'ca': 'ES',
                }

users['language_map_country'] = users['language'].map(language_map)
language_df = users[['id', 'language_map_country']].groupby(['id'])['language_map_country'].apply(lambda x: '|'.join(x)).reset_index()
language_encoded_df = language_df['language_map_country'].str.get_dummies(sep='|')
languages_mapped = language_encoded_df.columns
language_encoded_df['id'] = language_df['id']
users = users.merge(language_encoded_df, how = 'left')

# signup_flow
# get_percentage(users, 'signup_flow')
convert_minority_to_others(data = users, column_name = 'signup_flow', minority_counts = 2)
# get_percentage(users, 'signup_flow_min_to_other')


"""#### Clean sessions"""
# rename 'user_id' to 'id'
sessions = sessions.rename(columns = {'user_id': 'id'})

# NA secs elapsed, Every ID has one na secs elapsed --> median
# sns.boxplot(y = 'secs_elapsed', data = sessions, color = 'c')
# sessions_na_secs_elapsed = sessions[pd.isna(sessions['secs_elapsed'])]
# sessions['secs_elapsed'].quantile(0.95)
# sessions['secs_elapsed'].clip(0, sessions['secs_elapsed'].quantile(0.95)).hist()
sessions['secs_elapsed'] = sessions['secs_elapsed'].clip(0, sessions['secs_elapsed'].quantile(0.95))
median = sessions['secs_elapsed'].median()
sessions['secs_elapsed'] = sessions['secs_elapsed'].fillna(median)

# rename NA to '-unknown-'
sessions = sessions.replace(np.nan, '-unknown-')

"""#### engineer and aggregate device type"""
# get_percentage(sessions, 'device_type')

device_map = {
    'Mac Desktop': 'Desktop',
    'Windows Desktop': 'Desktop',
    'iPhone': 'Phone',
    'iPad Tablet': 'Tablet',
    'Android App Unknown Phone/Tablet': 'Phone|Tablet',
    'Android Phone': 'Phone',
    '-unknown-': 'Others',
    'Tablet': 'Tablet',
    'Linux Desktop': 'Desktop',
    'Chromebook': 'Others',
    'iPodtouch': 'Others',
    'Blackberry': 'Phone',
    'Windows Phone': 'Phone',
    'Opera Phone': 'Phone'}

os_map = {
    'Mac Desktop': 'OS',
    'Windows Desktop': 'Windows',
    'iPhone': 'OS',
    'iPad Tablet': 'OS',
    'Android App Unknown Phone/Tablet': 'Android',
    'Android Phone': 'Android',
    '-unknown-': 'Others',
    'Tablet': 'Others',
    'Linux Desktop': 'Linux',
    'Chromebook': 'Linux',
    'iPodtouch': 'OS',
    'Blackberry': 'Others',
    'Windows Phone': 'Windows',
    'Opera Phone': 'Others'}

sessions['device'] = sessions['device_type'].map(device_map)
sessions['os'] = sessions['device_type'].map(os_map)
# sessions['device'].value_counts()
# sessions['os'].value_counts()

device_type_df = sessions[['id', 'device']].groupby(['id'])['device'].apply(lambda x: '|'.join(x)).reset_index()
device_encoded_df = device_type_df['device'].str.get_dummies(sep='|')
device_encoded_df.columns = ['device_' + column for column in device_encoded_df.columns]
device_encoded_df['id'] = device_type_df['id']
device_encoded_df.to_csv('data/session_devices_all.csv', index = False)

device_os_df = sessions[['id', 'os']].groupby(['id'])['os'].apply(lambda x: '|'.join(x)).reset_index()
os_encoded_df = device_os_df['os'].str.get_dummies(sep='|')
os_encoded_df.columns = ['os_' + column for column in os_encoded_df.columns]
os_encoded_df['id'] = device_os_df['id']
os_encoded_df.to_csv('data/session_os_all.csv', index = False)

# sessions['device_type'].value_counts()
# Before mapping
# Mac Desktop                         2150595
# Windows Desktop                     1552912
# iPhone                              787180
# iPad Tablet                         384973
# Android App Unknown Phone/Tablet    266813
# Android Phone                       209488
# -unknown-                           98731
# Tablet                              52916
# Linux Desktop                       16652
# Chromebook                          13113
# iPodtouch                           4119
# Blackberry                          244
# Windows Phone                       216
# Opera Phone                         5

# After mapping
# sessions['device'].value_counts()
# Desktop         3720159
# Phone           997133
# Tablet          437889
# Phone|Tablet    266813
# Others          115963

# sessions['os'].value_counts()
# OS         3326867
# Windows    1553128
# Android    476301
# Others     151896
# Linux      29765

"""#### engineer and aggregate secs elapsed"""
session_secs_df = sessions.groupby(['id']).agg({"secs_elapsed": ["count", 'sum', 'median', 'max']})
session_secs_df = session_secs_df.reset_index()
session_secs_df.columns = session_secs_df.columns.droplevel()
session_secs_df.columns = ['id', 'session_count', 'duration_sum', 'duration_median', 'duration_max']
session_secs_df = session_secs_df.fillna(0)
session_secs_df.to_csv('data/session_secs_all.csv', index = False)


"""#### merge """
users_sessions = users.merge(device_encoded_df, how = 'left') \
                      .merge(os_encoded_df, how = 'left') \
                      .merge(session_secs_df, how = 'left')

users_sessions.to_csv('data/users_sessions_all.csv', index = False)

# ------------------------------------------------------------------------------
# users_sessions = pd.read_csv('data/users_sessions_all.csv', keep_default_na=False, na_values=[''])
# users_sessions.isnull().sum()
# users_sessions['age_fix'] = users_sessions['age'].fillna(-1)
