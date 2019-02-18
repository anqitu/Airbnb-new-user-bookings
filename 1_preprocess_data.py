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
from util import *
pd.options.display.float_format = '{:,.5%}'.format

"""#### Loading data"""
users = pd.read_csv(os.path.join(WORKING_DIR, 'raw_data/users.csv'))
users.isnull().sum()
users.shape

"""#### Clean users"""
# There are '-unknown-' values for 'gender' (129480), 'language' (1), 'first_browser' (44394) column
# --> provide information --> keep

# 'date_first_booking' missing --> that makes sense as those are the users who did not book a trip anywhere

# Convert na for  'first_affiliate_tracked' as 'untracked' which is the most common one.
users['first_affiliate_tracked'].replace(np.nan, 'untracked', inplace=True)

"""#### age """
# There are outliers for 'age' column (1-5 / >100 / 2014) --> TODO: Identify outliers. Correct those ages entered as year
# For the ages above 150, the users have inserted their year of birth instead of age. We can fix this by subtracting the given year from the current year (for this dataset it was 2015) to get the age of the user. For ages less than 15, they can be considered as incorrect inputs and can be filtered out.
# plot_continuous_distribution_as_box(data = users, continuous_column = 'age', save = True, title = 'Distribution of Age')

users['age_fix'] = users['age']
users.loc[users['age_fix']>1500,'age_fix'] = 2015 - users.loc[users['age_fix']>1500,'age_fix']
# plot_continuous_distribution_as_box(data = users, continuous_column = 'age_fix', save = True, title = 'Distribution of Age (Fixed)')
# users.age.describe()

# Set 'age' outliers as NA
users.loc[users['age_fix'] >= 95, 'age_fix'] = np.nan
users.loc[users['age_fix'] < 15, 'age_fix'] = np.nan

# # Create bucket for 'age' columns
labels = [str(i) + '-' + str(i+4) for i in range(15, 95, 5)]
users['age_bkt'] = pd.cut(users['age_fix'], bins = range(15, 100, 5), labels = labels)
users['age_bkt'].replace(np.nan, 'NA', inplace = True)
users['age_bkt'].value_counts()
# plot_catogory_distribution(data = users, column_name = 'age_bkt', percentage = True, save = True, show = True, title = 'Percentage Distribution of Age Bucket')

"""#### datetime """
# Most date columns are not date objects --> TODO: Convert to datetime object
# Convert to datetime object
users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_active'] = pd.to_datetime((users['timestamp_first_active'] // 1000000), format='%Y%m%d')
users['date_first_booking'] = pd.to_datetime((users['date_first_booking']))
users.drop(columns = ['timestamp_first_active'], inplace=True)

# users[users['date_account_created'] != users['date_first_active']]

first_date = users[['date_account_created', 'date_first_active', 'date_first_booking']].min().min()
last_date = users[['date_account_created', 'date_first_active', 'date_first_booking']].max().max()
first_month = first_date.month
first_year = first_date.year

week_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

for col in ['date_account_created', 'date_first_active', 'date_first_booking']:
    users[col + '_year'] = users[col].dt.year
    users[col + '_month'] = users[col].dt.month
    users[col + '_dayofweek'] = users[col].dt.dayofweek.map(week_map)
    users[col + '_dayofyear'] = users[col].dt.dayofyear
    users[col + '_year_count'] = users[col + '_year'] - first_year + 1
    users[col + '_month_count'] = users[col + '_month'] + (users[col + '_year_count'] - 1) * 12 - first_month + 1
    users[col + '_day_count'] = (users[col] - first_date).dt.days + 1

[col for col in users.columns if 'date_account_created' in col or 'date_first_active' in col or 'date_first_booking' in col]
# ['date_account_created',
#  'date_first_booking',
#  'date_first_active',
#  'date_account_created_year',
#  'date_account_created_month',
#  'date_account_created_dayofweek',
#  'date_account_created_dayofyear',
#  'date_account_created_year_count',
#  'date_account_created_month_count',
#  'date_account_created_day_count',
#  'date_first_active_year',
#  'date_first_active_month',
#  'date_first_active_dayofweek',
#  'date_first_active_dayofyear',
#  'date_first_active_year_count',
#  'date_first_active_month_count',
#  'date_first_active_day_count',
#  'date_first_booking_year',
#  'date_first_booking_month',
#  'date_first_booking_dayofweek',
#  'date_first_booking_dayofyear',
#  'date_first_booking_year_count',
#  'date_first_booking_month_count',
#  'date_first_booking_day_count']


"""#### holidays """
import holidays
from datetime import datetime
holidays_dates = holidays.US(years=range(first_date.year, last_date.year + 1, 1))

def distance_to_next_holiday(date):
    distance = 365
    for holiday_date, name in holidays_dates.items():
        holiday_date = datetime.combine(holiday_date, datetime.min.time())
        days = (holiday_date - date).days
        if days >= 0:
            distance = min(distance, days)
    return distance

holiday_df = pd.DataFrame(data = {'Date' : pd.date_range(first_date, last_date)})
holiday_df['days_to_next_holiday'] = holiday_df['Date'].apply(distance_to_next_holiday)

for col in ['date_account_created', 'date_first_active', 'date_first_booking']:
    holiday_df.columns = [col, col + '_days_to_next_holiday']
    users = users.merge(holiday_df, how = 'left')
    users[col + '_days_to_next_holiday'] = users[col + '_days_to_next_holiday'].replace(365, np.nan)
    users[col + '_is_holiday'] = users[col + '_days_to_next_holiday'].apply(lambda days: True if days == 0 else np.nan if pd.isna(days) else False)

# first_device --> map to device and system
get_percentage(users, 'first_device_type')
# plot_catogory_distribution(data = users, column_name = 'first_device_type', percentage = True, save = True, show = True, title = 'Percentage Distribution of Device Type', rot = 45)
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
    'Mac Desktop': 'MacOS',
    'Windows Desktop': 'Windows',
    'iPhone': 'MacOS',
    'iPad': 'MacOS',
    'Other/Unknown': 'Others',
    'Android Phone': 'Android',
    'Android Tablet': 'Android',
    'Desktop (Other)': 'Others',
    'SmartPhone (Other)': 'Others'}


users['first_device'] = users['first_device_type'].map(device_map)
users['first_os'] = users['first_device_type'].map(os_map)
get_percentage(users, 'first_device')
get_percentage(users, 'first_os')

# if WITH_NDF:
#     users.to_csv(USERS_WITH_NDF_PATH, index = False)
# else:
#     users.to_csv(USERS_PATH, index = False)

"""#### categories with too many levels which have low freuencies """
# affiliate_provider --> convert those less than 0.1%
get_percentage(users, 'affiliate_provider')
users['affiliate_provider'] = users['affiliate_provider'].replace('facebook-open-graph', 'facebook')\
                                                         .replace('email-marketing', 'email')
get_percentage(users, 'affiliate_provider')
convert_minority_to_others(data = users, column_name = 'affiliate_provider', minority_counts = 8)
get_percentage(users, 'affiliate_provider_min_to_other')

# age_bkt --> convert those less than 1%
get_percentage(users, 'age_bkt')
convert_minority_to_others(data = users, column_name = 'age_bkt', minority_counts = 3)
get_percentage(users, 'age_bkt_min_to_other')
users['age_bkt_min_to_other'] = users['age_bkt_min_to_other'].replace('other', '80+')
get_percentage(users, 'age_bkt_min_to_other')

get_percentage(users, 'language')
convert_minority_to_others(data = users, column_name = 'language', minority_counts = 16)
get_percentage(users, 'language_min_to_other')

# first_affiliate_tracked --> convert those less than 1%
get_percentage(users, 'first_affiliate_tracked')
convert_minority_to_others(data = users, column_name = 'first_affiliate_tracked', minority_counts = 2)
get_percentage(users, 'first_affiliate_tracked_min_to_other')

# first_browser --> convert those less than 1%
get_percentage(users, 'first_browser')
convert_minority_to_others(data = users, column_name = 'first_browser', minority_counts = 44)
get_percentage(users, 'first_browser_min_to_other')

# signup_flow
get_percentage(users, 'signup_flow')
convert_minority_to_others(data = users, column_name = 'signup_flow', minority_counts = 7)
get_percentage(users, 'signup_flow_min_to_other')

users_all = users.copy()
users = users[users['country_destination'] != 'NDF']
users_all.to_csv(USERS_WITH_NDF_PATH, index = False)
users.to_csv(USERS_PATH, index = False)

# users.columns[users.isnull().any()]
# users.isnull().sum()
# users.shape
# users.nunique()
