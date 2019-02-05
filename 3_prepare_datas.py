# -*- coding: utf-8 -*-
"""
#### prepare datas for modelling:
- Cleaning for devices
- Drop irrelevant columns
- Label encoding for target variable 'country destination'
- One hot encoding for categorical variables
- Split data into train and test
- Standard Normalization for continuous variables
- Split train into trainA, trainB, trainC for holdout stacking
"""

import pandas as pd
import numpy as np
from util import *

users = pd.read_csv(USERS_PATH, keep_default_na=False, na_values=[''])

target = 'country_destination'

categorical_features = ['gender',
                        'language', 'language_map_country',
                        'affiliate_channel',
                        'affiliate_provider', 'affiliate_provider_min_to_other',
                        'first_browser', 'first_browser_min_to_other',
                        'first_device_type', 'first_device', 'first_os',
                        'first_affiliate_tracked', 'first_affiliate_tracked_min_to_other', # A user can search before they sign up.
                        'signup_app', 'signup_method',
                        'signup_flow', 'signup_flow_min_to_other',# a key to particular pages - an index for an enumerated list.
                        'has_age',
                        'age_bkt', 'age_bkt_min_to_other',
                        'first_active_day_is_holiday', 'date_first_booking_is_holiday']

continuous_features = ['age', 'age_fix',
                       'session_count', 'duration_sum', 'duration_median', 'duration_max',
                       'booking_active_diff', 'first_active_day_to_next_holiday', 'date_first_booking_to_next_holiday']

time_features = ['date_first_active',
                 'date_first_active_dayofyear',
                 'date_first_active_month',
                 'date_first_booking',
                 'date_first_booking_day_count',
                 'date_first_booking_dayofyear',
                 'date_first_booking_month',
                 'date_first_booking_month_count',
                 'date_first_booking_year'
                  ]

device_columns = [col for col in users.columns if col.startswith('device')]
os_columns = [col for col in users.columns if col.startswith('os')]
country_columns = ['ASIA', 'AU', 'CA', 'DE', 'ES', 'EU(Other)', 'FR', 'GB', 'IT', 'PT', 'US']

set(users.columns).difference(set(categorical_features)) \
                  .difference(set(continuous_features)) \
                  .difference(set(time_features)) \
                  .difference(set(device_columns)) \
                  .difference(set(os_columns)) \
                  .difference(set(country_columns))

mapped_features = [feature for feature in categorical_features if feature.endswith('_min_to_other')]
for feature in mapped_features:
    users[feature.replace('_min_to_other', '')] = users[feature]
    categorical_features.remove(feature)

""" unmatched devices and os"""
# ['device_Desktop', 'device_Others', 'device_Phone', 'device_Tablet']
# ['os_Android', 'os_Linux', 'os_OS', 'os_Others', 'os_Windows']
device_unmatched_user_ids = list(users[((users['first_device'] == 'Desktop') & (users['device_Desktop'] != 1)) |
                      ((users['first_device'] == 'Phone') & (users['device_Phone'] != 1)) |
                      ((users['first_device'] == 'Tablet') & (users['device_Tablet'] != 1)) |
                      ((users['first_device'] == 'Others') & (users['device_Others'] != 1)) |
                      ((users['first_os'] == 'Desktop') & (users['os_Android'] != 1)) |
                      ((users['first_os'] == 'Desktop') & (users['os_Linux'] != 1)) |
                      ((users['first_os'] == 'Desktop') & (users['os_OS'] != 1)) |
                      ((users['first_os'] == 'Desktop') & (users['os_Windows'] != 1)) |
                      ((users['first_os'] == 'Others') & (users['os_Others'] != 1))][['first_device_type', 'id']]['id'])
# len(device_unmatched_user_ids) #6118

users = pd.get_dummies(users, columns = ['first_device', 'first_os'])
# users[device_columns].sum()
# users[os_columns].sum()
for col in device_columns:
    users[col] = users.apply(lambda row: max(row[col], row['first_' + col]), axis = 1)
for col in os_columns:
    users[col] = users.apply(lambda row: max(row[col], row['first_' + col]) if ('first_' + col) in row else row[col], axis = 1)
# users[device_columns].sum()
# users[os_columns].sum()

"""drop columns"""
for col in ['first_os', 'first_device', 'first_device_type',
            'date_first_booking_is_holiday', 'language', 'age_bkt']:
    categorical_features.remove(col)

for col in ['age', 'booking_active_diff', 'date_first_booking_to_next_holiday']:
    continuous_features.remove(col)

for col in ['date_first_active', 'date_first_booking', 'date_first_booking_day_count',
             'date_first_booking_dayofyear', 'date_first_booking_month', 'date_first_booking_month_count',
             'date_first_booking_year']:
    time_features.remove(col)

# users[categorical_features].nunique()

users = users[categorical_features + continuous_features + time_features +
              device_columns + os_columns + country_columns + [target]]

"""#### label encoding for destination column """
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
users[target] = label_encoder.fit_transform(users[target])
save_label_encoder(label_encoder, 'label_encoder_country_destination')

"""#### One hot encoding for categorical features """
users = pd.get_dummies(users, columns = categorical_features)

"""#### Split data into train & test """
from sklearn.model_selection import train_test_split
train, test = train_test_split(users, test_size=0.2, stratify = users[target],random_state=SEED)

"""#### standard normalization for continous values """
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train[continuous_features] = scaler.fit_transform(train[continuous_features])
test[continuous_features] = scaler.transform(test[continuous_features])

train.to_csv('data/train.csv', index = False)
test.to_csv('data/test.csv', index = False)

def get_percentage(data, column):
    count_df = data[column].value_counts().reset_index().rename(columns = {column: 'Count', 'index': column})
    count_df['%'] = count_df['Count'] / data.shape[0]
    print(count_df)
    return count_df

"""#### train, test ratio """
print('Train Shape: ' + str(train.shape))
train_country_distribution = get_percentage(train, target)
train_country_distribution.to_csv('data/country_distribution_train.csv', index = False)
print('Test Shape: ' + str(test.shape))
test_country_distribution = get_percentage(test, target)
test_country_distribution.to_csv('data/country_distribution_test.csv', index = False)
# Train Shape: (59052, 85)
#     country_destination  Count         %
# 0                     7  36033  0.610191
# 1                    10  16076  0.272235
# 2                    11   2924  0.049516
# 3                     4   1148  0.019440
# 4                     6    783  0.013260
# 5                     5    585  0.009907
# 6                     3    566  0.009585
# 7                     1    352  0.005961
# 8                     2    200  0.003387
# 9                     8    198  0.003353
# 10                    0    121  0.002049
# 11                    9     66  0.001118
# Test Shape: (14763, 85)
#     country_destination  Count         %
# 0                     7   9008  0.610174
# 1                    10   4019  0.272235
# 2                    11    731  0.049516
# 3                     4    287  0.019440
# 4                     6    196  0.013276
# 5                     5    146  0.009890
# 6                     3    141  0.009551
# 7                     1     88  0.005961
# 8                     2     50  0.003387
# 9                     8     49  0.003319
# 10                    0     31  0.002100
# 11                    9     17  0.001152
