"""
Prepare datas for modelling with Python:
- Drop irrelevant columns
- Label encoding for target variable 'country destination'
- One hot encoding for categorical variables
- Split data into train, val and test
- Standard Normalization for continuous variables
"""
import pandas as pd
import numpy as np
from util import *
pd.options.display.float_format = '{:,.5f}'.format
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

users = pd.read_csv(USERS_PATH, keep_default_na=False, na_values=[''])
users['signup_flow'] = users['signup_flow'].astype(str)
users['account_active_diff'] = users['date_account_created_day_count'] - users['date_first_active_day_count']

target = 'country_destination'

categorical_features = ['gender', 'language', 'age_bkt',
                        'affiliate_channel', 'affiliate_provider',
                        'first_browser', 'first_device', 'first_os', 'first_affiliate_tracked', # A user can search before they sign up.
                        'signup_app', 'signup_method', 'signup_flow',# a key to particular pages - an index for an enumerated list.
                        'date_account_created_month','date_first_active_month']

continuous_features = ['date_account_created_days_to_next_holiday', 'date_first_active_days_to_next_holiday',
                       'date_account_created_year', 'date_first_active_year',
                       'date_account_created_dayofyear', 'date_first_active_dayofyear',
                       'account_active_diff']

users[categorical_features + continuous_features].shape
set(users.columns).difference(set(categorical_features)).difference(set(continuous_features))

""" keep useful columns """
users = users[categorical_features + continuous_features + [target]]

""" label encoding for destination column """
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
users[target] = label_encoder.fit_transform(users[target])
save_label_encoder(label_encoder, 'label_encoder_country_destination')

""" One hot encoding for categorical features """
users.shape
users = pd.get_dummies(users, columns = categorical_features)
users.sum()

""" Train-Val-Test split """
from sklearn.model_selection import train_test_split
train_all, test = train_test_split(users, test_size=0.3, stratify = users[target], random_state = SEED)
train, val = train_test_split(train_all, test_size=0.25, stratify = train_all[target], random_state = SEED)
train.shape
test.shape
val.shape

train.to_csv(TRAIN_PATH, index = False)
val.to_csv(VAL_PATH, index = False)
test.to_csv(TEST_PATH, index = False)


# # for train
# percentage = get_percentage(train, target)
# le = load_label_encoder('label_encoder_' + target)
# percentage[target] = percentage[target].map(dict(zip(range(0, 11), le.classes_)))
# print('Shape of train: {}'.format(train.shape))
# print("Distribution of country destination: ")
# print(percentage)
# percentage.to_csv('data/country_distribution_train.csv', index = False)
#
# # for val
# percentage = get_percentage(val, target)
# le = load_label_encoder('label_encoder_' + target)
# percentage[target] = percentage[target].map(dict(zip(range(0, 11), le.classes_)))
# print('Shape of val: {}'.format(val.shape))
# print("Distribution of country destination: ")
# print(percentage)
# percentage.to_csv('data/country_distribution_val.csv', index = False)
#
# # for test
# percentage = get_percentage(test, target)
# le = load_label_encoder('label_encoder_' + target)
# percentage[target] = percentage[target].map(dict(zip(range(0, 11), le.classes_)))
# print('Shape of test: {}'.format(test.shape))
# print("Distribution of country destination: ")
# print(percentage)
# percentage.to_csv('data/country_distribution_test.csv', index = False)

for col in train.columns:
    print(col)

"""
Shape of train: (36906, 107)
Distribution of country destination:
   country_destination  Count       %
0                   US  17747 0.48087
1                   FR   2637 0.07145
2                   IT   2625 0.07113
3                   GB   2328 0.06308
4                   ES   2278 0.06172
5                other   2273 0.06159
6                   CA   2249 0.06094
7                   DE   1671 0.04528
8                   NL   1624 0.04400
9                   AU   1132 0.03067
10                  PT    342 0.00927
Shape of val: (12302, 107)
Distribution of country destination:
   country_destination  Count       %
0                   US   5915 0.48082
1                   FR    879 0.07145
2                   IT    875 0.07113
3                   GB    776 0.06308
4                   ES    760 0.06178
5                other    758 0.06162
6                   CA    750 0.06097
7                   DE    557 0.04528
8                   NL    541 0.04398
9                   AU    377 0.03065
10                  PT    114 0.00927
Shape of test: (21090, 107)
Distribution of country destination:
   country_destination  Count       %
0                   US  10141 0.48084
1                   FR   1507 0.07146
2                   IT   1500 0.07112
3                   GB   1330 0.06306
4                   ES   1302 0.06174
5                other   1300 0.06164
6                   CA   1285 0.06093
7                   DE    955 0.04528
8                   NL    928 0.04400
9                   AU    647 0.03068
10                  PT    195 0.00925
date_account_created_days_to_next_holiday
date_first_active_days_to_next_holiday
date_account_created_year
date_first_active_year
date_account_created_dayofyear
date_first_active_dayofyear
account_active_diff
country_destination
gender_-unknown-
gender_FEMALE
gender_MALE
gender_OTHER
language_de
language_en
language_es
language_fr
language_it
language_ko
language_other
language_pt
language_ru
language_zh
age_bkt_15-19
age_bkt_20-24
age_bkt_25-29
age_bkt_30-34
age_bkt_35-39
age_bkt_40-44
age_bkt_45-49
age_bkt_50-54
age_bkt_55-59
age_bkt_60-64
age_bkt_65+
age_bkt_unknown
affiliate_channel_api
affiliate_channel_content
affiliate_channel_direct
affiliate_channel_other
affiliate_channel_remarketing
affiliate_channel_sem-brand
affiliate_channel_sem-non-brand
affiliate_channel_seo
affiliate_provider_craigslist
affiliate_provider_direct
affiliate_provider_email-marketing
affiliate_provider_facebook
affiliate_provider_google
affiliate_provider_other
first_browser_-unknown-
first_browser_Chrome
first_browser_Firefox
first_browser_IE
first_browser_Mobile Safari
first_browser_Safari
first_browser_other
first_device_Desktop
first_device_Others
first_device_Phone
first_device_Tablet
first_os_Android
first_os_MacOS
first_os_Others
first_os_Windows
first_affiliate_tracked_linked
first_affiliate_tracked_omg
first_affiliate_tracked_other
first_affiliate_tracked_tracked-other
first_affiliate_tracked_untracked
signup_app_Android
signup_app_Moweb
signup_app_Web
signup_app_iOS
signup_method_basic
signup_method_facebook
signup_method_google
signup_flow_0
signup_flow_12
signup_flow_2
signup_flow_23
signup_flow_24
signup_flow_25
signup_flow_3
signup_flow_other
date_account_created_month_1
date_account_created_month_2
date_account_created_month_3
date_account_created_month_4
date_account_created_month_5
date_account_created_month_6
date_account_created_month_7
date_account_created_month_8
date_account_created_month_9
date_account_created_month_10
date_account_created_month_11
date_account_created_month_12
date_first_active_month_1
date_first_active_month_2
date_first_active_month_3
date_first_active_month_4
date_first_active_month_5
date_first_active_month_6
date_first_active_month_7
date_first_active_month_8
date_first_active_month_9
date_first_active_month_10
date_first_active_month_11
date_first_active_month_12
"""
