"""
 prepare datas for modelling and Association Rules:
- Cleaning for devices
- Drop irrelevant columns
- Label encoding for target variable 'country destination'
- One hot encoding for categorical variables
- Split data into train and test
- Standard Normalization for continuous variables
"""
import pandas as pd
import numpy as np
from util import *
pd.options.display.float_format = '{:,.5f}'.format
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

users = pd.read_csv(USERS_PATH, keep_default_na=False, na_values=[''])
# users = users.sample(5000)

target = 'country_destination'

categorical_features = ['gender',
                        'language', 'language_min_to_other',
                        'affiliate_channel',
                        'affiliate_provider', 'affiliate_provider_min_to_other',
                        'first_browser', 'first_browser_min_to_other',
                        'first_device', 'first_os',
                        'first_affiliate_tracked', 'first_affiliate_tracked_min_to_other', # A user can search before they sign up.
                        'signup_app', 'signup_method',
                        'signup_flow', 'signup_flow_min_to_other',# a key to particular pages - an index for an enumerated list.
                        'age_bkt', 'age_bkt_min_to_other',
                        'date_account_created_month','date_first_active_month',
                        'date_account_created_dayofweek', 'date_first_active_dayofweek']

continuous_features = ['date_account_created_days_to_next_holiday', 'date_first_active_days_to_next_holiday']
users[categorical_features + continuous_features].shape
set(users.columns).difference(set(categorical_features)).difference(set(continuous_features))

mapped_features = [feature for feature in users.columns if feature.endswith('_min_to_other')]
for feature in mapped_features:
    users[feature.replace('_min_to_other', '')] = users[feature]
    categorical_features.remove(feature)

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
train_all, test = train_test_split(users, test_size=0.2, stratify = users[target], random_state = SEED)
train, val = train_test_split(train_all, test_size=0.25, stratify = train_all[target], random_state = SEED)
train.shape
test.shape
val.shape

""" standard normalization for continous values """
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train[continuous_features] = scaler.fit_transform(train[continuous_features])
val[continuous_features] = scaler.transform(val[continuous_features])
test[continuous_features] = scaler.transform(test[continuous_features])

train.to_csv(TRAIN_PATH, index = False)
val.to_csv(VAL_PATH, index = False)
test.to_csv(TEST_PATH, index = False)


# for train
percentage = get_percentage(train, target)
le = load_label_encoder('label_encoder_' + target)
percentage[target] = percentage[target].map(dict(zip(range(0, 11), le.classes_)))
print('Shape of train: {}'.format(train.shape))
print("Distribution of country destination: ")
print(percentage)
percentage.to_csv('data/country_distribution_train.csv', index = False)

# for val
percentage = get_percentage(val, target)
le = load_label_encoder('label_encoder_' + target)
percentage[target] = percentage[target].map(dict(zip(range(0, 11), le.classes_)))
print('Shape of val: {}'.format(val.shape))
print("Distribution of country destination: ")
print(percentage)
percentage.to_csv('data/country_distribution_val.csv', index = False)

# for test
percentage = get_percentage(test, target)
le = load_label_encoder('label_encoder_' + target)
percentage[target] = percentage[target].map(dict(zip(range(0, 11), le.classes_)))
print('Shape of test: {}'.format(test.shape))
print("Distribution of country destination: ")
print(percentage)
percentage.to_csv('data/country_distribution_test.csv', index = False)

for col in train.columns:
    print(col)

# val.sum().min()
"""
Shape of train: (53344, 134)
Distribution of country destination:
   country_destination  Count       %
0                   US  37425 0.70158
1                other   6056 0.11353
2                   FR   3013 0.05648
3                   IT   1701 0.03189
4                   GB   1394 0.02613
5                   ES   1349 0.02529
6                   CA    857 0.01607
7                   DE    637 0.01194
8                   NL    458 0.00859
9                   AU    323 0.00606
10                  PT    131 0.00246
Shape of val: (17782, 134)
Distribution of country destination:
   country_destination  Count       %
0                   US  12476 0.70161
1                other   2019 0.11354
2                   FR   1005 0.05652
3                   IT    567 0.03189
4                   GB    465 0.02615
5                   ES    450 0.02531
6                   CA    285 0.01603
7                   DE    212 0.01192
8                   NL    152 0.00855
9                   AU    108 0.00607
10                  PT     43 0.00242
Shape of test: (17782, 134)
Distribution of country destination:
   country_destination  Count       %
0                   US  12475 0.70155
1                other   2019 0.11354
2                   FR   1005 0.05652
3                   IT    567 0.03189
4                   GB    465 0.02615
5                   ES    450 0.02531
6                   CA    286 0.01608
7                   DE    212 0.01192
8                   NL    152 0.00855
9                   AU    108 0.00607
10                  PT     43 0.00242
date_account_created_days_to_next_holiday
date_first_active_days_to_next_holiday
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
affiliate_channel_api
affiliate_channel_content
affiliate_channel_direct
affiliate_channel_other
affiliate_channel_remarketing
affiliate_channel_sem-brand
affiliate_channel_sem-non-brand
affiliate_channel_seo
affiliate_provider_bing
affiliate_provider_craigslist
affiliate_provider_direct
affiliate_provider_facebook
affiliate_provider_google
affiliate_provider_other
affiliate_provider_padmapper
affiliate_provider_vast
affiliate_provider_yahoo
first_browser_-unknown-
first_browser_Android Browser
first_browser_Chrome
first_browser_Chrome Mobile
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
first_affiliate_tracked_product
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
signup_flow_8
signup_flow_12
signup_flow_23
signup_flow_24
signup_flow_25
signup_flow_0
signup_flow_1
signup_flow_12
signup_flow_2
signup_flow_23
signup_flow_24
signup_flow_25
signup_flow_3
signup_flow_6
signup_flow_8
signup_flow_other
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
age_bkt_65-69
age_bkt_70-74
age_bkt_75-79
age_bkt_80+
age_bkt_NA
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
date_account_created_dayofweek_Friday
date_account_created_dayofweek_Monday
date_account_created_dayofweek_Saturday
date_account_created_dayofweek_Sunday
date_account_created_dayofweek_Thursday
date_account_created_dayofweek_Tuesday
date_account_created_dayofweek_Wednesday
date_first_active_dayofweek_Friday
date_first_active_dayofweek_Monday
date_first_active_dayofweek_Saturday
date_first_active_dayofweek_Sunday
date_first_active_dayofweek_Thursday
date_first_active_dayofweek_Tuesday
date_first_active_dayofweek_Wednesday
"""
