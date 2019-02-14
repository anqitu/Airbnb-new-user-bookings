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

MIN_TO_OTHERS = True
if MIN_TO_OTHERS:
    USERS_PATH = USERS_MIN_TO_OTHERS_PATH

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

continuous_features = ['date_account_created_to_next_holiday', 'date_first_active_to_next_holiday']

set(users.columns).difference(set(categorical_features)).difference(set(continuous_features))
mapped_features = [feature for feature in users.columns if feature.endswith('_min_to_other')]

if MIN_TO_OTHERS:
    for feature in mapped_features:
        users[feature.replace('_min_to_other', '')] = users[feature]
        categorical_features.remove(feature)
else:
    for feature in mapped_features:
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
users.sum().min()

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

train.to_csv(TRAIN_RAW_PATH, index = False)
val.to_csv(VAL_RAW_PATH, index = False)
test.to_csv(TEST_PATH, index = False)

""" SMOTE """
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=SEED)

# for train
percentage = get_percentage(train, target)
le = load_label_encoder('label_encoder_' + target)
percentage[target] = percentage[target].map(dict(zip(range(0, 11), le.classes_)))
print('Before OverSampling, the shape of train: {}'.format(train.shape))
print("Distribution of country destination: ")
print(percentage)
percentage.to_csv('data/country_distribution_train.csv', index = False)

X_train_smote, y_train_smote = smt.fit_sample(train.drop(columns = target).values, train[target].ravel())
train_smote = np.concatenate([X_train_smote, y_train_smote.reshape(-1, 1)], axis = 1)
train_smote = pd.DataFrame(train_smote)
columns = list(train.columns)
columns.remove(target)
columns.append(target)
train_smote.columns = columns
train_smote.to_csv(TRAIN_SMOTE_PATH, index = False)

percentage = get_percentage(train_smote, target)
le = load_label_encoder('label_encoder_' + target)
percentage[target] = percentage[target].map(dict(zip(range(0, 11), le.classes_)))
print('After OverSampling, the shape of train_smote: {}'.format(train_smote.shape))
print("Distribution of country destination: ")
print(percentage)
percentage.to_csv('data/country_distribution_train_smote.csv', index = False)

# for val
percentage = get_percentage(val, target)
le = load_label_encoder('label_encoder_' + target)
percentage[target] = percentage[target].map(dict(zip(range(0, 11), le.classes_)))
print('Before OverSampling, the shape of val: {}'.format(val.shape))
print("Distribution of country destination: ")
print(percentage)
percentage.to_csv('data/country_distribution_val.csv', index = False)

X_val_smote, y_val_smote = smt.fit_sample(val.drop(columns = target), val[target])
val_smote = np.concatenate([X_val_smote, y_val_smote.reshape(-1, 1)], axis = 1)
val_smote = pd.DataFrame(val_smote)
columns = list(val.columns)
columns.remove(target)
columns.append(target)
val_smote.columns = columns
val_smote.to_csv(VAL_SMOTE_PATH, index = False)

percentage = get_percentage(val_smote, target)
le = load_label_encoder('label_encoder_' + target)
percentage[target] = percentage[target].map(dict(zip(range(0, 11), le.classes_)))
print('After OverSampling, the shape of val_smote: {}'.format(val_smote.shape))
print("Distribution of country destination: ")
print(percentage)
percentage.to_csv('data/country_distribution_val_smote.csv', index = False)

# for test
percentage = get_percentage(test, target)
le = load_label_encoder('label_encoder_' + target)
percentage[target] = percentage[target].map(dict(zip(range(0, 11), le.classes_)))
print('The shape of test: {}'.format(test.shape))
print("Distribution of country destination: ")
print(percentage)
percentage.to_csv('data/country_distribution_test.csv', index = False)

for col in train.columns:
    print(col)

val.sum()
"""
Before OverSampling, the shape of train: (53344, 187)
Distribution of country destination:
   country_destination  Count         %
0                   US  37425  0.701578
1                other   6056  0.113527
2                   FR   3013  0.056482
3                   IT   1701  0.031887
4                   GB   1394  0.026132
5                   ES   1349  0.025289
6                   CA    857  0.016066
7                   DE    637  0.011941
8                   NL    458  0.008586
9                   AU    323  0.006055
10                  PT    131  0.002456
After OverSampling, the shape of train_smote: (411049, 187)
Distribution of country destination:
   country_destination  Count         %
0                   PT  37425  0.091048
1                   AU  37424  0.091045
2                   NL  37417  0.091028
3                   DE  37415  0.091023
4                   CA  37403  0.090994
5                   ES  37388  0.090958
6                   GB  37378  0.090933
7                   IT  37369  0.090911
8                   FR  37330  0.090816
9                   US  37269  0.090668
10               other  37231  0.090576
Before OverSampling, the shape of val: (17782, 187)
Distribution of country destination:
   country_destination  Count         %
0                   US  12476  0.701608
1                other   2019  0.113542
2                   FR   1005  0.056518
3                   IT    567  0.031886
4                   GB    465  0.026150
5                   ES    450  0.025306
6                   CA    285  0.016027
7                   DE    212  0.011922
8                   NL    152  0.008548
9                   AU    108  0.006074
10                  PT     43  0.002418
After OverSampling, the shape of val_smote: (137144, 187)
Distribution of country destination:
   country_destination  Count         %
0                   NL  12476  0.090970
1                   DE  12476  0.090970
2                   AU  12476  0.090970
3                   PT  12475  0.090963
4                   GB  12473  0.090948
5                   IT  12473  0.090948
6                   CA  12473  0.090948
7                   ES  12470  0.090926
8                   FR  12461  0.090861
9                   US  12446  0.090751
10               other  12445  0.090744
The shape of test: (17782, 187)
Distribution of country destination:
   country_destination  Count         %
0                   US  12475  0.701552
1                other   2019  0.113542
2                   FR   1005  0.056518
3                   IT    567  0.031886
4                   GB    465  0.026150
5                   ES    450  0.025306
6                   CA    286  0.016084
7                   DE    212  0.011922
8                   NL    152  0.008548
9                   AU    108  0.006074
10                  PT     43  0.002418
"""
