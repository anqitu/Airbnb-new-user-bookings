import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Draw inline
%matplotlib inline

# Set figure aesthetics
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = [15,8]

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# See more colums
pd.set_option('display.max_columns', 500)

SEED = 2019
TRAIN_PATH = 'raw_data/train_users.csv'
TEST_PATH = 'raw_data/test_users.csv'
SESSIONS_PATH = 'raw_data/sessions.csv'
AGE_GENDER_PATH = 'raw_data/age_gender_bkts.csv'
COUNTRIES_PATH = 'raw_data/countries.csv'
SAMPLE_SUBMISSION_PATH = 'raw_data/sample_submission_NDF.csv'

categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'first_affiliate_tracked', # A user can search before they sign up.
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method',
    'signup_flow' # a key to particular pages - an index for an enumerated list.
]

# Loading data -----------------------------------------------------------------
train_users = pd.read_csv(TRAIN_PATH)
test_users = pd.read_csv(TEST_PATH)
# sessions = pd.read_csv(SESSIONS_PATH)
# age_gender_bkts = pd.read_csv(AGE_GENDER_PATH)
# countries = pd.read_csv(COUNTRIES_PATH)
# sample_submission_NDF = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# Analyze Data -----------------------------------------------------------------
def display_null_percentage(data):
    df = data.isnull().sum().reset_index().rename(columns = {0: 'Count', 'index': 'Column'})
    df['Frequency'] = df['Count'] / data.shape[0] * 100
    pd.options.display.float_format = '{:.2f}%'.format
    print(df)
    pd.options.display.float_format = None

# train_users.shape
# train_users.info()
# train_users.head()
# train_users.describe()
# train_users.nunique()
# display_null_percentage(train_users)
# for categorical_feature in categorical_features:
#     print('-' * 30)
#     print(categorical_feature)
#     print(train_users[categorical_feature].value_counts(dropna=False))
# train_users['first_affiliate_tracked'].value_counts()
# train_users[~pd.isna(train_users['age'])].sort_values(['age'])
# train_users[train_users['age'] > 100].sort_values(['age'])
# train_users[train_users['age'] > 100]['age'].hist()
# train_users[train_users['age'] > 1500].sort_values(['age'])
# train_users[train_users['age'] > 1500]['age'].describe()
# train_users[train_users['age'] < 20]['age'].hist()
# train_users[train_users['age'] < 15].sort_values(['age'])
# train_users[(train_users['age'] > 15) & (train_users['age'] < 100)]['age'].hist()
# train_users[(train_users['age'] > 15) & (train_users['age'] < 100)]['age'].describe()
# train_users[(train_users['age'] > 15) & (train_users['age'] < 100)].shape
# train_users[~pd.isna(train_users['age'])].shape
#
#
# test_users.shape
# test_users.info()
# test_users.head()
# test_users.describe()
# test_users.nunique()
# display_null_percentage(test_users)
# for categorical_feature in categorical_features:
#     print('-' * 30)
#     print(categorical_feature)
#     print(test_users[categorical_feature].value_counts(dropna=False))
# test_users['first_affiliate_tracked'].value_counts()
# test_users['gender'].value_counts()
# test_users['language'].value_counts()
# test_users[~pd.isna(test_users['age'])].sort_values(['age'])
# test_users[test_users['age'] > 100].sort_values(['age'])
# test_users[test_users['age'] > 100]['age'].hist()
# test_users[test_users['age'] > 1500].sort_values(['age'])
# test_users[test_users['age'] > 1500]['age'].describe()
# test_users[test_users['age'] < 20]['age'].hist()
# test_users[test_users['age'] < 15].sort_values(['age'])
# test_users[(test_users['age'] > 15) & (test_users['age'] < 100)]['age'].hist()
# test_users[(test_users['age'] > 15) & (test_users['age'] < 100)]['age'].describe()
# test_users[(test_users['age'] > 15) & (test_users['age'] < 100)].shape
# test_users[~pd.isna(test_users['age'])].shape
#
# sessions.shape
# sessions.info()
# sessions.head()
# sessions.describe()
# sessions.nunique()
# display_null_percentage(sessions)
#
# set(train_users['id']).intersection(set(test_users['id']))
# len(set(train_users['id']).intersection(set(sessions['user_id'])))
# len(set(train_users['id']).difference(set(sessions['user_id'])))
# len(set(test_users['id']).intersection(set(sessions['user_id'])))
# len(set(test_users['id']).difference(set(sessions['user_id'])))
#
# age_gender_bkts.shape
# age_gender_bkts.head()
# age_gender_bkts.describe()
# age_gender_bkts.nunique()
# display_null_percentage(age_gender_bkts)
#
# countries.shape
# countries.head()
# countries.describe()
# countries.nunique()
# display_null_percentage(countries)


# Findings:
# 1. test_user data has all 'date_first_booking' missing --> TODO: Drop column 'date_first_booking'
# 2. Both train and test data has almost half 'age' missing --> TODO: Engineer 'age_missing' column
# 3. There are outliers for 'age' column (1-5 / >100 / 2014) --> TODO: Identify outliers
# 7. Most date columns are not date objects --> TODO: Convert to datetime object
# 4. There is '-unknown-' values for 'gender', 'language', 'first_browser' column --> TODO: Convert '-unknown-' to na
# 5. Both train and test has less than 3% 'first_affiliate_tracked' missing --> TODO: Drop those rows or replace with 'untracked'
# 6. Web sessions log for users are not complete for both train data and test data

# Prepare Data (Clean + Engineer) ----------------------------------------------
train_users['data'] = 'train'
test_users['data'] = 'test'
# Drop column 'date_first_booking'
combine_users = pd.concat([train_users, test_users])
combine_users = combine_users.drop(columns = ['date_first_booking'])

# Convert '-unknown-' to 'NA'
combine_users['gender'].replace('-unknown-', 'NA', inplace=True)
combine_users['language'].replace('-unknown-', 'NA', inplace=True)
combine_users['first_browser'].replace('-unknown-', 'NA', inplace=True)
# for categorical_feature in categorical_features:
#     print('-' * 30)
#     print(categorical_feature)
#     print(combine_users[categorical_feature].value_counts(dropna=False))

# Convert na for  'first_affiliate_tracked' as 'untracked' which is the most common one.
combine_users['first_affiliate_tracked'].replace(np.nan, 'untracked', inplace=True)

# combine_users.replace(np.nan, 'NA', inplace=True)
# display_null_percentage(combine_users)

# Set 'age' outliers as NA
combine_users.loc[combine_users['age'] > 95, 'age'] = np.nan
combine_users.loc[combine_users['age'] < 15, 'age'] = np.nan

# for categorical_feature in categorical_features:
#     print('-' * 30)
#     print(categorical_feature)
#     print(combine_users[categorical_feature].value_counts(dropna=False))

# Convert to datetime object
combine_users['date_account_created'] = pd.to_datetime(combine_users['date_account_created'])
combine_users['date_first_active'] = pd.to_datetime((combine_users['timestamp_first_active'] // 1000000), format='%Y%m%d')
combine_users.drop(columns = ['timestamp_first_active'], inplace=True)

# combine_users['date_account_created']
combine_users['date_account_created_year'] = combine_users['date_account_created'].dt.year
combine_users['date_account_created_month'] = combine_users['date_account_created'].dt.month
combine_users['date_account_created_dow'] = combine_users['date_account_created'].dt.dayofweek
combine_users['date_first_active_year'] = combine_users['date_first_active'].dt.year
combine_users['date_first_active_month'] = combine_users['date_first_active'].dt.month
combine_users['date_first_active_dow'] = combine_users['date_first_active'].dt.dayofweek

# Create a age_missing column
combine_users['age_missing'] = pd.isna(combine_users['age'])
categorical_features.append('age_missing')

# Create bucket for 'age' columns
labels = [str(i) + '-' + str(i+9) for i in range(15, 95, 10)]
combine_users['age_bkt'] = pd.cut(combine_users['age'], bins = range(15, 105, 10), labels = labels)
combine_users['age_bkt'].replace(np.nan, 'NA', inplace = True)
combine_users = combine_users.drop(columns = ['age'])
# combine_users['age_bkt'].value_counts()
categorical_features.append('age_bkt')

# Modelling --------------------------------------------------------------------
# Split data into train and test
from sklearn.model_selection import train_test_split
train = combine_users[combine_users['data'] == 'train']
y = train['country_destination']
x = train.drop(columns = ['data', 'country_destination', 'id', 'date_account_created', 'date_first_active'])

#one hot encoding to prepare for modelling
x_encoded = pd.get_dummies(x, columns = categorical_features)
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.3, stratify = y,random_state=SEED)


# Model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
def score(true, pred):
    scores = {
    'accuracy_score': accuracy_score(true, pred),
    'precision_score': precision_score(true, pred, average='weighted'),
    'recall_score': recall_score(true, pred, average='weighted'),
    'f1_score': f1_score(true, pred, average='weighted'),
    }
    return scores


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
score(y_test, y_pred)
score(y_train, rfc.predict(x_train))
