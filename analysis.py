import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Draw inline
%matplotlib inline

# Set figure aesthetics
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = [15,10]

import warnings
warnings.filterwarnings('ignore')

SEED = 0
TRAIN_PATH = 'raw_data/train_users.csv'
TEST_PATH = 'raw_data/test_users.csv'
SESSIONS_PATH = 'raw_data/sessions.csv'
AGE_GENDER_PATH = 'raw_data/age_gender_bkts.csv'
COUNTRIES_PATH = 'raw_data/countries.csv'
SAMPLE_SUBMISSION_PATH = 'raw_data/sample_submission_NDF.csv'

def display_null_percentage(data):
    df = data.isnull().sum().reset_index().rename(columns = {0: 'Count', 'index': 'Column'})
    df['Frequency'] = df['Count'] / data.shape[0] * 100
    pd.options.display.float_format = '{:.2f}%'.format
    print(df)
    pd.options.display.float_format = None

#Loading data
train_users = pd.read_csv(TRAIN_PATH)
test_users = pd.read_csv(TEST_PATH)
sessions = pd.read_csv(SESSIONS_PATH)
age_gender_bkts = pd.read_csv(AGE_GENDER_PATH)
countries = pd.read_csv(COUNTRIES_PATH)
sample_submission_NDF = pd.read_csv(SAMPLE_SUBMISSION_PATH)

train_users.shape
train_users.info()
train_users.head()
train_users.describe()
train_users.nunique()
display_null_percentage(train_users)
train_users['first_affiliate_tracked'].value_counts()
train_users['gender'].value_counts()
train_users[~pd.isna(train_users['age'])].sort_values(['age'])
train_users[train_users['age'] > 100].sort_values(['age'])
train_users[train_users['age'] > 100]['age'].hist()
train_users[train_users['age'] > 1500].sort_values(['age'])
train_users[train_users['age'] > 1500]['age'].describe()
train_users[train_users['age'] < 20]['age'].hist()
train_users[train_users['age'] < 15].sort_values(['age'])
train_users[(train_users['age'] > 15) & (train_users['age'] < 100)]['age'].hist()
train_users[(train_users['age'] > 15) & (train_users['age'] < 100)]['age'].describe()
train_users[(train_users['age'] > 15) & (train_users['age'] < 100)].shape
train_users[~pd.isna(train_users['age'])].shape


test_users.shape
test_users.info()
test_users.head()
test_users.describe()
test_users.nunique()
display_null_percentage(test_users)
test_users['first_affiliate_tracked'].value_counts()
test_users['gender'].value_counts()
test_users[~pd.isna(test_users['age'])].sort_values(['age'])
test_users[test_users['age'] > 100].sort_values(['age'])
test_users[test_users['age'] > 100]['age'].hist()
test_users[test_users['age'] > 1500].sort_values(['age'])
test_users[test_users['age'] > 1500]['age'].describe()
test_users[test_users['age'] < 20]['age'].hist()
test_users[test_users['age'] < 15].sort_values(['age'])
test_users[(test_users['age'] > 15) & (test_users['age'] < 100)]['age'].hist()
test_users[(test_users['age'] > 15) & (test_users['age'] < 100)]['age'].describe()
test_users[(test_users['age'] > 15) & (test_users['age'] < 100)].shape
test_users[~pd.isna(test_users['age'])].shape

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
# 3. There are outliers for 'age' column (1-5 / >100 / 2014) --> TODO: Identify outliers and compute for those putting birth years.
# 7. Most date columns are not date objects --> TODO: Convert to datetime object
# 4. There is '-unknown-' values for 'gender' column --> TODO: Convert '-unknown-' to na
# 5. Both train and test has less than 3% 'first_affiliate_tracked' missing --> TODO: Drop or impute column 'first_affiliate_tracked'
# 6. Web sessions log for users are not complete for both train data and test data

# Drop column 'date_first_booking'
combine_users = pd.concat([train_users, test_users])
combine_users = combine_users.drop(columns = ['date_first_booking'])

# Convert 'gender' '-unknown-' to na
combine_users['gender'].replace('-unknown-', np.nan, inplace=True)

# Set outliers as NA
combine_users.loc[combine_users['age'] > 95, 'age'] = np.nan
combine_users.loc[combine_users['age'] < 13, 'age'] = np.nan

# Convert to datetime object
combine_users['date_account_created'] = pd.to_datetime(combine_users['date_account_created'])
combine_users['date_first_active'] = pd.to_datetime((combine_users['timestamp_first_active'] // 1000000), format='%Y%m%d')


# Exploratory Data Analysis
def plot_catogory_distribution(data, column_name, title = None):
    data[column_name].value_counts(dropna=False).plot(kind='bar', rot=0)
    plt.title(title)
    plt.show()

categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'country_destination',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method'
]

for categorical_feature in categorical_features:
    plot_catogory_distribution(data = combine_users, column_name = categorical_feature, title = categorical_feature)

plot_catogory_distribution(data = combine_users, column_name = 'gender', title = 'Gender')
plot_catogory_distribution(data = combine_users, column_name = 'affiliate_channel', title = 'Affiliate Channel')

sns.catplot(data=combine_users, x="country_destination", hue="gender", kind="count", edgecolor=".6", height=10, aspect=1.5)
sns.catplot(data=combine_users, x="country_destination", hue="gender", kind="count", edgecolor=".6", height=10, aspect=1.5)
