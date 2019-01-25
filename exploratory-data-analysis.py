# Ideas
# - Trend of any aspect of ratio
# - Map of each destinations
# - Columns for NDF

import os
os.getcwd()
WORKING_DIR = '/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings'
os.listdir(WORKING_DIR)

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Draw inline
# %matplotlib inline

# Set figure aesthetics
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = [15,8]

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Pandas settings
pd.set_option('display.max_columns', 500)
pd.options.display.float_format = '{:,.5f}'.format

SEED = 2019
TRAIN_PATH = os.path.join(WORKING_DIR, 'raw_data/train_users.csv')
TEST_PATH = os.path.join(WORKING_DIR, 'raw_data/test_users.csv')
SESSIONS_PATH = os.path.join(WORKING_DIR, 'raw_data/sessions.csv')
AGE_GENDER_PATH = os.path.join(WORKING_DIR, 'raw_data/age_gender_bkts.csv')
COUNTRIES_PATH = os.path.join(WORKING_DIR, 'raw_data/countries.csv')
SAMPLE_SUBMISSION_PATH = os.path.join(WORKING_DIR, 'raw_data/sample_submission_NDF.csv')

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
# train_users = pd.read_csv(TRAIN_PATH)
train_users = pd.read_csv(TRAIN_PATH, nrows=20000)
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

def display_category_counts(data, categorical_features):
  for categorical_feature in categorical_features:
    print('-' * 30)
    print(categorical_feature)
    print(data[categorical_feature].value_counts(dropna=False))

# train_users.shape
# train_users.info()
# train_users.head()
# train_users.describe()
# train_users.nunique()
# display_null_percentage(train_users)
# display_category_counts(data = train_users, categorical_features = categorical_features)

# test_users.shape
# test_users.info()
# test_users.head()
# test_users.describe()
# test_users.nunique()
# display_null_percentage(test_users)
# display_category_counts(data = test_users, categorical_features = categorical_features)


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

# For the ages above 150, the users have inserted their year of birth instead of age.
# We can fix this by subtracting the given year from the current year (for this dataset it was 2015) to get the age of the user.
# For ages less than 15, they can be considered as incorrect inputs and can be filtered out.
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



# Findings:
# 1. test_user data has all 'date_first_booking' missing --> Because not sure whether has booked
#    train_user data has 44% 'date_first_booking' missing --> Because no booking yet
# 2. Both train and test data has almost half 'age' missing --> TODO: Engineer 'age_missing' column
# 3. There are outliers for 'age' column (1-5 / >100 / 2014) --> TODO: Identify outliers
# 7. Most date columns are not date objects --> TODO: Convert to datetime object
# 4. There is '-unknown-' values for 'gender', 'language', 'first_browser' column --> TODO: Convert '-unknown-' to na
# 5. Both train and test has less than 3% 'first_affiliate_tracked' missing --> TODO: Drop those rows or replace with 'untracked'
# 6. Web sessions log for users are not complete for both train data and test data

# Prepare Data (Clean + Engineer) ----------------------------------------------
train_users['data'] = 'train'
test_users['data'] = 'test'

users = pd.concat([train_users, test_users])

# Convert '-unknown-' to 'NA'
users['gender'].replace('-unknown-', 'NA', inplace=True)
users['language'].replace('-unknown-', 'NA', inplace=True)
users['first_browser'].replace('-unknown-', 'NA', inplace=True)
# display_category_counts(data = users, categorical_features = categorical_features)

# Convert na for  'first_affiliate_tracked' as 'untracked' which is the most common one.
users['first_affiliate_tracked'].replace(np.nan, 'untracked', inplace=True)

# Correct those ages entered as year
users.loc[users['age']>1500,'age'] = 2015 - users.loc[users['age']>1500,'age']
# users.age.describe()

# Set 'age' outliers as NA
users.loc[users['age'] > 95, 'age'] = np.nan
users.loc[users['age'] < 15, 'age'] = np.nan
# users.age.describe()

# Convert to datetime object
users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_active'] = pd.to_datetime((users['timestamp_first_active'] // 1000000), format='%Y%m%d')
users.drop(columns = ['timestamp_first_active'], inplace=True)

# Extract year, month and day from datetime
users['date_account_created_year'] = users['date_account_created'].dt.year
users['date_account_created_month'] = users['date_account_created'].dt.month
users['date_account_created_dayofyear'] = users['date_account_created'].dt.dayofyear
users['date_account_created_day'] = (users['date_account_created'] - users['date_account_created'].min()).dt.days

users['date_first_active_year'] = users['date_first_active'].dt.year
users['date_first_active_month'] = users['date_first_active'].dt.month
users['date_first_active_dayofyear'] = users['date_first_active'].dt.dayofyear
users['date_first_active_day'] = (users['date_first_active'] - users['date_first_active'].min()).dt.days

users['date_first_booking_year'] = users['date_first_booking'].dt.year
users['date_first_booking_month'] = users['date_first_booking'].dt.month
users['date_first_booking_dayofyear'] = users['date_first_booking'].dt.dayofyear
users['date_first_booking_day'] = (users['date_first_booking'] - users['date_first_booking'].min()).dt.days


# Create a has_age column
users['has_age'] = ~pd.isna(users['age'])
categorical_features.append('has_age')

# Create bucket for 'age' columns
labels = [str(i) + '-' + str(i+9) for i in range(15, 95, 10)]
users['age_bkt'] = pd.cut(users['age'], bins = range(15, 105, 10), labels = labels)
users['age_bkt'].replace(np.nan, 'NA', inplace = True)
# users['age_bkt'].value_counts()
categorical_features.append('age_bkt')



# Plot -------------------------------------------------------------------------
def plot_catogory_distribution(data, column_name, title = None):
    data[column_name].value_counts(dropna=False).plot(kind='bar', rot=0, color = 'c')
    plt.ylabel("No. of users")
    plt.title(title)
    plt.show()

def plot_continuous_distribution_as_bar(data, column_name, title = None, bins = None):
    sns.distplot(data[column_name].dropna(), bins = bins)
    plt.title(title)
    plt.show()

def plot_continuous_distribution_as_box(data, continuous_column, category_column, title = None):
    sns.boxplot(y = continuous_column , x = category_column, data = data, color = 'c')
    plt.title(title)
    plt.show()

def plot_category_stack(data, column_name):
    pivot_df = data.groupby(['country_destination', column_name])['date_account_created'].count().reset_index().pivot(index='country_destination', columns=column_name, values='date_account_created')
    pivot_df.plot.bar(stacked=True)

def plot_category_bar(data, column_name):
    sns.catplot(data = data, x = "country_destination", hue = column_name, kind = "count", height = 8, aspect = 1.5)

def plot_train_test_diff(data, column_name):
    pivot_df = data.groupby([column_name, 'data'])['date_account_created'].count().reset_index().pivot(index=column_name, columns='data', values='date_account_created')
    pivot_df.plot.bar(stacked=True)

train = users[users['data'] == 'train'].drop(columns = ['data'])
test = users[users['data'] == 'test'].drop(columns = ['data'])
train['has_destination'] = (train['country_destination'] != 'NDF')
train_has_destination = train[train['country_destination'] !='NDF']
# display_null_percentage(train_has_destination)


# Distribution of user's selection of country
plot_catogory_distribution(train, 'country_destination', title = "Distribution of destination countries")
plot_catogory_distribution(train_has_destination, 'country_destination', title = "Distribution of destination countries")
# Finding: Almost all the countries have a similar median age. Only users tavelling to Spain and Portugal are slightly younger.

# Distribution of user's age
plot_continuous_distribution_as_bar(data = users, column_name = 'age', title = "Age Distribution of users", bins = 16)
# Finding: Most of our users have never booked.
# US is the most populor choice as the dataset is from the US users who would likely to prefer travel to nearer place within their home country

# Distribution of user's age across the destination countries
plot_continuous_distribution_as_box(data = users, continuous_column = 'age', category_column = 'country_destination', title = 'Age Distribution across the destinations')
# Finding: Almost all the countries have a similar median age. Only users tavelling to Spain and Portugal are slightly younger.
# Users of age 80 and above mostly choose US as their destination.
# The reason might be the US user data i.e. as all the users are from US, older people in US prefer not to travel outside their home country.



# for categorical_feature in categorical_features:
#     plot_catogory_distribution(data = train, column_name = categorical_feature, title = categorical_feature)

# for categorical_feature in categorical_features:
#     plot_category_stack(data = users, column_name = categorical_feature)

# for categorical_feature in ['gender', 'age_missing']:
#     plot_category_bar(data = users, column_name = categorical_feature)

# for categorical_feature in categorical_features:
#     plot_train_test_diff(data = users, column_name = categorical_feature)

# users.date_account_created.value_counts().plot(kind='line', linewidth=1)
# users.date_first_active.value_counts().plot(kind='line', linewidth=1)


# Datatime Trend ---------------------------------------------------------------
# users_2013 = users[users['date_first_active'] > pd.to_datetime(20130101, format='%Y%m%d')]
# users_2013 = users_2013[users_2013['date_first_active'] < pd.to_datetime(20140101, format='%Y%m%d')]
# users_2013.date_first_active.value_counts().plot(kind='line', linewidth=2, color='#FD5C64')
# plt.show()
#
# weekdays = []
# for date in users.date_account_created:
#     weekdays.append(date.weekday())
# weekdays = pd.Series(weekdays)
# sns.barplot(x = weekdays.value_counts().index, y=weekdays.value_counts().values, order=range(0,7))
# plt.xlabel('Week Day')
# sns.despine()
#
#
# users = users
# date = pd.to_datetime(20140101, format='%Y%m%d')
#
# before = sum(users.loc[users['date_first_active'] < date, 'country_destination'].value_counts())
# after = sum(users.loc[users['date_first_active'] > date, 'country_destination'].value_counts())
# before_destinations = users.loc[users['date_first_active'] < date,
#                                 'country_destination'].value_counts() / before * 100
# after_destinations = users.loc[users['date_first_active'] > date,
#                                'country_destination'].value_counts() / after * 100
# before_destinations.plot(kind='bar', width=5, color='#63EA55', position=0, label='Before 2014', rot=0)
# after_destinations.plot(kind='bar', width=5, color='#4DD3C9', position=1, label='After 2014', rot=0)
#
# plt.legend()
# plt.xlabel('Destination Country')
# plt.ylabel('Percentage')
#
# sns.despine()
# plt.show()
