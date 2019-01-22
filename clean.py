import numpy as np
import pandas as pd

SEED = 0
TRAIN_PATH = 'raw_data/train_users.csv'
TEST_PATH = 'raw_data/test_users.csv'
SESSIONS_PATH = 'raw_data/sessions.csv'
AGE_GENDER_PATH = 'raw_data/age_gender_bkts.csv'
COUNTRIES_PATH = 'raw_data/countries.csv'
SAMPLE_SUBMISSION_PATH = 'raw_data/sample_submission_NDF.csv'

np.random.seed(SEED)

#Loading data
train_users = pd.read_csv(TRAIN_PATH)
test_users = pd.read_csv(TEST_PATH)
sessions = pd.read_csv(SESSIONS_PATH)

train_users = train_users.drop(columns = ['date_first_booking'])





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
    users[categorical_feature] = users[categorical_feature].astype('category')
