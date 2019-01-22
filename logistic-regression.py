import numpy as np
import pandas as pd

SEED = 0
TRAIN_PATH = 'data/train_users.csv'
TEST_PATH = 'data/test_users.csv'
SESSIONS_PATH = 'data/sessions.csv'
AGE_GENDER_PATH = 'data/age_gender_bkts.csv'
COUNTRIES_PATH = 'data/countries.csv'
SAMPLE_SUBMISSION_PATH = 'data/sample_submission_NDF.csv'

np.random.seed(SEED)

#Loading data
# sessions = pd.read_csv(SESSIONS_PATH)
train_users = pd.read_csv(TRAIN_PATH)
test_users = pd.read_csv(TEST_PATH)
sessions = pd.read_csv(SESSIONS_PATH)
age_gender_bkts = pd.read_csv(AGE_GENDER_PATH)
countries = pd.read_csv(COUNTRIES_PATH)
sample_submission_NDF = pd.read_csv(SAMPLE_SUBMISSION_PATH)
