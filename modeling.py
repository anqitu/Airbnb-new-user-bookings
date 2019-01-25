# Ideas:
# - NDCG formula?
# - Time Series?
# - Fine tuning
# - Stack Meta Data
# - Compare Performance

import os
os.getcwd()
WORKING_DIR = '/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings'
os.listdir(WORKING_DIR)

# import libraries
import time
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

import datetime
def current_time():
    return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Loading data -----------------------------------------------------------------
# train_users = pd.read_csv(TRAIN_PATH)
train_users = pd.read_csv(TRAIN_PATH, nrows=20000)
test_users = pd.read_csv(TEST_PATH)
# sessions = pd.read_csv(SESSIONS_PATH)
# age_gender_bkts = pd.read_csv(AGE_GENDER_PATH)
# countries = pd.read_csv(COUNTRIES_PATH)
# sample_submission_NDF = pd.read_csv(SAMPLE_SUBMISSION_PATH)

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
# users['date_account_created_dayofyear'] = users['date_account_created'].dt.dayofyear
users['date_account_created_day'] = (users['date_account_created'] - users['date_account_created'].min()).dt.days

users['date_first_active_year'] = users['date_first_active'].dt.year
users['date_first_active_month'] = users['date_first_active'].dt.month
# users['date_first_active_dayofyear'] = users['date_first_active'].dt.dayofyear
users['date_first_active_day'] = (users['date_first_active'] - users['date_first_active'].min()).dt.days

# users['date_first_booking_year'] = users['date_first_booking'].dt.year
# users['date_first_booking_month'] = users['date_first_booking'].dt.month
# users['date_first_booking_dayofyear'] = users['date_first_booking'].dt.dayofyear
# users['date_first_booking_day'] = (users['date_first_booking'] - users['date_first_booking'].min()).dt.days

categorical_features = categorical_features + ['date_account_created_year', 'date_account_created_month', 'date_first_active_year', 'date_first_active_month']


# Create a has_age column
# users['has_age'] = ~pd.isna(users['age'])
# categorical_features.append('has_age')

# Create bucket for 'age' columns
labels = [str(i) + '-' + str(i+9) for i in range(15, 95, 10)]
users['age_bkt'] = pd.cut(users['age'], bins = range(15, 105, 10), labels = labels)
users['age_bkt'].replace(np.nan, 'NA', inplace = True)
# users['age_bkt'].value_counts()
categorical_features.append('age_bkt')


# Modelling --------------------------------------------------------------------
# Useful Functions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
def score(true, pred):
    scores = {
    'accuracy_score': accuracy_score(true, pred),
    'precision_score': precision_score(true, pred, average='weighted'),
    'recall_score': recall_score(true, pred, average='weighted'),
    'f1_score': f1_score(true, pred, average='weighted'),
    #  NDCG (Normalized discounted cumulative gain) @k
    }
    return scores

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Split data into train and test
from sklearn.model_selection import train_test_split
users_encoded = pd.get_dummies(users, columns = categorical_features)
train_encoded = users_encoded[users_encoded['data'] == 'train']
test_encoded = users_encoded[users_encoded['data'] == 'test']

y = train_encoded['country_destination']
# label encoding for destination column
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

dropped_columns = ['data', 'country_destination', 'id', 'age', 'date_account_created', 'date_first_active', 'date_first_booking']
x_encoded = train_encoded.drop(columns = dropped_columns)
# (set(x.columns)).difference(set(categorical_features))

# one hot encoding for categorical values
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size=0.3, stratify = y,random_state=SEED)

# cross validation
# from sklearn import model_selection # RFE
# cv_split = model_selection.ShuffleSplit(n_splits = 5, test_size = .3, train_size = .7, random_state = SEED)



# Model Tuning ---------------------------------------------------------------
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.xgb import XGBClassifier


# RFECV: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
from sklearn.feature_selection import RFECV

SCORING_METHOD =

start_total = time.perf_counter()

estimators = {
    'lr': LogisticRegressionCV(random_state = SEED),
    'dtree': DecisionTreeClassifier(random_state = SEED),
    # 'rfc': RandomForestClassifier(random_state = SEED),
    # 'xgb': XGBClassifier(seed = SEED)
    }
estimators_RFECV = {}
for estimatorName, estimator in estimators.items():
    print(current_time() + ': Start training ' + estimatorName)
    start = time.perf_counter()
    rfecv = RFECV(estimator = estimator, step = 1, scoring = 'accuracy', cv = 5)
    rfecv_result = rfecv.fit(x_train, y_train)
    run = time.perf_counter() - start

    print('{} runs for {:.2f} seconds.'.format(estimator.__class__.__name__, run))
    # # clf[1].set_params(**best_param)
    estimators_RFECV[estimatorName] = rfecv_result

run_total = time.perf_counter() - start_total
print('Total running time was {:.2f} minutes.'.format(run_total/60))




from sklearn.model_selection import GridSearchCV

# LogisticRegression
lr = LogisticRegression(random_state = SEED)


clf = dtc
print(clf.__class__.__name__)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print('Trainset Scores')
print(score(y_train, clf.predict(x_train)))
print('Testset Scores')
print(score(y_test, y_pred))

df_confusion = pd.crosstab(pd.Series(label_encoder.inverse_transform(y_test), name='True'), pd.Series(label_encoder.inverse_transform(y_pred), name='Predict'))
plot_confusion_matrix(confusion_matrix(y_test, y_pred), label_encoder.classes_)




dtc = DecisionTreeClassifier(random_state = SEED)





forest_class = RandomForestClassifier(random_state = SEED)

n_estimators = [100, 500]
min_samples_split = [10, 20]

param_grid_forest = {'n_estimators' : n_estimators, 'min_samples_split' : min_samples_split}


rand_search_forest = GridSearchCV(forest_class, param_grid_forest, cv = 4, refit = True,
                                 n_jobs = -1, verbose=2)

rand_search_forest.fit(x_train2, encoded_y_train)

random_estimator = rand_search_forest.best_estimator_

#
# # Feature importance by RandomForest
# # feature_importance = pd.DataFrame(data = {'feature': x_encoded.columns, 'feature_importance': rfc.feature_importances_})
# # feature_importance.sort_values(['feature_importance'])


y_pred_random_estimator = random_estimator.predict_proba(final_train_X)
y_pred = random_estimator.predict_proba(final_test_X)




# for clf in [lr, dtc, rfc, mlpc]:
#     print(clf.__class__.__name__)
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#     print('Trainset Scores')
#     print(score(y_train, clf.predict(x_train)))
#     print('Testset Scores')
#     print(score(y_test, y_pred))

from xgboost.sklearn import XGBClassifier
xgb = XGBClassifier(max_depth= 6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=SEED)

print(current_time() + ': Start training XGBClassifier')
xgb.fit(x_encoded, y_encoded)
print(current_time() + ': Finish training XGBClassifier')

id_test = test_encoded['id']
x_test = test_encoded.drop(columns = dropped_columns)

# preds
y_pred = xgb.predict(x_test)
y_pred = label_encoder.inverse_transform(y_pred)
# y_pred[0]
# np.percentile([item for sublist in y_pred for item in sublist], 50)

#Generate submission
sub = pd.DataFrame(np.column_stack((id_test, y_pred)), columns=['id', 'country'])
sub.to_csv('submission/xgb.csv',index=False)
print(current_time() + ': Saved xgb.csv')


# probs
y_prob = xgb.predict_proba(x_test)

ids = np.repeat(id_test, 5)
destinations = label_encoder.inverse_transform([item for prob in y_prob for item in (np.argsort(y_prob[0])[::-1])[:5]])

# # Taking the 5 classes with highest probabilities
# ids = []  #list of ids
# destinations = []  #list of countries
# for i in range(len(id_test)):
#     idx = id_test[i]
#     ids += [idx] * 5
#     destinations += label_encoder.inverse_transform(np.argsort(y_prob[i])[::-1])[:5].tolist()

sub = pd.DataFrame(np.column_stack((ids, destinations)), columns=['id', 'country'])
sub.to_csv('submission/xgb_5.csv',index=False)
print(current_time() + ': Saved xgb.csv')

# Model oof Stacking -----------------------------------------------------------



# Ignore ------------------------------------------------------------------------
