# Ideas:
# - NDCG formula?
# - Time Series?
# - Feature Selection & Fine tuning
# - Stack Meta Data
# - Compare Performance

import os
os.getcwd()
WORKING_DIR = '/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings'
# WORKING_DIR = '/content'
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
USERS_PATH = os.path.join(WORKING_DIR, 'raw_data/users.csv') if WORKING_DIR == '/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings' else os.path.join(WORKING_DIR, 'users.csv')
SESSIONS_PATH = os.path.join(WORKING_DIR, 'raw_data/sessions.csv')
COUNTRIES_PATH = os.path.join(WORKING_DIR, 'raw_data/countries.csv')

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
users = pd.read_csv(USERS_PATH)
users = users.sample(5000)
# sessions = pd.read_csv(SESSIONS_PATH)
# countries = pd.read_csv(COUNTRIES_PATH)

# Prepare Data (Clean + Engineer) ----------------------------------------------
# Convert '-unknown-' to 'NA'
# users['gender'].replace('-unknown-', 'NA', inplace=True)
# users['language'].replace('-unknown-', 'NA', inplace=True)
# users['first_browser'].replace('-unknown-', 'NA', inplace=True)
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

# # Convert to datetime object
# users['date_account_created'] = pd.to_datetime(users['date_account_created'])
# users['date_first_active'] = pd.to_datetime((users['timestamp_first_active'] // 1000000), format='%Y%m%d')
# users.drop(columns = ['timestamp_first_active'], inplace=True)
#
# # Extract year, month and day from datetime
# users['date_account_created_year'] = users['date_account_created'].dt.year
# users['date_account_created_month'] = users['date_account_created'].dt.month
# # users['date_account_created_dayofyear'] = users['date_account_created'].dt.dayofyear
# users['date_account_created_day_count'] = (users['date_account_created'] - users['date_account_created'].min()).dt.days
# min_year = users['date_account_created'].min().year
# min_month = users['date_account_created'].min().month
# users['date_account_created' + '_month_count'] = users['date_account_created' + '_month'] + (users['date_account_created' + '_year'] - min_year) * 12 - min_month + 1
#
#
# users['date_first_active_year'] = users['date_first_active'].dt.year
# users['date_first_active_month'] = users['date_first_active'].dt.month
# # users['date_first_active_dayofyear'] = users['date_first_active'].dt.dayofyear
# users['date_first_active_day_count'] = (users['date_first_active'] - users['date_first_active'].min()).dt.days
# min_year = users['date_first_active'].min().year
# min_month = users['date_first_active'].min().month
# users['date_first_active' + '_month_count'] = users['date_first_active' + '_month'] + (users['date_first_active' + '_year'] - min_year) * 12 - min_month + 1

# categorical_features = categorical_features + ['date_account_created_year', 'date_account_created_month', 'date_first_active_year', 'date_first_active_month']


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
from sklearn.metrics import accuracy_score, confusion_matrix

import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, path = 'test'):
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
    plt.savefig(path)
    plt.close()

def get_matrix(y_test, y_test_pred, y_train, y_train_pred, estimator_name, label_encoder, transformed = ''):
    title = 'Confusion Matrix for ' + estimator_name + ' Test'
    if transformed != '':
        title += ' (Transformed)'
    df_confusion = pd.crosstab(pd.Series(label_encoder.inverse_transform(y_test), name='True'), pd.Series(label_encoder.inverse_transform(y_test_pred), name='Predict'))
    path = title.replace(' ', '_').replace('(', '').replace(')', '')
    df_confusion.to_csv(path + '.csv')
    plot_confusion_matrix(confusion_matrix(y_test, y_test_pred), label_encoder.classes_, path = path, title = title)

    title = 'Confusion Matrix for ' + estimator_name  + ' Train'
    if transformed != '':
        title += ' (Transformed)'
    df_confusion = pd.crosstab(pd.Series(label_encoder.inverse_transform(y_train), name='True'), pd.Series(label_encoder.inverse_transform(y_train_pred), name='Predict'))
    path = title.replace(' ', '_').replace('(', '').replace(')', '')
    df_confusion.to_csv(path + '.csv')
    plot_confusion_matrix(confusion_matrix(y_train, y_train_pred), label_encoder.classes_, path = path, title = title)



def plot_estimator_no_feature_vs_accuracy_score(rfecv_result, estimator):
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv_result.grid_scores_) + 1), rfecv_result.grid_scores_)
    plt.ylim([0.56,0.64])
    title = 'Number of Features vs Accuracy Score for ' + estimator.__class__.__name__
    plt.title(title)
    plt.savefig(title.replace(' ', '_'))
    plt.close()

def save_label_encoder(label_encoder):
    np.save('label_encoder.npy', label_encoder.classes_)

def load_label_encoder():
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('label_encoder.npy')
    return label_encoder

def train_clf(clf, x_train, x_test, x_val, transformed = ''):
    clf_name = clf.__class__.__name__

    print(current_time() + ': Start training ' + clf_name)
    start_total = time.perf_counter()
    clf.fit(x_train, y_train)
    run_total = time.perf_counter() - start_total
    print(current_time() + ': Finish training ' + clf_name)
    run_times_df[clf_name + transformed] = [run_total]

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    y_val_pred = clf.predict(x_val)
    train_result_df['y_train_pred_' + clf_name + transformed] = y_train_pred
    test_result_df['y_test_pred_' + clf_name + transformed] = y_test_pred
    val_result_df['y_val_pred_' + clf_name + transformed] = y_val_pred

    accuracy_scores = []
    accuracy_scores.append(accuracy_score(y_train_pred, y_train))
    accuracy_scores.append(accuracy_score(y_test_pred, y_test))
    pd.DataFrame(clf.predict_proba(x_train)).to_csv("Prob_Train_{}{}.csv".format(clf_name, transformed), index = False)
    pd.DataFrame(clf.predict_proba(x_test)).to_csv("Prob_Test_{}{}.csv".format(clf_name, transformed), index = False)
    pd.DataFrame(clf.predict_proba(x_val)).to_csv("Prob_Val_{}{}.csv".format(clf_name, transformed), index = False)

    score_df = pd.DataFrame(data = {'Prediction': ['Train Predction', 'Test Prediction'], 'Accuracy Score': accuracy_scores})
    score_df.to_csv('Scores_for_{}{}.csv'.format(clf_name, transformed), index = False)
    get_matrix(y_test = y_test, y_test_pred = y_test_pred,
                y_train = y_train, y_train_pred = y_train_pred,
                estimator_name = clf_name, label_encoder = label_encoder, transformed = transformed)

    print(current_time() + ': Finish getting confusion matrix for ' + clf_name)

    save_model(clf, clf.__class__.__name__ + transformed)

    return clf

def train_clf_transformed(clf):
    x_train_transformed = estimators_RFECV[clf.__class__.__name__].transform(x_train)
    x_test_transformed = estimators_RFECV[clf.__class__.__name__].transform(x_test)
    x_val_transformed = estimators_RFECV[clf.__class__.__name__].transform(x_val)
    train_clf(clf, x_train_transformed, x_test_transformed, x_val_transformed, transformed = '_transformed')

    return clf

def get_feature_importance_by_tree(clf):
    feature_importance = pd.DataFrame(data = {'feature': x_train.columns, 'feature_importance': clf.feature_importances_})
    feature_importance.sort_values(['feature_importance'], ascending = False)
    feature_importance.to_csv('Feature_Importance_{}.csv'.format(clf.__class__.__name__), index = False)

    return feature_importance

import pickle
def save_model(model, filename):
    pickle.dump(lr, open(filename + '.sav', 'wb'))

def load_model(filename):
    return pickle.load(open(filename + '.sav', 'rb'))


# label encoding for destination column
from sklearn.preprocessing import LabelEncoder
y = users['country_destination']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
save_label_encoder(label_encoder)

x_users = users[['gender', 'age_bkt', 'language', 'signup_method', 'signup_app',
                # 'signup_flow', 'affiliate_channel', 'affiliate_provider', 'first_device_type', 'first_browser',
                # 'date_account_created_year', 'date_first_active_year',
                # 'date_account_created_month', 'date_first_active_month',
                # 'date_account_created_day_count', 'date_first_active_day_count',
                # 'date_account_created_month_count','date_first_active_month_count'
                ]]

print(users.columns)

# one hot encoding for categorical columns
from sklearn.model_selection import train_test_split
x_users_encoded = pd.get_dummies(x_users, columns =
                ['gender', 'age_bkt', 'language', 'signup_method', 'signup_app',
                # 'signup_flow', 'affiliate_channel', 'affiliate_provider', 'first_device_type', 'first_browser',
                # 'date_account_created_year', 'date_first_active_year',
                # 'date_account_created_month', 'date_first_active_month',
                # 'date_account_created_day_count', 'date_first_active_day_count',
                # 'date_account_created_month_count','date_first_active_month_count'
                ])

# Split data into train, test and val
x_train, x_val, y_train, y_val = train_test_split(x_users_encoded, y_encoded, test_size=0.3, stratify = y_encoded,random_state=SEED)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, stratify = y_train,random_state=SEED)

print('Train Shape: ' + str(x_train.shape))
print('Test Shape: ' + str(x_test.shape))
print('Validation Shape: ' + str(x_val.shape))

train_result_df = pd.DataFrame(data = {'y_train': y_train})
test_result_df = pd.DataFrame(data = {'y_test': y_test})
val_result_df = pd.DataFrame(data = {'y_test': y_val})
run_times_df = pd.DataFrame()

# Model Tuning ---------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.feature_selection import RFECV, RFE

# RFECV: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

start_total = time.perf_counter()

estimators = {clf.__class__.__name__: clf for clf in [LogisticRegression(random_state = SEED),
                                                    DecisionTreeClassifier(random_state = SEED),
                                                    RandomForestClassifier(random_state = SEED),
                                                    XGBClassifier(seed = SEED)
                                                    ]}
estimators_RFECV = {}
for estimatorName, estimator in estimators.items():
    print(current_time() + ': Start training ' + estimatorName)
    start = time.perf_counter()
    rfecv = RFECV(estimator = estimator, step = 1, scoring = 'accuracy', cv = 3)
    rfecv_result = rfecv.fit(x_train, y_train)
    run = time.perf_counter() - start

    print('{} runs for {:.2f} seconds.'.format(estimator.__class__.__name__, run))
    estimators_RFECV[estimatorName] = rfecv_result

    rfe_result_rank = pd.DataFrame(data = {'Ranking': rfecv_result.ranking_, 'Column': x_train.columns}).sort_values('Ranking')
    rfe_result_rank.to_csv('RFECV_Ranking_for_{}.csv'.format(estimator.__class__.__name__), index = False)

    plot_estimator_no_feature_vs_accuracy_score(rfecv_result, estimator)

run_total = time.perf_counter() - start_total
print('Total running time was {:.2f} minutes.'.format(run_total/60))

# LogisticRegression
lr = LogisticRegression(random_state = SEED)
lr = train_clf(lr, x_train, x_test, x_val)

lr_t = LogisticRegression(random_state = SEED)
lr_t = train_clf_transformed(lr_t)

# DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
tree = DecisionTreeClassifier(random_state = SEED)
min_samples_split = [10, 30, 50, 100, 200, 300]
param_grid_tree = {'min_samples_split' : min_samples_split}
grid_search_tree = GridSearchCV(tree, param_grid_tree, cv = 5, refit = True,
                                 n_jobs = -1, verbose=2)
grid_search_tree.fit(x_train, y_train)
for score in grid_search_tree.grid_scores_:
    print(score)
print(grid_search_tree.best_params_)
tree = grid_search_tree.best_estimator_
print(tree.get_params)
get_feature_importance_by_tree(tree)

tree = DecisionTreeClassifier(random_state = SEED, **grid_search_tree.best_params_)
tree = train_clf(tree, x_train, x_test, x_val)

tree_t = DecisionTreeClassifier(random_state = SEED, **grid_search_tree.best_params_)
tree_t = train_clf_transformed(tree_t)


# RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier(random_state = SEED)

n_estimators = [100, 300]
min_samples_split = [30, 50, 100, 200]
param_grid_rfc = {'min_samples_split' : min_samples_split, 'n_estimators': n_estimators}
grid_search_rfc = GridSearchCV(rfc, param_grid_rfc, cv = 5, refit = True,
                                 n_jobs = -1, verbose=2)
grid_search_rfc.fit(x_train, y_train)
for score in grid_search_rfc.grid_scores_:
    print(score)
print(grid_search_rfc.best_params_)
rfc = grid_search_rfc.best_estimator_
print(rfc.get_params)
get_feature_importance_by_tree(rfc)

rfc = RandomForestClassifier(random_state = SEED, **grid_search_rfc.best_params_)
rfc = train_clf(rfc, x_train, x_test, x_val)

rfc_t = RandomForestClassifier(random_state = SEED, **grid_search_rfc.best_params_)
rfc_t = train_clf_transformed(rfc_t)

# XGBClassifier
xgb = XGBClassifier(max_depth= 6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=SEED)
xgb_t = XGBClassifier(max_depth= 6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=SEED)

xgb = train_clf(xgb, x_train, x_test, x_val)
xgb_t = train_clf_transformed(xgb_t)

train_result_df.to_csv('Predicts_Train.csv', index = False)
test_result_df.to_csv('Predicts_Test.csv', index = False)
val_result_df.to_csv('Predicts_Val.csv', index = False)
run_times_df.to_csv('Run_Time.csv', index = False)


# Model oof Stacking -----------------------------------------------------------
test_prob_files = [file for file in os.listdir(WORKING_DIR) if file.startswith('Prob_Test_') and 'transformed' not in file]
prob_dfs = []
for prob_file in test_prob_files:
    df = pd.read_csv(prob_file)
    model_name = prob_file.replace('Prob_Test_', '').replace('.csv', '')
    df.columns = [model_name + '_' + str(i) for i in range(12)]
    prob_dfs.append(df)

x_test_meta = pd.concat(prob_dfs, axis = 1)
y_test_meta = y_test

xgb_stacking = XGBClassifier(max_depth= 6, learning_rate=0.3, n_estimators=25,
                            objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=SEED)
xgb_stacking.fit(x_test_meta, y_test_meta)

y_test_pred_meta = xgb_stacking.predict(x_test_meta)
print(accuracy_score(y_test_meta, y_test_pred_meta))
save_model(xgb_stacking, 'XGBClassifier_Stacking')


val_prob_files = [file for file in os.listdir(WORKING_DIR) if file.startswith('Prob_Val_') and 'transformed' not in file]
prob_dfs = []
for prob_file in val_prob_files:
    df = pd.read_csv(prob_file)
    model_name = prob_file.replace('Prob_Val_', '').replace('.csv', '')
    df.columns = [model_name + '_' + str(i) for i in range(12)]
    prob_dfs.append(df)

x_val_meta = pd.concat(prob_dfs, axis = 1)
x_val_meta = x_val_meta[x_test_meta.columns]
y_val_pred = xgb_stacking.predict(x_val_meta)
print(accuracy_score(y_val, y_val_pred))

get_matrix(y_test = y_val, y_test_pred = y_val_pred,
            y_train = y_test_meta, y_train_pred = y_test_pred_meta,
            estimator_name = 'XGBClassifier_Stacking', label_encoder = label_encoder)
