"""Modelling
1. Feature Importance by RFECV
2. Feature Selection using sklearn.feature_selection.RFECV on train
3. Fine tune model, using cross validation. —> must sampled after splitting
Steps for Original:
    - Perform self-implemented grid search by training on train, validating on val to get best hyperparameters
    - Use the best hyperparameter to train a model on train + val
    - Predict and save probability for test
"""

""" #### Environment Setup """
# import libraries
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, log_loss, accuracy_score

from util import *
from myhypopt.model_selection import GridSearch
# from hypopt.model_selection import GridSearch

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('fivethirtyeight')
PLOT_HEIGHT = 6
PLOT_WIDTH = PLOT_HEIGHT * 1.618
plt.rcParams["figure.figsize"] = [PLOT_WIDTH,PLOT_HEIGHT]

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Pandas settings
pd.set_option('display.max_columns', 500)
pd.options.display.float_format = '{:,.5f}'.format

""" Constants """
SAMPLE = False

"""#### Load Data"""
test = pd.read_csv(TEST_PATH)
train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)

if SAMPLE:
    train = train.sample(5000)
    val = val.sample(1000)

print('Train Shape: ' + str(train.shape))
print('Val Shape: ' + str(val.shape))
print('Test Shape: ' + str(test.shape))

target = 'country_destination'
y_test = test[target]
x_test = test.drop(columns = target)
y_train = train[target]
x_train = train.drop(columns = target)
y_val = val[target]
x_val = val.drop(columns = target)

run_times_df = pd.DataFrame()

"""#### Useful Functions"""
def get_prob_top_n(prob, n):
    prob_sorted = prob.argsort()[:, ::-1]
    return prob_sorted[:, n]
def top_5_accuracy(y_true, y_prob):
    score = np.array([accuracy_score(get_prob_top_n(y_prob, n), y_true) for n in range(5)]).sum()
    return score
def neg_log_loss(y_true, y_prob):
    score = -1. * log_loss(y_true, y_prob)
    return score

# Grid-search all parameter combinations using a validation set.
def plot_estimator_no_feature_vs_score(rfecv_result, estimator_name, title = None, save = False, show = True):
    plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    plt.xlabel("No. of features selected")
    plt.ylabel("Top 5 Accuracy")
    plt.plot(range(1, rfecv_result.shape[0] + 1), rfecv_result['Score'])
    if title is None:
        title = 'Score vs No. of Features for ' + estimator_name
    plt.title(title, loc = 'center', y=1.1, fontsize = 18)
    if save:
        check_dir(IMAGE_MODEL_DIRECTORY)
        saved_path = os.path.join(IMAGE_MODEL_DIRECTORY, convert_filename(title))
        plt.savefig(saved_path, dpi=200, bbox_inches="tight")
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()
    plt.close()

def process_rfecv_reult(rfecv_selector, columns):
    rfecv_result = pd.DataFrame(data = {'Ranking': rfecv_selector.ranking_, 'Column': columns}).sort_values('Ranking')
    rfecv_result['Score'] = rfecv_selector.grid_scores_
    rfecv_result = rfecv_result.reset_index().drop(columns = ['index'])
    rfecv_result.to_csv(os.path.join(TRAIN_RESULT_PATH, 'LogisticRegression_rfecv_result.csv'), index = False)

    plot_estimator_no_feature_vs_score(rfecv_result, rfecv_selector.estimator.__class__.__name__, save = 'test', show = False,
        title = 'Top 5 Accuracy vs No. of Features for {}'.format(rfecv_selector.estimator.__class__.__name__))
    plot_estimator_no_feature_vs_score(rfecv_result.iloc[:rfecv_selector.support_.sum()], rfecv_selector.estimator.__class__.__name__, save = 'test', show = False,
        title = 'Top 5 Accuracy vs No. of Features for {} (Selected)'.format(rfecv_selector.estimator.__class__.__name__))

def select_columns(selector, x):
    cols = x.columns[selector.support_]
    x = pd.DataFrame(selector.transform(x))
    x.columns = cols
    return x
# estimator = LogisticRegression()

my_scorer = make_scorer(top_5_accuracy, needs_proba = True, greater_is_better=True)
def rfecv_estimator(estimator, x_train, y_train):
    estimator_name = estimator.__class__.__name__
    print('{}: Start RFECV for {}'.format(current_time(), estimator_name))
    selector = RFECV(estimator, step=1, cv=3, verbose=2, scoring=my_scorer)
    selector = selector.fit(x_train, y_train)
    process_rfecv_reult(selector, x_train.columns)

    return selector

def gridsearch_estimator(estimator, param_grid, x_train, y_train, x_val, y_val):
    check_dir(TRAIN_RESULT_PATH)

    estimator_name = estimator.__class__.__name__

    print(current_time() + ': Start grid searching for ' + estimator_name)
    start = time.perf_counter()
    gridsearch_param = {'scoring': my_scorer, 'verbose': 2 }
    # gridsearch_param = {'scoring': 'neg_log_loss', 'verbose': 2 }
    gridsearcher = GridSearch(model = estimator, param_grid = param_grid)
    gridsearcher.fit(x_train, y_train, x_val, y_val, **gridsearch_param)
    run = time.perf_counter() - start
    print('Grid search for {} runs for {:.2f} seconds.'.format(estimator_name, run))

    save_obj(gridsearcher.best_params, 'GridSearch_Best_Params_' + estimator_name)
    save_obj(gridsearcher.best_estimator_.get_params(), 'Params_' + estimator_name)

    print(current_time() + ': Finished grid searching for ' + estimator_name)
    print('Best Params: \n{}'.format(gridsearcher.best_params))
    return gridsearcher

import graphviz
from sklearn import tree
def plot_decision_tree(model, class_names = True, feature_names = None):
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names = feature_names,
                                    class_names = class_names,
                                    filled = False, rounded = True)
    graph = graphviz.Source(dot_data, format="png")
    save_path = os.path.join(IMAGE_MODEL_DIRECTORY, model.__class__.__name__)
    graph.render(save_path)

def process_coefs(model, columns):
    check_dir(TRAIN_RESULT_PATH)
    coefs = pd.DataFrame(model.coef_).transpose()
    coefs.index = columns
    coefs.columns = load_label_encoder('label_encoder_country_destination').classes_
    coefs = coefs.reset_index()
    coefs.to_csv(os.path.join(TRAIN_RESULT_PATH,'Feature_Importance_{}.csv').format(model.__class__.__name__), index = False)

def plot_feature_importance(feature_importance, estimator_name, top = 12):
    feature_importance = feature_importance.sort_values(['Feature Importance'], ascending = False)
    fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    plt.bar(feature_importance.iloc[:top, :]['Feature'], feature_importance.iloc[:top, :]['Feature Importance'])
    plt.ylabel('Feature Importance')
    plt.xlabel('Column')
    plt.xticks(rotation=90)
    title = 'Feature Importance of ' + estimator_name
    plt.title(title, loc = 'center', y=1.1, fontsize = 20)
    saved_path = os.path.join(IMAGE_MODEL_DIRECTORY, convert_filename(title))
    plt.savefig(saved_path, dpi=200, bbox_inches="tight")

def process_feature_importance_by_tree(model):
    check_dir(TRAIN_RESULT_PATH)
    estimator_name = model.__class__.__name__
    feature_importance = pd.DataFrame(data = {'Feature': x_train.columns, 'Feature Importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values(['Feature Importance'], ascending = False)
    feature_importance.to_csv(os.path.join(TRAIN_RESULT_PATH,'Feature_Importance_{}.csv').format(estimator_name), index = False)
    plot_feature_importance(feature_importance, estimator_name, top = 12)


def train_model(estimator, x_train, y_train, x_val, y_val, x_test, y_test, model_param = None, grid_param = None):
    if (model_param is None) == (grid_param is None):
        print('WARNING: One of model_param and grid_param must be not None!')
        return
    check_dir(TRAIN_RESULT_PATH)

    estimator_name = estimator().__class__.__name__
    print('=' * 50)
    print('{}: Start Training {}'.format(current_time(), estimator_name))

    if estimator_name is 'LogisticRegression':
        # estimator = LogisticRegression
        selector = rfecv_estimator(estimator(), x_train.append(x_val), y_train.append(y_val))

        model = estimator(**model_param)
        print(current_time() + ': Start training ' + estimator_name)
        start_total = time.perf_counter()
        model.fit(select_columns(selector, x_train.append(x_val)), y_train.append(y_val))
        run_total = time.perf_counter() - start_total
        print(current_time() + ': Finish training ' + estimator_name)

        process_coefs(model, list(select_columns(selector, x_train).columns))

        x_train_prob = model.predict_proba(select_columns(selector, x_train.append(x_val)))
        x_test_prob = model.predict_proba(select_columns(selector, x_test))

    else:
        # estimator = DecisionTreeClassifier
        if model_param:
            model = estimator(**model_param)
        else:
            gridsearchergridsearch_estimator(estimator(), grid_param, x_train, y_train, x_val, y_val)
            model = estimator(**load_obj('Params_{}'.format(estimator_name)))

        print(current_time() + ': Start training ' + estimator_name)
        start_total = time.perf_counter()
        model.fit(x_train.append(x_val), y_train.append(y_val))
        run_total = time.perf_counter() - start_total
        print(current_time() + ': Finish training ' + estimator_name)

        if estimator_name == 'DecisionTreeClassifier':
            plot_decision_tree(model, class_names = load_label_encoder('label_encoder_country_destination').classes_, feature_names = x_train.columns)

        try:
            process_feature_importance_by_tree(model)
        except:
            print('{} does not have feature importance'.format(estimator_name))

        x_train_prob = model.predict_proba(x_train.append(x_val))
        x_test_prob = model.predict_proba(x_test)

    run_times_df[estimator_name] = [run_total]

    np.save(os.path.join(TRAIN_RESULT_PATH, "Prob_Train_{}.npy".format(estimator_name)), x_train_prob)
    np.save(os.path.join(TRAIN_RESULT_PATH, "Prob_Test_{}.npy".format(estimator_name)), x_test_prob)

    print(current_time() + ': Saved Probability for Train and Test'.format(estimator_name))

    print('Train Score for Optimized Parameters:', top_5_accuracy(y_train.append(y_val), x_train_prob))
    print('Test Score for Optimized Parameters:', top_5_accuracy(y_test, x_test_prob))

    return estimator

estimators_params_grid = {
    'RandomForestClassifier': {
        'min_samples_split' : [10, 30, 50, 100, 200],
        'n_estimators': [25, 100, 200],
        'random_state': [SEED]},
    'LogisticRegression': {'random_state': [SEED]},
    'DecisionTreeClassifier': {
        'min_samples_split' : [300, 400, 500, 600, 700, 800, 900, 1000],
        'random_state': [SEED]},
    'XGBClassifier': {
        'earning_rate': [0.1, 0.3, 0.5], #default: .3
        'max_depth': [2,4, 6,10], #default 2
        'n_estimators': [10, 25, 50, 100],
        'subsample': [0.5],
        'colsample_bytree': [0.5],
        'objective': ['multi:softprob'],
        'random_state': [SEED]},
    }
estimators_params_grid = {
    'RandomForestClassifier': {
        'min_samples_split' : [200],
        'n_estimators': [25],
        'random_state': [SEED]},
    'LogisticRegression': {'random_state': [SEED]},
    'DecisionTreeClassifier': {
        'min_samples_split' : [800],
        'random_state': [SEED]},
    'XGBClassifier': {
        'earning_rate': [0.3],
        'max_depth': [4],
        'n_estimators': [30],
        'subsample': [0.7],
        'colsample_bytree': [0.7],
        'objective': ['multi:softprob'],
        'random_state': [SEED]}
    }
estimators_params = {
    'RandomForestClassifier': {
        'min_samples_split' : 200,
        'n_estimators': 25,
        'random_state': SEED},
    'LogisticRegression': {'random_state': SEED},
    'DecisionTreeClassifier': {
        'min_samples_split' : 800,
        'random_state': SEED},
    'XGBClassifier': {
        'earning_rate': .3,
        'max_depth': 4,
        'n_estimators': 30,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'multi:softprob',
        'random_state': SEED}
    }

train_model(LogisticRegression, x_train, y_train, x_val, y_val, x_test, y_test, model_param = estimators_params['LogisticRegression'])
train_model(DecisionTreeClassifier, x_train, y_train, x_val, y_val, x_test, y_test, grid_param = estimators_params_grid['DecisionTreeClassifier'])
train_model(RandomForestClassifier, x_train, y_train, x_val, y_val, x_test, y_test, grid_param = estimators_params_grid['RandomForestClassifier'])
train_model(XGBClassifier, x_train, y_train, x_val, y_val, x_test, y_test, grid_param = estimators_params_grid['XGBClassifier'])

from pyearth import Earth
model = Earth(max_degree=2, )
model.fit(x_train,y_train)
model.summary()

#
# 'Earth Model\n------------------------------------------------------------------------------\n
# Basis Function                                           Pruned  Coefficient  \n
# ------------------------------------------------------------------------------\n
# (Intercept)                                              No      -370.848     \n
# affiliate_provider_email                                 No      1.44863      \n
# affiliate_channel_other                                  No      0.959827     \n
# date_first_active_year                                   No      0.187482     \n
# first_os_Android*affiliate_provider_email                No      -4.49792     \n
# first_browser_-unknown-*date_first_active_year           No      0.000224682  \n
# date_account_created_month_9*affiliate_provider_email    No      -2.88116     \n
# date_first_active_dayofyear                              No      -0.00206883  \n
# language_de                                              No      -1.68519     \n
# age_bkt_25-29*date_first_active_year                     No      0.000122373  \n
# date_first_active_dayofyear*date_first_active_dayofyear  No      9.11704e-06  \n
# ------------------------------------------------------------------------------\n
# MSE: 8.3100, GCV: 8.3346, RSQ: 0.0286, GRSQ: 0.0259'
# run_times_df.to_csv(os.path.join(TRAIN_RESULT_PATH, 'Run_Time.csv'), index = False)

# """#### Neural Network"""
# # Neural Network
# import numpy
# import pandas
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.utils import np_utils
#
# numpy.random.seed(SEED)
# x_train_values = x_train.values
# x_test_values = x_test.values
# y_train_cat = np_utils.to_categorical(y_train)
#
# def get_model():
#   # create model
#   model = Sequential()
#   model.add(Dense(64, input_dim=x_train_values.shape[1], activation='relu'))
#   model.add(Dropout(0.2))
#   model.add(Dense(64, activation='relu'))
#   model.add(Dense(32, activation='relu'))
#   model.add(Dropout(0.3))
#   model.add(Dense(11, activation='softmax'))
#   # Compile model
#   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#   return model
#
# # RMSProp/adam
#
# print(current_time() + ': Start training ' + 'NeuralNetwork')
# start_total = time.perf_counter()
#
# model = get_model()
# history = model.fit(x_train_values, y_train_cat, epochs=100, batch_size=32,
#                     callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=0),
#                     ModelCheckpoint(os.path.join(MODEL_PATH, 'NeuralNetwork.hdf5'),
#                     monitor='val_loss', verbose=1, save_best_only=True, mode='max')],
#                     verbose=2, validation_split=0.3, shuffle=True)
#
# run_total = time.perf_counter() - start_total
# print(current_time() + ': Finish training ' + 'NeuralNetwork')
# run_times_df['NeuralNetwork'] = [run_total]
#
# # Weight
# from keras.wrappers.scikit_learn import KerasClassifier
# import eli5
# from eli5.sklearn import PermutationImportance
# sk_params = {'epochs':100,
#             'batch_size':32,
#             'callbacks':[EarlyStopping(monitor='val_loss', patience=3, verbose=0),
#                         ModelCheckpoint(os.path.join(MODEL_PATH, 'NeuralNetwork.hdf5'),monitor='val_loss', verbose=1, save_best_only=True, mode='max')],
#             'verbose':2,
#             'validation_split':0.3,
#             'shuffle':True}
# my_model = KerasClassifier(build_fn=get_model, **sk_params)
# my_model.fit(x_train_values, y_train_cat)
# perm = PermutationImportance(my_model, random_state=SEED).fit(x_train_values, y_train_cat)
# eli5.show_weights(perm, feature_names = x_train.columns.tolist(), include_styles = True)
#
#
# # # list all data in history
# # print(history.history.keys())
# #
# # # summarize history for accuracy
# # plt.figure(figsize=(8,6))
# # plt.plot(history.history['acc'])
# # plt.plot(history.history['val_acc'])
# # plt.legend(['train', 'test'], loc='bottom right', prop={'size': 18})
# # plt.title('model accuracy')
# # plt.ylabel('accuracy')
# # plt.xlabel('epoch')
# # plt.ylim(0.5, 0.7)
# # plt.savefig('NeuralNetwork_Accuracy_History')
# #
# # # summarize history for loss
# # plt.figure(figsize=(8,6))
# # plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# # plt.title('model loss')
# # plt.ylabel('loss')
# # plt.xlabel('epoch')
# # plt.ylim(1.5, 0.8)
# # plt.legend(['train', 'test'], loc='bottom right', prop={'size': 18})
# # plt.savefig('NeuralNetwork_Loss_History')
#
# # model.save(os.path.join(MODEL_PATH, 'nn_2.h5')) # save model
#
# clf_name = 'NeuralNetwork'
# y_train_prob = model.predict(x_train.append(x_val))
# y_test_prob = model.predict(x_test)
# np.save(os.path.join(TRAIN_RESULT_PATH, "Prob_Train_{}.npy").format(clf_name), y_train_prob)
# np.save(os.path.join(TRAIN_RESULT_PATH, "Prob_Test_{}.npy").format(clf_name), y_test_prob)
#
# run_times_df.to_csv(os.path.join(TRAIN_RESULT_PATH, 'Run_Time.csv'), index = False)
