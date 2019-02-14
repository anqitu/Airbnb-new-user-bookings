"""Modelling
1. Feature Importance by RFECV
2. Feature Selection using sklearn.feature_selection.RFECV on train
3. Fine tune model, using cross validation. —> must sampled after splitting

So, we have train, val and test train_smote, val_smote

Steps for Original:
    - Perform self-implemented grid search by training on train, validating on val to get best hyperparameters
    - Use the best hyperparameter to train a model on train + val
    - Predict and save probability for test
Steps for Oversampled:
- Perform self-implemented grid search by training on train_smote, validating on val to get best hyperparameters
- Use the best hyperparameter to train a model on train_smote + val_smote
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


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Pandas settings
pd.set_option('display.max_columns', 500)
pd.options.display.float_format = '{:,.5f}'.format

""" Constant """
MAP_MIN_TO_OTHER = False

if MAP_MIN_TO_OTHER:
    TRAIN_RAW_PATH = TRAIN_RAW_MIN_TO_OTHERS_PATH
    VAL_RAW_PATH = VAL_RAW_MIN_TO_OTHERS_PATH

    TRAIN_SMOTE_PATH = TRAIN_SMOTE_MIN_TO_OTHERS_PATH
    VAL_SMOTE_PATH = VAL_SMOTE_MIN_TO_OTHERS_PATH


"""#### Load Data"""
test = pd.read_csv(TEST_PATH)
train_raw = pd.read_csv(TRAIN_RAW_PATH)
val_raw = pd.read_csv(VAL_RAW_PATH)
train_smote = pd.read_csv(TRAIN_SMOTE_PATH)
val_smote = pd.read_csv(VAL_SMOTE_PATH)

print('Train Shape: ' + str(train_raw.shape))
print('Val Shape: ' + str(val_raw.shape))
print('Sampled Train Shape: ' + str(train_smote.shape))
print('Sampled Val Shape: ' + str(val_smote.shape))
print('Test Shape: ' + str(test.shape))

target = 'country_destination'
y_test = test[target]
x_test = test.drop(columns = target)
y_train_raw = train_raw[target]
x_train_raw = train_raw.drop(columns = target)
y_val_raw = val_raw[target]
x_val_raw = val_raw.drop(columns = target)
y_train_smote = train_smote[target]
x_train_smote = train_smote.drop(columns = target)
y_val_smote = val_smote[target]
x_val_smote = val_smote.drop(columns = target)

run_times_df = pd.DataFrame()

"""#### Useful Functions"""
def top_3_accuracy(y_true, y_prob):
    score = np.array([accuracy_score(get_prob_top_n(y_prob, n), y_true) for n in range(3)]).sum()
    return score
def neg_log_loss(y_true, y_prob):
    score = -1. * log_loss(y_true, y_prob)
    return score

my_scorer = make_scorer(top_3_accuracy, needs_proba = True, greater_is_better=True)
gridsearch_param = {'scoring': my_scorer, 'verbose': 2 }
gridsearch_param = {'scoring': 'neg_log_loss', 'verbose': 2 }

# Grid-search all parameter combinations using a validation set.
def gridsearch_estimator(estimator, param_grid, x_train, y_train, x_val, y_val, smote):

    if smote not in ['', '_Smote']:
        print('smote must be "", or "_Smote"')
    check_dir(TRAIN_RESULT_PATH)

    estimatorName = estimator.__class__.__name__ + smote

    print(current_time() + ': Start grid searching for ' + estimatorName)
    start = time.perf_counter()
    gridsearcher = GridSearch(model = estimator, param_grid = param_grid)
    gridsearcher.fit(x_train, y_train, x_val, y_val, **gridsearch_param)
    run = time.perf_counter() - start
    print('Grid search for {} runs for {:.2f} seconds.'.format(estimatorName, run))

    save_obj(gridsearcher.best_params, 'GridSearch_Best_Params_' + estimatorName)
    save_obj(gridsearcher.best_estimator_.get_params(), 'Params_' + estimatorName)

    print(current_time() + ': Finished grid searching for ' + estimatorName)
    print('Best Params: \n{}'.format(gridsearcher.best_params))
    return gridsearcher

def train_clf(estimator, x_train, y_train, x_val, y_val, x_test, y_test, smote):
    check_dir(TRAIN_RESULT_PATH)

    if smote not in ['', '_Smote']:
        print('smote must be "", or "_Smote"')
    estimatorName = estimator.__class__.__name__ + smote
    x_train = pd.concat([x_train, x_val])
    y_train = pd.concat([y_train, y_val])

    print(current_time() + ': Start training ' + estimatorName)
    start_total = time.perf_counter()
    estimator.fit(x_train, y_train)
    run_total = time.perf_counter() - start_total
    print(current_time() + ': Finish training ' + estimatorName)

    run_times_df[estimatorName] = [run_total]

    x_train_prob = estimator.predict_proba(x_train)
    x_test_prob = estimator.predict_proba(x_test)
    np.save(os.path.join(TRAIN_RESULT_PATH, "Prob_Train_{}.npy".format(estimatorName)), x_train_prob)
    np.save(os.path.join(TRAIN_RESULT_PATH, "Prob_Test_{}.npy".format(estimatorName)), x_test_prob)

    print(current_time() + ': Saved Probability for Train and Test'.format(estimatorName))
    save_model(estimator, estimatorName)

    print('Train Score for Optimized Parameters:', top_3_accuracy(y_train, x_train_prob))
    print('Test Score for Optimized Parameters:', top_3_accuracy(y_test, x_test_prob))

    return estimator

def get_feature_importance_by_tree(estimator, smote):
    check_dir(TRAIN_RESULT_PATH)
    if smote not in ['', '_Smote']:
        print('smote must be "", or "_Smote"')

    estimatorName = estimator.__class__.__name__ + smote
    feature_importance = pd.DataFrame(data = {'Feature': x_train_raw.columns, 'Feature Importance': estimator.feature_importances_})
    feature_importance = feature_importance.sort_values(['Feature Importance'], ascending = False)
    feature_importance.to_csv(os.path.join(TRAIN_RESULT_PATH,'Feature_Importance_{}.csv').format(estimatorName), index = False)

    return feature_importance

"""#### LogisticRegression"""
param_grid_lr = [
    {'C': [0.01, 0.1, 1, 5, 10],
     'penalty': ['l2', 'l1'],
     'solver' : ['liblinear'],
     'multi_class' : ['ovr'],
     'class_weight': [None, 'balanced'],
     'max_iter': [1000]}]

param_grid_lr = {'C': [1],
     'penalty': ['l2'],
     'solver' : ['liblinear'],
     'multi_class' : ['ovr'],
     'class_weight': [None],
     'max_iter': [1000]}

gridsearcher_lr_raw = gridsearch_estimator(estimator = LogisticRegression(random_state = SEED), param_grid = param_grid_lr, smote = '',
                x_train = x_train_raw, y_train = y_train_raw, x_val = x_val_raw, y_val = y_val_raw)
gridsearcher_lr_smote = gridsearch_estimator(estimator = LogisticRegression(random_state = SEED), param_grid = param_grid_lr, smote = '_Smote',
                x_train = x_train_smote, y_train = y_train_smote, x_val = x_val_raw, y_val = y_val_raw)

lr_raw = LogisticRegression(**load_obj('Params_LogisticRegression'))
lr_raw = train_clf(lr_raw, smote = '', x_train = x_train_raw, y_train = y_train_raw,
                x_val = x_val_raw, y_val = y_val_raw, x_test = x_test, y_test = y_test)
lr_smote = LogisticRegression(**load_obj('Params_LogisticRegression_Smote'))
lr_raw = train_clf(lr_raw, smote = '_Smote', x_train = x_train_smote, y_train = y_train_smote,
                x_val = x_val_smote, y_val = y_val_smote, x_test = x_test, y_test = y_test)

np.array([accuracy_score(get_prob_top_n(lr_raw.predict_proba(x_test), n), y_test) for n in range(11)]).cumsum()

"""@TODO"""
# if rfecv:
#     rfecv_result = pd.read_csv(os.path.join(TRAIN_RESULT_PATH,'RFECV_RANK_{}.csv').format(clf_name))
#     selected_features = list(rfecv_result[rfecv_result['Ranking'] == 1]['Column'])
#     x_train = x_train[selected_features]
#     x_test = x_test[selected_features]
# rfecv_param = {'cv': 3,
#                'scoring': 'neg_log_loss',
#                # 'scoring': 'accuracy',
#                'n_jobs': None,
#                'verbose': 2}
# if rfecv:
#     rfecv_result = gridsearcher.best_estimator_
#     estimators_RFECV[estimatorName] = rfecv_result
#     rfe_result_rank = pd.DataFrame(data = {'Ranking': rfecv_result.ranking_, 'Column': x_train.columns}).sort_values('Ranking')
#     rfe_result_rank.to_csv(os.path.join(TRAIN_RESULT_PATH,'RFECV_RANK_{}.csv').format(estimatorName), index = False)
#
# # Coefficient Abosolute Average
# lr = LogisticRegression()
# lr.fit(x_train, y_train)
#
# lr_coefs_df = pd.DataFrame(data = lr.coef_)
# lr_coefs_df.index = label_encoder.inverse_transform(lr_coefs_df.index)
# lr_coefs_df = lr_coefs_df.transpose()
# lr_coefs_df['Column'] = x_train.columns
# lr_coefs_df.to_csv(os.path.join(TRAIN_RESULT_PATH,'Feature_Importance_{}.csv').format(estimatorName), index = False)
#
# lr_abs_coefs_df = pd.DataFrame(data = {'Column': x_train.columns, 'Coefficient Abosolute Average': np.absolute(lr.coef_).mean(axis = 0)})
# lr_abs_coefs_df = lr_abs_coefs_df.sort_values(['Coefficient Abosolute Average'], ascending=False)
# lr_abs_coefs_df
#
# TOP = 12
# fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
# plt.bar(lr_abs_coefs_df.iloc[:TOP, :]['Column'], lr_abs_coefs_df.iloc[:TOP, :]['Coefficient Abosolute Average'])
# plt.ylabel('Coefficient Abosolute Average')
# plt.xlabel('Column')
# plt.xticks(rotation=45)
# title = 'Coefficient Abosolute Average of Columns (Top {})'.format(TOP)
# plt.title(title, loc = 'center', y=1.1, fontsize = 20)
# saved_path = os.path.join(IMAGE_MODEL_DIRECTORY, convert_title_to_filename(title))
# plt.savefig(saved_path, dpi=200, bbox_inches="tight")


"""#### DecisionTreeClassifier"""
param_grid_tree = {'min_samples_split' : [10, 30, 50, 100, 150, 200, 300, 400, 500],
                   'max_depth': [4,8,12],
                   'class_weight': ['balanced', None]}
param_grid_tree = {'min_samples_split' : [500],
                   'max_depth': [3, 4, 5],
                   'class_weight': [None]}
gridsearcher_tree_raw = gridsearch_estimator(estimator = DecisionTreeClassifier(random_state = SEED), param_grid = param_grid_tree, smote = '',
                x_train = x_train_raw, y_train = y_train_raw, x_val = x_val_raw, y_val = y_val_raw)
gridsearcher_tree_smote = gridsearch_estimator(estimator = DecisionTreeClassifier(random_state = SEED), param_grid = param_grid_tree, smote = '_Smote',
                x_train = x_train_smote, y_train = y_train_smote, x_val = x_val_raw, y_val = y_val_raw)

tree_raw = DecisionTreeClassifier(**load_obj('Params_DecisionTreeClassifier'))
tree_raw = train_clf(tree_raw, smote = '', x_train = x_train_raw, y_train = y_train_raw,
                x_val = x_val_raw, y_val = y_val_raw, x_test = x_test, y_test = y_test)
tree_smote = DecisionTreeClassifier(**load_obj('Params_DecisionTreeClassifier_Smote'))
tree_smote = train_clf(tree_smote, smote = '_Smote', x_train = x_train_smote, y_train = y_train_smote,
                x_val = x_val_smote, y_val = y_val_smote, x_test = x_test, y_test = y_test)


# np.array([accuracy_score(get_prob_top_n(tree_smote.predict_proba(x_train_smote), n), y_train_smote) for n in range(11)]).cumsum()
# np.array([accuracy_score(get_prob_top_n(tree_smote.predict_proba(x_train_raw), n), y_train_raw) for n in range(11)]).cumsum()
# np.array([accuracy_score(get_prob_top_n(tree_smote.predict_proba(x_val_raw), n), y_val_raw) for n in range(11)]).cumsum()
# np.array([accuracy_score(get_prob_top_n(tree_smote.predict_proba(x_val_raw), n), y_val_raw) for n in range(11)]).cumsum()
# np.array([accuracy_score(get_prob_top_n(tree_smote.predict_proba(x_test), n), y_test) for n in range(11)]).cumsum()

# tree.fit(x_train_smote, y_train_smote)
# np.array([accuracy_score(get_prob_top_n(tree.predict_proba(x_test), n), y_test) for n in range(11)]).cumsum()

# # Plot tree
# import graphviz
# from sklearn import tree
# dot_data = tree.export_graphviz(tree_raw, out_file=None,
#                                 feature_names = x_train_raw.columns,
#                                 class_names = True,
#                                 filled = True, rounded = True)
# graph = graphviz.Source(dot_data, format="png")
# save_path = os.path.join(IMAGE_MODEL_DIRECTORY, 'DecisionTreeClassifier')
# graph.render(save_path)
#
# # Feature Importance
# dtree_feature_importance_df = get_feature_importance_by_tree(tree_raw, smote = '')
# # dtree_feature_importance_df.iloc[:12, :].sum()
# TOP = 12
# fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
# plt.bar(dtree_feature_importance_df.iloc[:TOP, :]['Feature'], dtree_feature_importance_df.iloc[:TOP, :]['Feature Importance'])
# plt.ylabel('Feature Importance')
# plt.xlabel('Column')
# plt.xticks(rotation=90)
# title = 'Feature Importance of DecisionTree'
# plt.title(title, loc = 'center', y=1.1, fontsize = 20)
# saved_path = os.path.join(IMAGE_MODEL_DIRECTORY, convert_title_to_filename(title))
# plt.savefig(saved_path, dpi=200, bbox_inches="tight")


"""#### RandomForestClassifier"""
param_grid_rfc = {'min_samples_split' : [10, 30, 50, 100, 200],
                  'n_estimators': [25, 100, 200]}
param_grid_rfc = {'min_samples_split' : [400],
                  'n_estimators': [50]}
gridsearcher_rfc_raw = gridsearch_estimator(estimator = RandomForestClassifier(random_state = SEED), param_grid = param_grid_tree, smote = '',
                x_train = x_train_raw, y_train = y_train_raw, x_val = x_val_raw, y_val = y_val_raw)
gridsearcher_rfc_smote = gridsearch_estimator(estimator = RandomForestClassifier(random_state = SEED), param_grid = param_grid_rfc, smote = '_Smote',
                x_train = x_train_smote, y_train = y_train_smote, x_val = x_val_raw, y_val = y_val_raw)

rfc_raw = RandomForestClassifier(**load_obj('Params_RandomForestClassifier'))
rfc_raw = train_clf(rfc_raw, smote = '', x_train = x_train_raw, y_train = y_train_raw,
                x_val = x_val_raw, y_val = y_val_raw, x_test = x_test, y_test = y_test)
rfc_smote = DecisionrfcClassifier(**load_obj('Params_RandomForestClassifier_Smote'))
rfc_smote = train_clf(rfc_smote, smote = '_Smote', x_train = x_train_smote, y_train = y_train_smote,
                x_val = x_val_smote, y_val = y_val_smote, x_test = x_test, y_test = y_test)

# Feature Importance
rfc_feature_importance_df = get_feature_importance_by_tree(rfc, smote = '')
TOP = 12
fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
plt.bar(rfc_feature_importance_df.iloc[:TOP, :]['Feature'], rfc_feature_importance_df.iloc[:TOP, :]['Feature Importance'])
plt.ylabel('Feature Importance')
plt.xlabel('Column')
plt.xticks(rotation=90)
title = 'Feature Importance of RandomForest (Top {})'.format(TOP)
plt.title(title, loc = 'center', y=1.1, fontsize = 20)
saved_path = os.path.join(IMAGE_MODEL_DIRECTORY, convert_title_to_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")


"""#### XGBClassifier"""
param_grid_xgb = {'earning_rate': [.1, .3, .5], #default: .3
                  'max_depth': [2,4, 6,10], #default 2
                  'n_estimators': [10, 25, 50, 100],
                  'subsample': [0.5],
                  'colsample_bytree': [0.5],
                  'objective': ['multi:softprob'],}
param_grid_xgb = {'earning_rate': [.3], #default: .3
                  'max_depth': [4], #default 2
                  'n_estimators': [30],
                  'subsample': [0.7],
                  'colsample_bytree': [0.7],
                  'objective': ['multi:softprob'],}
gridsearcher_xgb_raw = gridsearch_estimator(estimator = XGBClassifier(seed=SEED), param_grid = param_grid_tree, smote = '',
                x_train = x_train_raw, y_train = y_train_raw, x_val = x_val_raw, y_val = y_val_raw)
gridsearcher_xgb_smote = gridsearch_estimator(estimator = XGBClassifier(seed=SEED), param_grid = param_grid_xgb, smote = '_Smote',
                x_train = x_train_smote, y_train = y_train_smote, x_val = x_val_raw, y_val = y_val_raw)

xgb_raw = XGBClassifier(**load_obj('Params_XGBClassifier'))
xgb_raw = train_clf(xgb_raw, smote = '', x_train = x_train_raw, y_train = y_train_raw,
                x_val = x_val_raw, y_val = y_val_raw, x_test = x_test, y_test = y_test)
xgb_smote = XGBClassifier(**load_obj('Params_XGBClassifier_Smote'))
xgb_smote = train_clf(xgb_smote, smote = '_Smote', x_train = x_train_smote, y_train = y_train_smote,
                x_val = x_val_smote, y_val = y_val_smote, x_test = x_test, y_test = y_test)

# Feature Importance
xgb_feature_importance_df = get_feature_importance_by_tree(xgb, smote = '')
TOP = 12
fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
plt.bar(rfc_feature_importance_df.iloc[:TOP, :]['Feature'], rfc_feature_importance_df.iloc[:TOP, :]['Feature Importance'])
plt.ylabel('Feature Importance')
plt.xlabel('Column')
plt.xticks(rotation=90)
title = 'Feature Importance of XGBClassifier (Top {})'.format(TOP)
plt.title(title, loc = 'center', y=1.1, fontsize = 20)
saved_path = os.path.join(IMAGE_MODEL_DIRECTORY, convert_title_to_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")


"""#### Neural Network"""
# Neural Network
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

numpy.random.seed(SEED)
x_train_values = x_train_raw.values
x_test_values = x_test.values
y_train_cat = np_utils.to_categorical(y_train)

def get_model():
  # create model
  model = Sequential()
  model.add(Dense(64, input_dim=x_train_values.shape[1], activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(11, activation='softmax'))
  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

# RMSProp/adam

print(current_time() + ': Start training ' + 'NeuralNetwork')
start_total = time.perf_counter()

model = get_model()
history = model.fit(x_train_values, y_train_cat, epochs=100, batch_size=32,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=0),
                    ModelCheckpoint(os.path.join(MODEL_PATH, 'NeuralNetwork.hdf5'),
                    monitor='val_loss', verbose=1, save_best_only=True, mode='max')],
                    verbose=2, validation_split=0.3, shuffle=True)

run_total = time.perf_counter() - start_total
print(current_time() + ': Finish training ' + 'NeuralNetwork')
run_times_df['NeuralNetwork'] = [run_total]

# Weight
# from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
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


# # list all data in history
# print(history.history.keys())
#
# # summarize history for accuracy
# plt.figure(figsize=(8,6))
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.legend(['train', 'test'], loc='bottom right', prop={'size': 18})
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.ylim(0.5, 0.7)
# plt.savefig('NeuralNetwork_Accuracy_History')
#
# # summarize history for loss
# plt.figure(figsize=(8,6))
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.ylim(1.5, 0.8)
# plt.legend(['train', 'test'], loc='bottom right', prop={'size': 18})
# plt.savefig('NeuralNetwork_Loss_History')

# model.save(os.path.join(MODEL_PATH, 'nn_2.h5')) # save model

clf_name = 'NeuralNetwork'
y_train_prob = model.predict(x_train)
y_test_prob = model.predict(x_test)
np.save(os.path.join(TRAIN_RESULT_PATH, "Prob_Train_{}.npy").format(clf_name), y_train_prob)
np.save(os.path.join(TRAIN_RESULT_PATH, "Prob_Test_{}.npy").format(clf_name), y_test_prob)

run_times_df.to_csv(os.path.join(TRAIN_RESULT_PATH, 'Run_Time.csv'), index = False)
