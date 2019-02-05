"""Modelling
1. Feature Importance by RFECV
2. Feature Selection using sklearn.feature_selection.RFECV on train
3. Fine tune model, using cross validation. —> must sampled after splitting
    1. model_original on train, using GridSeachCV
    2. model_sampled on sampled train, using ParamSearch, each with validation before sampled.
4. Meta holdout scheme with OOF meta-features.
    4.1. Fit models to whole train_data and predict for test_data.
       Let's call these features test_meta --> Save scores for each model's prediction
    4.2. Split train into K folds. Iterate through each fold:
       - retain N diverme models on all folds except current fold.
       - predict for the current fold
       After this step, for each object in train_data,
       we will have N meta-features (also known as out-of-fold predictions, OOF).
       Lets call them train_meta
    4.3. Split train_meta into two parts: train_metaA and train_metaB.
       Fit a meta-model to train_metaA while validating its hyperparameters on train_metaB.
    4.4. When the meta-model is validated, fit it to train_meta and predict for test_meta --> Save scores for meta-model's prediction
    * For oversampling version: all the data used for training any model are sampled right before fitting.
5. Compare performance for every single model, stacked model and oversampled model.
"""

""" #### Environment Setup """
import os
os.getcwd()
WORKING_DIR = '/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings'
# WORKING_DIR = '/content'
# os.listdir(WORKING_DIR)

# import libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from util import *

# Draw inline
%matplotlib inline

# Set figure aesthetics
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = [15,8]

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Pandas settings
pd.set_option('display.max_columns', 500)
pd.options.display.float_format = '{:,.5f}'.format


"""#### Load Data"""
train = pd.read_csv(TRAIN_PATH)
# train = train.sample(5000)
test = pd.read_csv(TEST_PATH)

"""#### Useful Functions"""
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title=None, cmap=plt.cm.Blues, save = False, show = True):
    fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    plt.imshow(cm, interpolation='nearest', cmap=cmap, )
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if title is None:
        title = 'Confusion Matrix'
        if normalize:
            title = title + ' (Normalized)'
    plt.title(title, loc = 'center', y=1.15, fontsize = 25)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save:
        check_dir(IMAGE_MATRIX_DIRECTORY)
        saved_path = os.path.join(IMAGE_MATRIX_DIRECTORY, convert_title_to_filename(title))
        plt.savefig(saved_path, dpi=200, bbox_inches="tight")
        plt.show()
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()


def get_matrix(y_test, y_test_pred, y_train, y_train_pred, estimator_name, label_encoder):
    check_dir(TRAIN_RESULT_PATH)

    title = 'Confusion Matrix for ' + estimator_name + ' Test'
    df_confusion = pd.crosstab(pd.Series(label_encoder.inverse_transform(y_test), name='True'), pd.Series(label_encoder.inverse_transform(y_test_pred), name='Predict'))
    df_confusion.to_csv(os.path.join(TRAIN_RESULT_PATH, convert_title_to_filename(title) + '.csv'))
    plot_confusion_matrix(confusion_matrix(y_test, y_test_pred), label_encoder.classes_, title = title, save = True)

    title = 'Confusion Matrix for ' + estimator_name  + ' Train'

    df_confusion = pd.crosstab(pd.Series(label_encoder.inverse_transform(y_train), name='True'), pd.Series(label_encoder.inverse_transform(y_train_pred), name='Predict'))
    df_confusion.to_csv(os.path.join(TRAIN_RESULT_PATH, convert_title_to_filename(title) + '.csv'))
    plot_confusion_matrix(confusion_matrix(y_train, y_train_pred), label_encoder.classes_, title = title, save = True)

# Plot number of features VS. cross-validation scores
def plot_estimator_no_feature_vs_accuracy_score(rfecv_result, estimatorName, title = None, save = False, show = True):
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (Accuracy)")
    plt.plot(range(1, len(rfecv_result.grid_scores_) + 1), rfecv_result.grid_scores_)
    plt.ylim([0.4,0.7])

    if title is None:
        title = 'Number of Features vs Accuracy Score for ' + estimatorName
    plt.title(title, loc = 'center', y=1.1, fontsize = 25)

    if save:
        check_dir(IMAGE_BIN_DIRECTORY)
        saved_path = os.path.join(IMAGE_BIN_DIRECTORY, convert_title_to_filename(title))
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()

def train_clf(clf, x_train, x_test):
    check_dir(TRAIN_RESULT_PATH)

    clf_name = clf.__class__.__name__

    x_train_transformed = estimators_RFECV[clf.__class__.__name__].transform(x_train)
    x_test_transformed = estimators_RFECV[clf.__class__.__name__].transform(x_test)

    print(current_time() + ': Start training ' + clf_name)
    start_total = time.perf_counter()
    clf.fit(x_train_transformed, y_train)
    run_total = time.perf_counter() - start_total
    print(current_time() + ': Finish training ' + clf_name)
    run_times_df[clf_name] = [run_total]

    y_train_pred = clf.predict(x_train_transformed)
    y_test_pred = clf.predict(x_test_transformed)
    train_result_df['y_train_pred_' + clf_name] = y_train_pred
    test_result_df['y_test_pred_' + clf_name] = y_test_pred

    accuracy_scores = []
    accuracy_scores.append(accuracy_score(y_train_pred, y_train))
    accuracy_scores.append(accuracy_score(y_test_pred, y_test))
    pd.DataFrame(clf.predict_proba(x_train_transformed)).to_csv(os.path.join(TRAIN_RESULT_PATH,"Prob_Train_{}.csv").format(clf_name), index = False)
    pd.DataFrame(clf.predict_proba(x_test_transformed)).to_csv(os.path.join(TRAIN_RESULT_PATH,"Prob_Test_{}.csv").format(clf_name), index = False)

    score_df = pd.DataFrame(data = {'Prediction': ['Train Prediction', 'Test Prediction'], 'Accuracy Score': accuracy_scores})
    score_df.to_csv(os.path.join(TRAIN_RESULT_PATH,'Scores_for_{}.csv'.format(clf_name)), index = False)

    get_matrix(y_test = y_test, y_test_pred = y_test_pred,
                y_train = y_train, y_train_pred = y_train_pred,
                estimator_name = clf_name, label_encoder = label_encoder)

    print(current_time() + ': Finish getting confusion matrix for ' + clf_name)

    save_model(clf, clf.__class__.__name__)

    return clf

def get_feature_importance_by_tree(clf):
    check_dir(TRAIN_RESULT_PATH)
    feature_importance = pd.DataFrame(data = {'feature': x_train.columns, 'feature_importance': clf.feature_importances_})
    feature_importance.sort_values(['feature_importance'], ascending = False)
    feature_importance.to_csv(os.path.join(TRAIN_RESULT_PATH,'Feature_Importance_{}.csv').format(clf.__class__.__name__), index = False)

    return feature_importance

import pickle
def save_obj(obj, name):
    with open(os.path.join(TRAIN_RESULT_PATH, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(os.path.join(TRAIN_RESULT_PATH, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

"""#### X and Y """
target = 'country_destination'
label_encoder = load_label_encoder('label_encoder_country_destination')
y_train = train[target]
y_test = test[target]
x_train = train.drop(columns = target)
x_test = test.drop(columns = target)

train_result_df = pd.DataFrame(data = {'y_train': y_train})
test_result_df = pd.DataFrame(data = {'y_test': y_test})
run_times_df = pd.DataFrame()


"""
1&2. Feature Importance & Feature Selection by RFECV
3. Fine tune model, using cross validation.
- model_original on train, using GridSeachCV
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV

estimators_params = {}
estimators_RFECV = {}
gridsearchers = {}
rfecv_param = {'cv': 3,
               'scoring': 'accuracy',
               'n_jobs': None,
               'verbose': 2}

gridsearch_param = {'cv': 3,
                    'scoring': 'accuracy',
                    'n_jobs': -1,
                    'verbose': 2,
                    'refit': True}

def gridsearch_rfecv_estimator(estimator, param_grid):
    check_dir(TRAIN_RESULT_PATH)

    estimatorName = estimator.__class__.__name__
    selector = RFECV(estimator, **rfecv_param)
    gridsearcher = GridSearchCV(selector, param_grid, **gridsearch_param)

    print(current_time() + ': Start training ' + estimatorName)
    start = time.perf_counter()
    gridsearcher.fit(x_train, y_train)
    run = time.perf_counter() - start
    print('{} runs for {:.2f} seconds.'.format(estimatorName, run))

    rfecv_result = gridsearcher.best_estimator_
    estimators_RFECV[estimatorName] = rfecv_result
    rfe_result_rank = pd.DataFrame(data = {'Ranking': rfecv_result.ranking_, 'Column': x_train.columns}).sort_values('Ranking')
    plot_estimator_no_feature_vs_accuracy_score(rfecv_result, estimatorName)

    estimators_params[estimatorName]= gridsearcher.best_estimator_.estimator.get_params()
    gridsearchers[estimatorName] = gridsearcher
    save_obj(gridsearcher.best_params_, 'GridSearch_Best_Params_' + estimatorName)
    save_obj(gridsearcher.best_estimator_.estimator.get_params(), 'Params_' + estimatorName)

    print(gridsearcher.best_params_)

    return gridsearcher

# """#### LogisticRegression"""
# param_grid_lr = [
#     {'estimator__C': [0.01, 0.1, 1, 5, 10],
#      'estimator__penalty': ['l2', 'l1'],
#      'estimator__solver' : ['liblinear'],
#      'estimator__multi_class' : ['ovr'],
#      'estimator__class_weight': [None, 'balanced'],
#      'estimator__max_iter': [1000]}]
#
# param_grid_lr = [
#     {'estimator__C': [1],
#      'estimator__penalty': ['l2'],
#      'estimator__solver' : ['liblinear'],
#      'estimator__multi_class' : ['ovr'],
#      'estimator__class_weight': [None],
#      'estimator__max_iter': [1000]}]
#
# lr = LogisticRegression(random_state = SEED)
# gridsearcher_lr = gridsearch_rfecv_estimator(estimator = lr, param_grid = param_grid_lr)
# load_obj('GridSearch_Best_Params_LogisticRegression')
#
# lr = LogisticRegression(**estimators_params['LogisticRegression'])
# lr = train_clf(lr, x_train, x_test)
#
# """#### DecisionTreeClassifier"""
# tree = DecisionTreeClassifier(random_state = SEED)
# param_grid_tree = {'estimator__min_samples_split' : [10, 30, 50, 100, 150, 200, 300, 400, 500],
#                    'estimator__max_depth': [4,8,12],
#                    'estimator__class_weight': ['balanced', None]}
# param_grid_tree = {'estimator__min_samples_split' : [400],
#                    'estimator__max_depth': [2],
#                    'estimator__class_weight': [None]}
# gridsearcher_tree = gridsearch_rfecv_estimator(estimator = tree, param_grid = param_grid_tree)
# load_obj('GridSearch_Best_Params_DecisionTreeClassifier')
# gridsearcher_tree.best_estimator_.estimator_
#
# dtree = DecisionTreeClassifier(**estimators_params['DecisionTreeClassifier'])
# dtree = train_clf(dtree, x_train, x_test)
#
# # Plot tree
# import graphviz
# from sklearn import tree
# dot_data = tree.export_graphviz(dtree, out_file=None,
#                                 feature_names = [x_train.columns[i] for i in range(0, len(x_train.columns)) if estimators_RFECV['DecisionTreeClassifier'].support_[i]],
#                                 class_names = True,
#                                 filled = True, rounded = True)
# graph = graphviz.Source(dot_data, format="png")
# save_path = os.path.join(IMAGE_DIRECTORY, 'tree', '1')
# graph.render(save_path)
#
#
# """#### RandomForestClassifier"""
# rfc = RandomForestClassifier(random_state = SEED)
# param_grid_rfc = {'estimator__min_samples_split' : [10, 30, 50, 100, 200],
#                   'estimator__n_estimators': [25, 100, 200]}
# param_grid_rfc = {'estimator__min_samples_split' : [100],
#                   'estimator__n_estimators': [50]}
# gridsearcher_rfc = gridsearch_rfecv_estimator(estimator = rfc, param_grid = param_grid_rfc)
# load_obj('GridSearch_Best_Params_RandomForestClassifier')
#
# rfc = RandomForestClassifier(**estimators_params['RandomForestClassifier'])
# rfc = train_clf(rfc, x_train, x_test)
#
#
# """#### XGBClassifier"""
# xgb = XGBClassifier(seed=SEED)
# param_grid_xgb = {'estimator__earning_rate': [.1, .3, .5], #default: .3
#                   'estimator__max_depth': [2,4, 6,10], #default 2
#                   'estimator__n_estimators': [10, 25, 50, 100],
#                   'estimator__subsample': [0.5],
#                   'estimator__colsample_bytree': [0.5],
#                   'estimator__objective': ['multi:softprob'],}
# param_grid_xgb = {'estimator__earning_rate': [.3], #default: .3
#                   'estimator__max_depth': [4], #default 2
#                   'estimator__n_estimators': [25],
#                   'estimator__subsample': [0.5],
#                   'estimator__colsample_bytree': [0.5],
#                   'estimator__objective': ['multi:softprob'],}
# grid_search_xgb = gridsearch_rfecv_estimator(estimator = xgb, param_grid = param_grid_xgb)
# load_obj('GridSearch_Best_Params_XGBClassifier')
#
# xgb = XGBClassifier(**estimators_params['XGBClassifier'])
# xgb = train_clf(xgb, x_train, x_test)


"""#### Neural Network"""

# Neural Network
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

numpy.random.seed(SEED)
x_train_values = x_train.values
x_test_values = x_test.values
y_train_cat = np_utils.to_categorical(y_train)

from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced',np.unique(y_train)
                                               ,y_train)

def get_model():
  # create model
  model = Sequential()
  model.add(Dense(64, input_dim=x_train_values.shape[1], activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(12, activation='softmax'))
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
                    verbose=2, validation_split=0.3, shuffle=True, class_weight = class_weight)

run_total = time.perf_counter() - start_total
print(current_time() + ': Finish training ' + 'NeuralNetwork')
run_times_df['NeuralNetwork'] = [run_total]


# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.figure(figsize=(8,6))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['train', 'test'], loc='bottom right', prop={'size': 18})
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim(0.5, 0.7)
plt.savefig('NeuralNetwork_Accuracy_History')

# summarize history for loss
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim(1.5, 0.8)
plt.legend(['train', 'test'], loc='bottom right', prop={'size': 18})
plt.savefig('NeuralNetwork_Loss_History')

# model.save(os.path.join(MODEL_PATH, 'nn_2.h5')) # save model

clf_name = 'NeuralNetwork'

y_train_prob = model.predict(x_train)
y_test_prob = model.predict(x_test)
check_dir(TRAIN_RESULT_PATH)
(pd.DataFrame(y_train_prob)).to_csv(os.path.join(TRAIN_RESULT_PATH,"Prob_Train_{}.csv").format(clf_name), index = False)
(pd.DataFrame(y_test_prob)).to_csv(os.path.join(TRAIN_RESULT_PATH,"Prob_Test_{}.csv").format(clf_name), index = False)

y_train_pred = y_train_prob.argmax(axis=-1)
train_result_df['y_train_pred_' + clf_name] = y_train_pred
y_test_pred = y_test_prob.argmax(axis=-1)
test_result_df['y_test_pred_' + clf_name] = y_test_pred

accuracy_scores = []
accuracy_scores.append(accuracy_score(y_train_pred, y_train))
accuracy_scores.append(accuracy_score(y_test_pred, y_test))

score_df = pd.DataFrame(data = {'Prediction': ['Train Prediction', 'Test Prediction'], 'Accuracy Score': accuracy_scores})
check_dir(TRAIN_RESULT_PATH)
score_df.to_csv(os.path.join(TRAIN_RESULT_PATH,'Scores_for_{}.csv').format(clf_name), index = False)
get_matrix(y_test = y_test, y_test_pred = y_test_pred,
            y_train = y_train, y_train_pred = y_train_pred,
            estimator_name = clf_name, label_encoder = label_encoder)
print(current_time() + ': Finish getting confusion matrix for ' + clf_name)

# train_result_df.to_csv(os.path.join(TRAIN_RESULT_PATH, 'Predicts_Train.csv'), index = False)
# test_result_df.to_csv(os.path.join(TRAIN_RESULT_PATH, 'Predicts_Test.csv'), index = False)
# run_times_df.to_csv(os.path.join(TRAIN_RESULT_PATH, 'Run_Time.csv'), index = False)
# train_result_df
# test_result_df
# run_times_df

"""#### Model Comparison"""
# Runtime
run_times_df = pd.read_csv(os.path.join(TRAIN_RESULT_PATH, 'Run_Time.csv'))

fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
plt.plot(run_times_df.transpose().rename(columns = {0: 'Run Time (sec)'}))
plt.ylabel('Run Time (sec)')
plt.xlabel('Model')
title = 'Runtime Across Models'
plt.title(title, loc = 'center', y=1.1, fontsize = 20)
saved_path = os.path.join(IMAGE_GENERAL_DIRECTORY, convert_title_to_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")

# Accuracy
score_filenames = [file for file in os.listdir(TRAIN_RESULT_PATH) if file.startswith('Scores_for_')]
score_dict = {}
for filename in score_filenames:
    estimatorName = filename.replace('Scores_for_', '').replace('.csv', '')
    score_df = pd.read_csv(os.path.join(TRAIN_RESULT_PATH, filename))
    score_df = score_df.rename(columns = {'Accuracy Score': estimatorName})
    score_dict[estimatorName] = score_df
scores_df = pd.concat([score_dict['LogisticRegression'],
                       score_dict['DecisionTreeClassifier'],
                       score_dict['RandomForestClassifier'],
                       score_dict['XGBClassifier'],
                       score_dict['NeuralNetwork']], axis = 1)
scores_df = scores_df.drop(columns = 'Prediction')
train_score_df = scores_df.iloc[:1, :]
test_score_df = scores_df.iloc[1:, :]

fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
plt.plot(train_score_df.transpose().rename(columns = {0: 'Accuracy (Train)'}), label='Accuracy (Train)')
plt.plot(test_score_df.transpose().rename(columns = {0: 'Accuracy (Test)'}), label='Accuracy (Test)')
plt.ylim(0.6, 0.7)
plt.ylabel('Accuracy %')
plt.xlabel('Model')
plt.legend()
title = 'Accuracy Scores Across Models'
plt.title(title, loc = 'center', y=1.1, fontsize = 20)
saved_path = os.path.join(IMAGE_GENERAL_DIRECTORY, convert_title_to_filename(title))
fig.savefig(saved_path, dpi=200, bbox_inches="tight")
