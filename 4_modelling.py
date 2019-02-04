"""Modelling
1. Feature Importance by RFECV
2. Feature Selection using sklearn.feature_selection.RFECV on train
3. Fine tune model, using cross validation. —> must sampled after splitting
    1. model_original on train, using GridSeachCV
    2. model_sampled on sampled train, using ParamSearch, each with validation before sampled.
4. Meta holdout scheme with OOF meta-features.
    1. Fit models to whole train_data and predict for test_data.
       Let's call these features test_meta --> Save scores for each model's prediction
    2. Split train into K folds. Iterate through each fold:
       - retain N diverme models on all folds except current fold.
       - predict for the current fold
       After this step, for each object in train_data,
       we will have N meta-features (also known as out-of-fold predictions, OOF).
       Lets call them train_meta
    3. Split train_meta into two parts: train_metaA and train_metaB.
       Fit a meta-model to train_metaA while validating its hyperparameters on train_metaB.
    4. When the meta-model is validated, fit it to train_meta and predict for test_meta --> Save scores for meta-model's prediction
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
train = train.sample(5000)
test = pd.read_csv(TEST_PATH)
trainA = pd.read_csv(TRAINA_PATH)
trainB = pd.read_csv(TRAINB_PATH)
trainC = pd.read_csv(TRAINC_PATH)

"""#### Useful Functions"""
from util import *

from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title=None, cmap=plt.cm.Blues, save = False, show = True):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
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

def get_matrix(y_test, y_test_pred, y_train, y_train_pred, estimator_name, label_encoder, transformed = ''):
    check(TRAIN_RESULT_PATH)

    title = 'Confusion Matrix for ' + estimator_name + ' Test'
    if transformed != '':
        title += ' (Transformed)'
    df_confusion = pd.crosstab(pd.Series(label_encoder.inverse_transform(y_test), name='True'), pd.Series(label_encoder.inverse_transform(y_test_pred), name='Predict'))
    df_confusion.to_csv(os.path.join(TRAIN_RESULT_PATH, path + '.csv'))
    plot_confusion_matrix(confusion_matrix(y_test, y_test_pred), label_encoder.classes_, title = title, save = True)

    title = 'Confusion Matrix for ' + estimator_name  + ' Train'
    if transformed != '':
        title += ' (Transformed)'
    df_confusion = pd.crosstab(pd.Series(label_encoder.inverse_transform(y_train), name='True'), pd.Series(label_encoder.inverse_transform(y_train_pred), name='Predict'))
    df_confusion.to_csv(os.path.join(TRAIN_RESULT_PATH, path + '.csv'))
    plot_confusion_matrix(confusion_matrix(y_train, y_train_pred), label_encoder.classes_, title = title, save = True)

# Plot number of features VS. cross-validation scores
def plot_estimator_no_feature_vs_accuracy_score(rfecv_result, estimator, title = None, save = False, show = True):
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (Accuracy)")
    plt.plot(range(1, len(rfecv_result.grid_scores_) + 1), rfecv_result.grid_scores_)
    # plt.ylim([0.56,0.64])

    if title is None:
        title = 'Number of Features vs Accuracy Score for ' + estimator.__class__.__name__
    plt.title(title, loc = 'center', y=1.1, fontsize = 25)

    if save:
        check_dir(IMAGE_BIN_DIRECTORY)
        saved_path = os.path.join(IMAGE_BIN_DIRECTORY, convert_title_to_filename(title))
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()

def train_clf(clf, x_train, x_test, transformed = ''):
    check_dir(TRAIN_RESULT_PATH)

    clf_name = clf.__class__.__name__

    print(current_time() + ': Start training ' + clf_name)
    start_total = time.perf_counter()
    clf.fit(x_train, y_train)
    run_total = time.perf_counter() - start_total
    print(current_time() + ': Finish training ' + clf_name)
    run_times_df[clf_name + transformed] = [run_total]

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    train_result_df['y_train_pred_' + clf_name + transformed] = y_train_pred
    test_result_df['y_test_pred_' + clf_name + transformed] = y_test_pred

    accuracy_scores = []
    accuracy_scores.append(accuracy_score(y_train_pred, y_train))
    accuracy_scores.append(accuracy_score(y_test_pred, y_test))
    pd.DataFrame(clf.predict_proba(x_train)).to_csv(os.path.join(TRAIN_RESULT_PATH,"Prob_Train_{}{}.csv").format(clf_name, transformed), index = False)
    pd.DataFrame(clf.predict_proba(x_test)).to_csv(os.path.join(TRAIN_RESULT_PATH,"Prob_Test_{}{}.csv").format(clf_name, transformed), index = False)

    score_df = pd.DataFrame(data = {'Prediction': ['Train Predction', 'Test Prediction'], 'Accuracy Score': accuracy_scores})
    score_df.to_csv(os.path.join(TRAIN_RESULT_PATH,'Scores_for_{}{}.csv'.format(clf_name, transformed)), index = False)
    get_matrix(y_test = y_test, y_test_pred = y_test_pred,
                y_train = y_train, y_train_pred = y_train_pred,
                estimator_name = clf_name, label_encoder = label_encoder, transformed = transformed)

    print(current_time() + ': Finish getting confusion matrix for ' + clf_name)

    save_model(clf, clf.__class__.__name__ + transformed)

    return clf

def train_clf_transformed(clf):
    x_train_transformed = estimators_RFECV[clf.__class__.__name__].transform(x_train)
    x_test_transformed = estimators_RFECV[clf.__class__.__name__].transform(x_test)
    train_clf(clf, x_train_transformed, x_test_transformed, transformed = '_transformed')

    return clf

def get_feature_importance_by_tree(clf):
    check_dir(TRAIN_RESULT_PATH)
    feature_importance = pd.DataFrame(data = {'feature': x_train.columns, 'feature_importance': clf.feature_importances_})
    feature_importance.sort_values(['feature_importance'], ascending = False)
    feature_importance.to_csv(os.path.join(TRAIN_RESULT_PATH,'Feature_Importance_{}.csv').format(clf.__class__.__name__), index = False)

    return feature_importance

"""#### X and Y """
target = 'country_destination'
y_train = train[target]
y_test = train[target]
y_trainA = trainA[target]
y_trainB = trainB[target]
y_trainC = trainC[target]

x_train = train.drop(columns = target)
x_test = train.drop(columns = target)
x_trainA = trainA.drop(columns = target)
x_trainB = trainB.drop(columns = target)
x_trainC = trainC.drop(columns = target)

train_result_df = pd.DataFrame(data = {'y_train': y_train})
test_result_df = pd.DataFrame(data = {'y_test': y_test})
run_times_df = pd.DataFrame()

"""#### 1. Feature Importance by RFECV """
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.feature_selection import RFECV, RFE

# # RFECV: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
# import time
# start_total = time.perf_counter()
#
# estimators = {clf.__class__.__name__: clf for clf in [LogisticRegression(random_state = SEED),
#                                                     DecisionTreeClassifier(random_state = SEED),
#                                                     RandomForestClassifier(random_state = SEED),
#                                                     # XGBClassifier(seed = SEED)
#                                                     ]}
# estimators_RFECV = {}
# for estimatorName, estimator in estimators.items():
#     print(current_time() + ': Start training ' + estimatorName)
#     start = time.perf_counter()
#     rfecv = RFECV(estimator = estimator, step = 1, scoring = 'accuracy', cv = 5, n_jobs = -1)
#     rfecv_result = rfecv.fit(x_train, y_train)
#     run = time.perf_counter() - start
#
#     print('{} runs for {:.2f} seconds.'.format(estimator.__class__.__name__, run))
#     estimators_RFECV[estimatorName] = rfecv_result
#
#     rfe_result_rank = pd.DataFrame(data = {'Ranking': rfecv_result.ranking_, 'Column': x_train.columns}).sort_values('Ranking')
#     check_dir(TRAIN_RESULT_PATH)
#     rfe_result_rank.to_csv(os.path.join(TRAIN_RESULT_PATH,'RFECV_Ranking_for_{}.csv').format(estimator.__class__.__name__), index = False)
#
#     plot_estimator_no_feature_vs_accuracy_score(rfecv_result, estimator)
#
# run_total = time.perf_counter() - start_total
# print('Total running time was {:.2f} minutes.'.format(run_total/60))

"""#### 2. Feature Selection by RFECV """



"""
3. Fine tune model, using cross validation.
- model_original on train, using GridSeachCV
"""
from sklearn.model_selection import GridSearchCV
best_params = {}
grid_search_cv_param = {'cv': 3,
                        'scoring': 'accuracy',
                        'n_jobs': -1,
                        'verbose': 2}

"""#### LogisticRegression"""
c_values = list(np.arange(1, 10, 2))
param_grid_lr = [
    {'C': c_values, 'penalty': ['l1'], 'solver' : ['liblinear'], 'multi_class' : ['ovr']},
    {'C': c_values, 'penalty': ['l2'], 'solver' : ['liblinear', 'newton-cg', 'lbfgs'], 'multi_class' : ['ovr']}]

grid_search_lr = GridSearchCV(LogisticRegression(random_state = SEED), param_grid_lr, **grid_search_cv_param)
grid.fit(x_train, y_train)

"""#### DecisionTreeClassifier"""
min_samples_split = [10, 30, 50, 100, 200, 300]
param_grid_tree = {'min_samples_split' : min_samples_split}
grid_search_tree = GridSearchCV(DecisionTreeClassifier(random_state = SEED), param_grid_tree, **grid_search_cv_param)
grid_search_tree.fit(x_train, y_train)
for score in grid_search_tree.grid_scores_:
    print(score)
print(grid_search_tree.best_params_)
tree = grid_search_tree.best_estimator_
print(tree.get_params)
get_feature_importance_by_tree(tree)

tree = DecisionTreeClassifier(random_state = SEED, **grid_search_tree.best_params_)
"""#### RandomForestClassifier"""

from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier(random_state = SEED)

n_estimators = [100, 300]
min_samples_split = [30, 50, 100, 200]
param_grid_rfc = {'min_samples_split' : min_samples_split, 'n_estimators': n_estimators}
grid_search_rfc = GridSearchCV(rfc, param_grid_rfc, cv = 5, **grid_search_cv_param)
grid_search_rfc.fit(x_train, y_train)
for score in grid_search_rfc.grid_scores_:
    print(score)
print(grid_search_rfc.best_params_)
rfc = grid_search_rfc.best_estimator_
print(rfc.get_params)
get_feature_importance_by_tree(rfc)

rfc = RandomForestClassifier(random_state = SEED, **grid_search_rfc.best_params_)

"""#### XGBClassifier"""

xgb = XGBClassifier(max_depth= 6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=SEED)


"""

"""

"""#### LogisticRegression"""

lr = LogisticRegression(random_state = SEED)
lr = train_clf(lr, x_train, x_test)

lr_t = LogisticRegression(random_state = SEED)
lr_t = train_clf_transformed(lr_t)

"""#### DecisionTreeClassifier"""

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
tree = train_clf(tree, x_train, x_test)

tree_t = DecisionTreeClassifier(random_state = SEED, **grid_search_tree.best_params_)
tree_t = train_clf_transformed(tree_t)

"""#### RandomForestClassifier"""

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
rfc = train_clf(rfc, x_train, x_test)

rfc_t = RandomForestClassifier(random_state = SEED, **grid_search_rfc.best_params_)
rfc_t = train_clf_transformed(rfc_t)

"""#### XGBClassifier"""

xgb = XGBClassifier(max_depth= 6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=SEED)
xgb_t = XGBClassifier(max_depth= 6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=SEED)

xgb = train_clf(xgb, x_train, x_test)
xgb_t = train_clf_transformed(xgb_t)

"""#### Neural Network"""

# Neural Network
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

numpy.random.seed(SEED)
x_train_values = x_train.values

y_train_cat = np_utils.to_categorical(y_train)

x_test_values = x_test.values

def get_model():
  # create model
  model = Sequential()
  model.add(Dense(128, input_dim=x_train_values.shape[1], activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(12, activation='softmax'))
  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

# RMSProp

print(current_time() + ': Start training ' + 'NeuralNetwork')
start_total = time.perf_counter()

model = get_model()
model.fit(x_train_values, y_train_cat, epochs=100, batch_size=32, verbose=2)

run_total = time.perf_counter() - start_total
print(current_time() + ': Finish training ' + 'NeuralNetwork')
run_times_df['NeuralNetwork'] = [run_total]

# model.save('nn_1.h5') # save model

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

score_df = pd.DataFrame(data = {'Prediction': ['Train Predction', 'Test Prediction'], 'Accuracy Score': accuracy_scores})
check_dir(TRAIN_RESULT_PATH)
score_df.to_csv(os.path.join(TRAIN_RESULT_PATH,'Scores_for_{}.csv').format(clf_name), index = False)
get_matrix(y_test = y_test, y_test_pred = y_test_pred,
            y_train = y_train, y_train_pred = y_train_pred,
            estimator_name = clf_name, label_encoder = label_encoder)

print(current_time() + ': Finish getting confusion matrix for ' + clf_name)

# """#### Ensembling"""
#
# test_prob_files = [file for file in os.listdir(WORKING_DIR) if file.startswith('Prob_Test_') and 'transformed' not in file]
# prob_dfs = []
# for prob_file in test_prob_files:
#     df = pd.read_csv(prob_file)
#     model_name = prob_file.replace('Prob_Test_', '').replace('.csv', '')
#     df.columns = [model_name + '_' + str(i) for i in range(12)]
#     prob_dfs.append(df)
#
# x_test_meta = pd.concat(prob_dfs, axis = 1)
# y_test_meta = y_test
#
# xgb_stacking = XGBClassifier(max_depth= 6, learning_rate=0.3, n_estimators=25,
#                             objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=SEED)
# xgb_stacking.fit(x_test_meta, y_test_meta)
#
# y_test_pred_meta = xgb_stacking.predict(x_test_meta)
# print(accuracy_score(y_test_meta, y_test_pred_meta))
# save_model(xgb_stacking, 'XGBClassifier_Stacking')
