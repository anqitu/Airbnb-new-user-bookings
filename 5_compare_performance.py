"""Compare performance
- for every single model
- oversampled
- Cumulative Gain by Top 1/2/3/4.....
"""

""" #### Environment Setup """
# import libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
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

label_encoder = load_label_encoder('label_encoder_country_destination')

test = pd.read_csv(TEST_PATH)
train_raw = pd.read_csv(TRAIN_RAW_PATH)
val_raw = pd.read_csv(VAL_RAW_PATH)
train_smote = pd.read_csv(TRAIN_SMOTE_PATH)
val_smote = pd.read_csv(VAL_SMOTE_PATH)

train_raw = pd.concat([train_raw, val_raw])
train_smote = pd.concat([train_smote, val_smote])

target = 'country_destination'
y_test = test[target]
x_test = test.drop(columns = target)
y_train_raw = train_raw[target]
x_train_raw = train_raw.drop(columns = target)
y_train_smote = train_smote[target]
x_train_smote = train_smote.drop(columns = target)
y_val_smote = val_smote[target]
x_val_smote = val_smote.drop(columns = target)

# """Runtime"""
# run_times_df = pd.read_csv(os.path.join(TRAIN_RESULT_PATH, 'Run_Time.csv'))
#
# fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
# plt.bar(run_times_df.columns, run_times_df.iloc[0, :])
# plt.ylabel('Run Time (sec)')
# plt.xlabel('Model')
# title = 'Runtime Across Models'
# plt.title(title, loc = 'center', y=1.1, fontsize = 20)
# saved_path = os.path.join(IMAGE_MODEL_DIRECTORY, convert_title_to_filename(title))
# plt.savefig(saved_path, dpi=200, bbox_inches="tight")


""" Accuracy """
# test_probs_filenames = [file for file in os.listdir(TRAIN_RESULT_PATH) if file.startswith('Prob_Test_')]
# cum_accuracy = {}
# y_test_df = pd.DataFrame(y_test)
# cum_accuracy['Benchmark'] = get_percentage(y_test_df, target)['%'].cumsum()
#
# for filename in test_probs_filenames:
#     estimatorName = filename.replace('Prob_Test_', '').replace('.npy', '')
#     prob_df = np.load(os.path.join(TRAIN_RESULT_PATH, filename))
#     cum_accuracy[estimatorName] = np.array([accuracy_score(get_prob_top_n(prob_df, n), y_test) for n in range(11)]).cumsum()

train_probs_filenames = [file for file in os.listdir(TRAIN_RESULT_PATH) if file.startswith('Prob_Train_') and 'Smote' not in file]
cum_accuracy = {}
y_train_df = pd.DataFrame(y_train_raw)
cum_accuracy['Benchmark'] = get_percentage(y_train_df, target)['%'].cumsum()
# cum_accuracy['Benchmark'] = np.linspace(0, 1, 11, endpoint=False).cumsum()

for filename in train_probs_filenames:
    estimatorName = filename.replace('Prob_Train_', '').replace('.npy', '')
    prob_df = np.load(os.path.join(TRAIN_RESULT_PATH, filename))
    cum_accuracy[estimatorName] = np.array([accuracy_score(get_prob_top_n(prob_df, n), y_train_raw) for n in range(11)]).cumsum()

cum_accuracy_df = pd.DataFrame(cum_accuracy)
cum_accuracy_df.index = list(range(1, 12))
fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
cum_accuracy_df.plot(lw = 0.8, figsize=(15,10))
plt.ylabel('Accuracy %')
plt.xlabel('Model')
# plt.ylim(0, 1)
plt.legend(prop={'size': 20})
title = 'Cumulative Accuracy Gain Across Models'
plt.title(title, loc = 'center', y=1.1, fontsize = 25)
saved_path = os.path.join(IMAGE_MODEL_DIRECTORY, convert_title_to_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")


""" Accuracy (Sampled) """
test_probs_filenames = [file for file in os.listdir(TRAIN_RESULT_PATH) if file.startswith('Prob_Test_') and 'Smote' in file]
cum_accuracy = {}
y_test_df = pd.DataFrame(y_test)
cum_accuracy['Benchmark'] = get_percentage(y_test_df, target)['%'].cumsum()

for filename in test_probs_filenames:
    estimatorName = filename.replace('Prob_Test_', '').replace('.npy', '')
    prob_df = np.load(os.path.join(TRAIN_RESULT_PATH, filename))
    cum_accuracy[estimatorName] = np.array([accuracy_score(get_prob_top_n(prob_df, n), y_test) for n in range(11)]).cumsum()

# train_probs_filenames = [file for file in os.listdir(TRAIN_RESULT_PATH) if file.startswith('Prob_Train_') and 'Smote' in file]
# cum_accuracy = {}
# y_train_df = pd.DataFrame(y_train_smote)
# cum_accuracy['Benchmark'] = get_percentage(y_train_df, target)['%'].cumsum()
#
# for filename in train_probs_filenames:
#     estimatorName = filename.replace('Prob_Train_', '').replace('.npy', '')
#     prob_df = np.load(os.path.join(TRAIN_RESULT_PATH, filename))
#     cum_accuracy[estimatorName] = np.array([accuracy_score(get_prob_top_n(prob_df, n), y_train_smote) for n in range(11)]).cumsum()

cum_accuracy_df = pd.DataFrame(cum_accuracy)
cum_accuracy_df.index = list(range(1, 12))
fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
cum_accuracy_df.plot(lw = 0.8, figsize=(15,10))
plt.ylabel('Accuracy %')
plt.xlabel('Model')
plt.xlim(1, 11)
plt.ylim(0, 1)
plt.legend(prop={'size': 20}, loc = 'lower right')
title = 'Cumulative Accuracy Gain Across Models (Sampled)'
plt.title(title, loc = 'center', y=1.1, fontsize = 25)
saved_path = os.path.join(IMAGE_MODEL_DIRECTORY, convert_title_to_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")
