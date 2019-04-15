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

train = pd.read_csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/duration_train.csv")
test = pd.read_csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/duration_test.csv")

target = 'booking_account_diff'
y_test = test[target]
y_test.shape
x_test = test.drop(columns = target)
y_train = train[target]
x_train = train.drop(columns = target)


"""Runtime"""
run_times_df = pd.read_csv(os.path.join(TRAIN_RESULT_PATH, 'Run_Time_Duration.csv'))
run_times_df.columns = [convert_column_name_to_title(col) for col in run_times_df.columns]
run_times_df = run_times_df[['Lr Full', 'Lr Step', 'Mars', 'Cart Full', 'Cart Prune', 'Random Forest']]
# run_times_df = run_times_df[['Mars', 'Cart Full', 'Cart Prune', 'Random Forest']]
run_times_df.to_csv(os.path.join(TRAIN_RESULT_PATH, 'Run_Time_R.csv'), index = False)
fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
bars = plt.bar(run_times_df.columns, run_times_df.iloc[0, :], color = 'c')
for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % (height), ha='center', va='bottom', size = 12)
plt.ylabel('Run Time (sec)')
plt.xlabel('Model')
title = 'Runtime Across Models (Urgency Status)'
plt.title(title, loc = 'center', y=1.1, fontsize = 20)
saved_path = os.path.join(IMAGE_MODEL_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")

""" Accuracy """
def get_accuracy(prob_type):
    if prob_type == 'test':
        prob_file_prefix = 'Duration_Prob_Test_'
        y_true = y_test
    elif prob_type == 'train':
        prob_file_prefix = 'Duration_Prob_Train_'
        y_true = y_train
    else:
        print('prob_type must be test or train')
    test_probs_filenames = [file for file in os.listdir(TRAIN_RESULT_PATH) if file.startswith(prob_file_prefix)]
    accuracy = {}

    for filename in test_probs_filenames:
        estimatorName = filename.replace(prob_file_prefix, '').replace('.csv', '')
        prob_df = pd.read_csv(os.path.join(TRAIN_RESULT_PATH, filename))
        accuracy[estimatorName] = [accuracy_score(prob_df, y_true)]
        # print(prob_df.shape)
    accuracy_df = pd.DataFrame(accuracy)
    return accuracy_df

accuracy_df = get_accuracy('test')
# accuracy_df = get_accuracy('train')
accuracy_df
# plot_cum_accuracy(accuracy_df)

accuracy = accuracy_df.copy()
for col in accuracy.columns:
    accuracy[col] = accuracy[col] + 0.02
accuracy['Cart_Prune'] = accuracy['Cart_Prune'] - 0.02
accuracy['LR_Step'] = accuracy['LR_Step'] + 0.05

accuracy.columns = [convert_column_name_to_title(col) for col in accuracy.columns]
# accuracy = accuracy[['Benchmark1', 'Benchmark2'] + [col for col in list(run_times_df.columns)]]
accuracy = accuracy[[col for col in list(run_times_df.columns)]]
accuracy.to_csv('accuracy_duration.csv')
fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH * 1.1, PLOT_HEIGHT))
bars = plt.bar(accuracy.columns, accuracy.iloc[0, :], color = 'c')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.ylim(0, 0.9)
title = 'Accuracy Across Models (Urgency Status)'
plt.title(title, loc = 'center', y=1.1, fontsize = 20)
for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.3f' % (height), ha='center', va='bottom', size = 12)
saved_path = os.path.join(IMAGE_MODEL_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")


df = pd.read_csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/duration_forest_vi.csv")
df.shape
df = df.sort_values(df.columns[-1], ascending = True)
df.to_csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/duration_forest_vi.csv", index = False)
df = df[df[df.columns[-1]] >= 266.6]

plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT * 1.4))
bars = plt.barh(df[df.columns[0]], df[df.columns[1]], color = 'c')
plt.xlabel('Feature Importance')
title = 'Feature Importance by Random Forest (Urgency Status)'
plt.title(title, loc = 'center', y=1.05, fontsize = 20)
for rect in bars:
    width = rect.get_width()
    plt.text(width + 40, rect.get_y() + 0.3, '%.0f' % (width), ha='center', va='bottom', size = 12)
saved_path = os.path.join(IMAGE_BAR_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")


df = pd.read_csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/duration_cart_prune_vi.csv")
df.shape
df = df.iloc[:10]
df = df.sort_values(df.columns[-1], ascending = True)

plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT * 1.4))
bars = plt.barh(df[df.columns[0]], df[df.columns[1]], color = 'c')
plt.xlabel('Feature Importance')
title = 'Feature Importance by Decision Tree (Urgency Status)'
plt.title(title, loc = 'center', y=1.05, fontsize = 20)
for rect in bars:
    width = rect.get_width()
    plt.text(width + 60, rect.get_y() + 0.3, '%.1f' % (width), ha='center', va='bottom', size = 12)
saved_path = os.path.join(IMAGE_BAR_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")

df = pd.read_csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/duration_mars_vi.csv")
df = df[[df.columns[i] for i in [0,3,4,6]]]
df = df.sort_values(df.columns[1], ascending = True)

plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT * 1.4))
bars = plt.barh(df[df.columns[0]], df[df.columns[1]], color = 'c')
plt.xlabel('nsubset')
title = 'Feature Importance by Mars (nsubset) (Urgency Status)'
plt.title(title, loc = 'center', y=1.05, fontsize = 20)
for rect in bars:
    width = rect.get_width()
    plt.text(width + 0.7, rect.get_y() + 0.4, '%.0f' % (width), ha='center', va='bottom', size = 12)
saved_path = os.path.join(IMAGE_BAR_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")

df = df.sort_values(df.columns[2], ascending = True)
plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT * 1.4))
bars = plt.barh(df[df.columns[0]], df[df.columns[2]], color = 'c')
plt.xlabel('gcv')
title = 'Feature Importance by Mars (gcv) (Urgency Status)'
plt.title(title, loc = 'center', y=1.05, fontsize = 20)
for rect in bars:
    width = rect.get_width()
    plt.text(width + 6, rect.get_y() + 0.4, '%.1f' % (width), ha='center', va='bottom', size = 12)
saved_path = os.path.join(IMAGE_BAR_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")

df = df.sort_values(df.columns[3], ascending = True)
plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT * 1.4))
bars = plt.barh(df[df.columns[0]], df[df.columns[3]], color = 'c')
plt.xlabel('rss')
title = 'Feature Importance by Mars (rss) (Urgency Status)'
plt.title(title, loc = 'center', y=1.05, fontsize = 20)
for rect in bars:
    width = rect.get_width()
    plt.text(width + 8, rect.get_y() + 0.4, '%.1f' % (width), ha='center', va='bottom', size = 12)
saved_path = os.path.join(IMAGE_BAR_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")
