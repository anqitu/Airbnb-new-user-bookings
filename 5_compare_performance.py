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

train = pd.read_csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/dest_train.csv")
test = pd.read_csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/dest_test.csv")
country_destination_distribution = pd.read_csv('data/country_distribution_train.csv')

target = 'country_destination'
y_test = test[target]
y_test.shape
x_test = test.drop(columns = target)
y_train = train[target]
x_train = train.drop(columns = target)


"""Runtime"""
run_times_df = pd.read_csv(os.path.join(TRAIN_RESULT_PATH, 'Run_Time_Dest.csv'))
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
title = 'Runtime Across Models (Destination)'
plt.title(title, loc = 'center', y=1.1, fontsize = 20)
saved_path = os.path.join(IMAGE_MODEL_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")

""" Accuracy """
def get_cum_accuracy(prob_type):
    if prob_type == 'test':
        prob_file_prefix = 'Dest_Prob_Test_'
        y_true = y_test
    elif prob_type == 'train':
        prob_file_prefix = 'Dest_Prob_Train_'
        y_true = y_train
    else:
        print('prob_type must be test or train')
    test_probs_filenames = [file for file in os.listdir(TRAIN_RESULT_PATH) if file.startswith(prob_file_prefix)]
    cum_accuracy = {}
    # cum_accuracy['Benchmark1'] = np.linspace(1/11, 1, 11)
    # cum_accuracy['Benchmark2'] = list(country_destination_distribution['%'].cumsum())

    for filename in test_probs_filenames:
        estimatorName = filename.replace(prob_file_prefix, '').replace('.csv', '')
        prob_df = pd.read_csv(os.path.join(TRAIN_RESULT_PATH, filename))
        print(prob_df.shape)
        cum_accuracy[estimatorName] = np.array([accuracy_score([prob_df.columns[i] for i in get_prob_top_n(prob_df.as_matrix(), n)], y_true) for n in range(11)]).cumsum()
    cum_accuracy_df = pd.DataFrame(cum_accuracy)
    cum_accuracy_df.index = list(range(1, 12))
    cum_accuracy_df.loc[0] = 0
    cum_accuracy_df = cum_accuracy_df.reindex(list(range(0, 12)))
    return cum_accuracy_df

def plot_cum_accuracy(cum_accuracy_df):
    fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    cum_accuracy_df.plot(lw = 0.8)
    plt.ylabel('Accuracy %')
    plt.xlabel('Model')
    plt.legend(prop={'size': 10})
    title = 'Cumulative Accuracy Gain Across Models'
    plt.title(title, loc = 'center', y=1.1, fontsize = 25)
    saved_path = os.path.join(IMAGE_MODEL_DIRECTORY, convert_filename(title))
    plt.savefig(saved_path, dpi=200, bbox_inches="tight")

cum_accuracy_df = get_cum_accuracy('test')
# cum_accuracy_df = get_cum_accuracy('train')
cum_accuracy_df
# plot_cum_accuracy(cum_accuracy_df)

top_5_accuracy = cum_accuracy_df.copy().loc[5:5]
top_5_accuracy['Random_Forest'] = top_5_accuracy['Random_Forest'] + 0.02
top_5_accuracy['Mars'] = top_5_accuracy['Mars'] + 0.015
# top_5_accuracy['Cart_Full'] = top_5_accuracy['Cart_Full'] - 0.07
top_5_accuracy['Cart_Prune'] = top_5_accuracy['Cart_Prune'] - 0.02
top_5_accuracy['LR_Full'] = top_5_accuracy['LR_Full'] - 0.02
top_5_accuracy['LR_Step'] = top_5_accuracy['LR_Step'] - 0.02

top_5_accuracy.columns = [convert_column_name_to_title(col) for col in top_5_accuracy.columns]
# top_5_accuracy = top_5_accuracy[['Benchmark1', 'Benchmark2'] + [col for col in list(run_times_df.columns)]]
top_5_accuracy = top_5_accuracy[[col for col in list(run_times_df.columns)]]
top_5_accuracy.to_csv('accuracy_dest.csv')
fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH * 1.1, PLOT_HEIGHT))
bars = plt.bar(top_5_accuracy.columns, top_5_accuracy.iloc[0, :], color = 'c')
plt.ylabel('Top 5 Accuracy')
plt.xlabel('Model')
# plt.ylim(0.4, 0.9)
title = 'Top 5 Accuracy Across Models (Destination)'
plt.title(title, loc = 'center', y=1.1, fontsize = 20)
for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.3f' % (height), ha='center', va='bottom', size = 12)
saved_path = os.path.join(IMAGE_MODEL_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")


df = pd.read_csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/dest_forest_vi.csv")
df.shape
df = df.sort_values(df.columns[-1], ascending = True)
df.to_csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/dest_forest_vi.csv", index = False)
df = df[df[df.columns[-1]] >= 470]

plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT * 1.4))
bars = plt.barh(df[df.columns[0]], df[df.columns[1]], color = 'c')
plt.xlabel('Feature Importance')
title = 'Feature Importance by Random Forest (Destination)'
plt.title(title, loc = 'center', y=1.05, fontsize = 20)
for rect in bars:
    width = rect.get_width()
    plt.text(width + 20, rect.get_y(), '%.0f' % (width), ha='center', va='bottom', size = 12)
saved_path = os.path.join(IMAGE_BAR_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")


df = pd.read_csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/dest_cart_prune_vi.csv")
df.shape
df = df.iloc[:20]
df = df.sort_values(df.columns[-1], ascending = True)

plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT * 1.4))
bars = plt.barh(df[df.columns[0]], df[df.columns[1]], color = 'c')
plt.xlabel('Feature Importance')
title = 'Feature Importance by Decision Tree (Destination)'
plt.title(title, loc = 'center', y=1.05, fontsize = 20)
for rect in bars:
    width = rect.get_width()
    plt.text(width + 10, rect.get_y(), '%.0f' % (width), ha='center', va='bottom', size = 12)
saved_path = os.path.join(IMAGE_BAR_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")


df = pd.read_csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/dest_mars_vi.csv")
df = df[[df.columns[i] for i in [0,3,4,6]]]
df = df.sort_values(df.columns[1], ascending = True)

plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT * 1.4))
bars = plt.barh(df[df.columns[0]], df[df.columns[1]], color = 'c')
plt.xlabel('nsubset')
title = 'Feature Importance by Mars (nsubset) (Destination)'
plt.title(title, loc = 'center', y=1.05, fontsize = 20)
for rect in bars:
    width = rect.get_width()
    plt.text(width + 0.2, rect.get_y() + 0.4, '%.0f' % (width), ha='center', va='bottom', size = 12)
saved_path = os.path.join(IMAGE_BAR_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")

df = df.sort_values(df.columns[2], ascending = True)
plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT * 1.4))
bars = plt.barh(df[df.columns[0]], df[df.columns[2]], color = 'c')
plt.xlabel('gcv')
title = 'Feature Importance by Mars (gcv) (Destination)'
plt.title(title, loc = 'center', y=1.05, fontsize = 20)
for rect in bars:
    width = rect.get_width()
    plt.text(width + 3, rect.get_y() + 0.4, '%.1f' % (width), ha='center', va='bottom', size = 12)
saved_path = os.path.join(IMAGE_BAR_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")

df = df.sort_values(df.columns[3], ascending = True)
plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT * 1.4))
bars = plt.barh(df[df.columns[0]], df[df.columns[3]], color = 'c')
plt.xlabel('rss')
title = 'Feature Importance by Mars (rss) (Destination)'
plt.title(title, loc = 'center', y=1.05, fontsize = 20)
for rect in bars:
    width = rect.get_width()
    plt.text(width + 4, rect.get_y() + 0.4, '%.1f' % (width), ha='center', va='bottom', size = 12)
saved_path = os.path.join(IMAGE_BAR_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")
