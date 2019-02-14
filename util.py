import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set figure aesthetics
plt.style.use('fivethirtyeight')
PLOT_HEIGHT = 6
PLOT_WIDTH = PLOT_HEIGHT * 1.618
plt.rcParams["figure.figsize"] = [PLOT_WIDTH,PLOT_HEIGHT]

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

WORKING_DIR = '/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings'
# WORKING_DIR = '/content'

SEED = 2019
USERS_PATH = os.path.join(WORKING_DIR, 'data/users.csv')
USERS_WITH_NDF_PATH = os.path.join(WORKING_DIR, 'data/users_with_NDF.csv')
USERS_MIN_TO_OTHERS_PATH = os.path.join(WORKING_DIR, 'data/users_min_to_others.csv')
USERS_WITH_NDF_MIN_TO_OTHERS_PATH = os.path.join(WORKING_DIR, 'data/users_with_NDF_min_to_others.csv')

TEST_PATH = os.path.join(WORKING_DIR, 'data/test.csv')
TRAIN_RAW_PATH = os.path.join(WORKING_DIR, 'data/train_raw.csv')
VAL_RAW_PATH = os.path.join(WORKING_DIR, 'data/val_raw.csv')
TRAIN_SMOTE_PATH = os.path.join(WORKING_DIR, 'data/train_smote.csv')
VAL_SMOTE_PATH = os.path.join(WORKING_DIR, 'data/val_smote.csv')

TRAIN_RAW_MIN_TO_OTHERS_PATH = os.path.join(WORKING_DIR, 'data/train_raw_min_to_others.csv')
VAL_RAW_MIN_TO_OTHERS_PATH = os.path.join(WORKING_DIR, 'data/val_raw_min_to_others.csv')
TRAIN_SMOTE_MIN_TO_OTHERS_PATH =  os.path.join(WORKING_DIR, 'data/train_smote_min_to_others.csv')
VAL_SMOTE_MIN_TO_OTHERS_PATH = os.path.join(WORKING_DIR, 'data/val_smote_min_to_others.csv')

IMAGE_DIRECTORY = os.path.join(WORKING_DIR, 'images')
IMAGE_PIE_DIRECTORY = os.path.join(IMAGE_DIRECTORY, 'pie')
IMAGE_BAR_DIRECTORY = os.path.join(IMAGE_DIRECTORY, 'bar')
IMAGE_BIN_DIRECTORY = os.path.join(IMAGE_DIRECTORY, 'bin')
IMAGE_BOX_DIRECTORY = os.path.join(IMAGE_DIRECTORY, 'box')
IMAGE_BARS_DIRECTORY = os.path.join(IMAGE_DIRECTORY, 'bars')
IMAGE_BUBBLE_DIRECTORY = os.path.join(IMAGE_DIRECTORY, 'bubble')
IMAGE_GENERAL_DIRECTORY = os.path.join(IMAGE_DIRECTORY, 'general')
IMAGE_MATRIX_DIRECTORY = os.path.join(IMAGE_DIRECTORY, 'matrix')
IMAGE_TIME_DIRECTORY = os.path.join(IMAGE_DIRECTORY, 'time')

TRAIN_RESULT_PATH = os.path.join(WORKING_DIR, 'training_result')
MODEL_PATH = os.path.join(WORKING_DIR, 'models')
IMAGE_MODEL_DIRECTORY = os.path.join(IMAGE_DIRECTORY, 'model')

SAMPLED_TRAIN_RESULT_PATH = os.path.join(WORKING_DIR, 'training_result_sampled')
SAMPLED_MODEL_PATH = os.path.join(WORKING_DIR, 'models_sampled')
SAMPLED_IMAGE_MODEL_DIRECTORY = os.path.join(IMAGE_DIRECTORY, 'model_sampled')


"""General"""
def check_dir(directory):
    if not os.path.exists(directory):
        check_dir(os.path.dirname(directory))
        os.mkdir(directory)
        print("{:<6} Make directory: {}".format('[INFO]', directory))

import datetime
def current_time():
    return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def save_obj(obj, name):
    with open(os.path.join(TRAIN_RESULT_PATH, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(os.path.join(TRAIN_RESULT_PATH, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

"""Check Data"""
def display_null_percentage(data):
    df = data.isnull().sum().reset_index().rename(columns = {0: 'Count', 'index': 'Column'})
    df['Frequency'] = df['Count'] / data.shape[0] * 100
    pd.options.display.float_format = '{:.2f}%'.format
    print(df)
    pd.options.display.float_format = None

def display_category_counts(data, categorical_features):
  for categorical_feature in categorical_features:
    print('-' * 30)
    print(categorical_feature)
    print(data[categorical_feature].value_counts(dropna=False))

def get_percentage(data, column):
    count_df = data[column].value_counts().reset_index().rename(columns = {column: 'Count', 'index': column})
    count_df['%'] = count_df['Count'] / data.shape[0]
    return count_df

"""Manipulate Data"""
def convert_minority_to_others(data, column_name, minority_counts = 0):
    minorities = list(data.groupby([column_name]).size().reset_index().rename(columns = {0: 'Count'}).sort_values('Count').head(minority_counts)[column_name])
    data[column_name + '_min_to_other'] = data[column_name].apply(lambda value: 'other' if value in minorities else value)
    print(data[column_name + '_min_to_other'].value_counts())

"""Save & Load Models"""
def save_label_encoder(label_encoder, model_name):
    np.save(os.path.join(MODEL_PATH, model_name + '.npy'), label_encoder.classes_)

from sklearn.preprocessing import LabelEncoder
def load_label_encoder(model_name):
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(os.path.join(MODEL_PATH, model_name + '.npy'))
    return label_encoder

import pickle
def save_model(model, model_name):
    pickle.dump(model, open(os.path.join(MODEL_PATH, model_name + '.sav'), 'wb'))

def load_model(model_name):
    return pickle.load(open(os.path.join(MODEL_PATH, model_name + '.sav'), 'rb'))

"""Plotting"""
def convert_title_to_filename(title):
    for unacceptable in [' ', ':', '.', '(', ')']:
        title = title.replace(unacceptable, '_')
    return title

def convert_column_name_to_title(column_name):
    return column_name.replace('_', ' ').title()

def plot_pie(data, column_name, title = None, save = False, show = True):
    temp = data[column_name].value_counts()
    temp = pd.DataFrame({'labels': temp.index,
                       'values': temp.values
                      })
    values = temp['values']
    labels = temp['labels']

    fig = plt.figure(figsize=(12, 12), facecolor='w')
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.4)
    patches = plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance = 0.95,
        textprops={'fontsize': 16, 'bbox': bbox_props})

    plt.axis('equal')

    if title is None:
        title = 'Distribution of ' + column_name

    plt.title(title, loc = 'center', y=1.1, fontsize = 25)
    plt.tight_layout()

    if save:
        check_dir(IMAGE_PIE_DIRECTORY)
        saved_path = os.path.join(IMAGE_PIE_DIRECTORY, convert_title_to_filename(title))
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()
    plt.close()

def plot_catogory_distribution(data, column_name, title = None, percentage = False, rot = 0, save = False, show = True):
    fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    if percentage == False:
        data[column_name].value_counts(dropna=False).plot(kind='bar', color = 'c', rot = 0)
    else:
        (data[column_name].value_counts(dropna=False) / data.shape[0]*100).plot(kind='bar', color = 'c', rot = rot)

    if percentage == False:
        plt.ylabel('No. of users')
    else:
        plt.ylabel('% of users')

    if title is None:
        title = 'Distribution of ' + column_name
        if percentage:
            title = 'Percentage ' + title

    plt.title(title, loc = 'center', y=1.1, fontsize = 25)
    plt.tight_layout()

    if save:
        check_dir(IMAGE_BAR_DIRECTORY)
        saved_path = os.path.join(IMAGE_BAR_DIRECTORY, convert_title_to_filename(title))
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()

def plot_continuous_distribution_as_bar(data, column_name, title = None, bins = None, save = False, show = True):
    fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    sns.distplot(data[column_name].dropna(), bins = bins, kde = False)

    if title is None:
        title = 'Distribution of ' + column_name
    plt.title(title, loc = 'center', y=1.1, fontsize = 25)
    plt.tight_layout()

    if save:
        check_dir(IMAGE_BIN_DIRECTORY)
        saved_path = os.path.join(IMAGE_BIN_DIRECTORY, convert_title_to_filename(title))
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()

def plot_continuous_distribution_as_box(data, continuous_column, category_column = None, title = None, save = False, show = True):
    fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    sns.boxplot(y = continuous_column , x = category_column, data = data, color = 'c')
    if title is None:
        title = 'Distribution of ' + continuous_column
        if category_column is not None:
            title = title + ' By ' + category_column
    plt.title(title, loc = 'center', y=1.1, fontsize = 25)
    plt.tight_layout()

    if save:
        check_dir(IMAGE_BOX_DIRECTORY)
        saved_path = os.path.join(IMAGE_BOX_DIRECTORY, convert_title_to_filename(title))
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()

def plot_category_stacked_bar(data, x_column, y_column, percentage = False, title = None, rot = 0, save = False, show = True):
    if percentage == False:
        ctab=pd.crosstab(data[x_column], data[y_column])
    else:
        ctab=pd.crosstab(data[x_column], data[y_column]).apply(lambda x: x/x.sum()*100, axis=1)

    ctab.plot(kind='bar', stacked=True, legend=True, rot = rot, figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    if percentage == False:
        plt.ylabel('No. of users')
    else:
        plt.ylabel('% of users')

    if title is None:
        title = 'Distribution of ' + y_column + ' By ' + x_column + ' (Stacked)'
        if percentage:
            title = 'Percentage ' + title

    plt.title(title, loc = 'center', y=1.15, fontsize = 15)
    plt.tight_layout()

    plt.legend(loc=(1.04,0))

    if save:
        check_dir(IMAGE_BARS_DIRECTORY)
        saved_path = os.path.join(IMAGE_BARS_DIRECTORY, convert_title_to_filename(title))
        plt.savefig(saved_path, dpi=200, bbox_inches="tight")
        plt.show()
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()

def plot_category_clustered_bar(data,level_1, level_2, title = None, save = False, show = True):

    sns.catplot(data = data, x = level_1, hue = level_2, kind = "count", height = PLOT_HEIGHT, aspect = 1.618)
    if title is None:
        title = 'Distribution of ' + level_2 + ' By ' + level_1 + ' (Clustered)'

    plt.title(title, loc = 'center', y=1.2, fontsize = 20)
    plt.tight_layout()

    if save:
        check_dir(IMAGE_BARS_DIRECTORY)
        saved_path = os.path.join(IMAGE_BARS_DIRECTORY, convert_title_to_filename(title))
        plt.savefig(saved_path, dpi=200, bbox_inches="tight")
        plt.show()
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()

def plot_bubble(data, column_name, target, title = None, save = False, show = True):
    df = data.reset_index().groupby([column_name]).agg({target:'mean',
                                        'index':'count'}).reset_index().rename(columns = {'index': 'Total Count'})
    ratio = df['Total Count'].max() / 3000
    fig = sns.relplot(x="Total Count", y=target, size="Total Count", hue="Total Count", palette="Blues",
                    sizes=(df['Total Count'].min() / ratio, 5000),
                    data=df, height=PLOT_HEIGHT, aspect=1.618, legend = False)
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.4)
    for line in range(0,df.shape[0]):
         plt.text(df["Total Count"].iloc[line] * 1.01, df[target].iloc[line] * 1.05, df[df.columns[0]].iloc[line],
            bbox=bbox_props, horizontalalignment='left', size='large', color='black', fontsize = 12)
    if title is None:
        title = 'Subscription Rate vc Total Count by ' + column_name.replace('_', ' ').replace('.', ' ').title()

    plt.xlim(-df['Total Count'].max()*0.01, df['Total Count'].max()*1.1)
    plt.ylim(0, df[target].max()*1.1)
    x = np.linspace(0, df['Total Count'].max()*1.1, df['Total Count'].max()*1.1)
    y = [df[target].mean()] * len(x)
    plt.ylabel('Booking Rate')
    plt.plot(x, y, linewidth = 2, color = '#2fb7b7')
    plt.text(df["Total Count"].iloc[-1] * -0.01, y[0] * 1.05, 'Average', horizontalalignment='left', size='large', color='#2fb7b7', fontsize = 20)
    plt.title(title, loc = 'center', y=1.3, fontsize = 20)
    plt.tight_layout()

    if save:
        check_dir(IMAGE_BUBBLE_DIRECTORY)
        saved_path = os.path.join(IMAGE_BUBBLE_DIRECTORY, convert_title_to_filename(title))
        plt.savefig(saved_path, dpi=200, bbox_inches="tight")
        plt.show()
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()

def plot_pairs(data,column_names, title = None, save = False, show = True):
    sns.pairplot(data[column_names])
    if title is None:
        title = 'pairs'
    plt.tight_layout()

    if save:
        check_dir(IMAGE_GENERAL_DIRECTORY)
        saved_path = os.path.join(IMAGE_GENERAL_DIRECTORY, convert_title_to_filename(title))
        plt.savefig(saved_path, dpi=200, bbox_inches="tight")
        plt.show()
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()

def plot_month_week_heatmap(data, title = None, save = False, show = True):
    fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    sns.heatmap(data, cmap="YlGnBu")

    if title is None:
        title = 'Total No. of Customers Heatmap'

    plt.title(title, loc = 'center', y=1.1, fontsize = 25)
    plt.xlabel('Month')
    plt.ylabel('Day of the Week')
    plt.tight_layout()

    if save:
        check_dir(IMAGE_TIME_DIRECTORY)
        saved_path = os.path.join(IMAGE_TIME_DIRECTORY, convert_title_to_filename(title))
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()

def get_prob_top_n(prob, n):
    prob_sorted = prob.argsort()[:, ::-1]
    return prob_sorted[:, n]

from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
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
