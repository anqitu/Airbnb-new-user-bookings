"""
#### exploratory-data-analysis:
- Plotting
- Insights

## Ideas
- Trend of any aspect of ratio
- Map of each destinations
"""

"""
## 1. Environment Setup
"""
WORKING_DIR = '/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings'

"""#### Import libraries"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from util import *
from datetime import date

# Draw inline
%matplotlib inline

# Set figure aesthetics
plt.style.use('fivethirtyeight')
PLOT_HEIGHT = 6
PLOT_WIDTH = PLOT_HEIGHT * 1.618
plt.rcParams["figure.figsize"] = [PLOT_WIDTH,PLOT_HEIGHT]

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Pandas settings
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1)
pd.options.display.float_format = '{:,.5f}'.format


"""#### Loading data"""
users = pd.read_csv(USERS_PLOT_PATH, keep_default_na=False, na_values=[''])
users.columns[users.isnull().any()]

get_percentage(users, 'affiliate_channel')

"""#### Useful sub-data and features """
users_has_destination = users[users['country_destination'] != 'NDF']
users_has_destination_not_US = users_has_destination[users_has_destination['country_destination'] != 'US']
users_has_destination['to_US'] = (users_has_destination['country_destination'] == 'US')
users_has_destination['to_US_num'] = users_has_destination['to_US'].astype(int)
first_date = users[['date_account_created', 'date_first_active', 'date_first_booking']].min().min()
last_date = users[['date_first_active']].max().max()

# """ #### Continuous Column """
# continuous_features = ['age', 'age_fix',
#                        'booking_active_diff', 'account_active_diff', 'booking_account_diff',
#                        'date_account_created_days_to_next_holiday', 'date_first_active_days_to_next_holiday', 'date_first_booking_days_to_next_holiday']
#
# """ #### age, age_fix, has_age, age_bkt """
# """ Distribution of user's age """
# plot_continuous_distribution_as_bar(data = users, column_name = 'age', title = "Age Distribution of users", bins = 100)
# plot_continuous_distribution_as_box(data = users, continuous_column = 'age', title = 'Age Distribution across the destinations')
# """ #### Finding:
#     - Outlier to be cleaned"""
#
# """ Distribution of user's age_fix """
# plot_continuous_distribution_as_bar(data = users, column_name = 'age_fix', title = "Age Distribution of users", bins = 16)
# """ #### Finding:
#     - Most users are from the age group of 25 - 40 """
#
# """ Distribution of user's age across the destination countries """
# plot_continuous_distribution_as_box(data = users, continuous_column = 'age_fix', category_column = 'country_destination', title = 'Age Distribution across the destinations')
# """ #### Finding:
#     - Almost all the countries have a similar median age. Only users tavelling to Spain and Portugal are slightly younger.
#     - Users of age 80 and above mostly choose US as their destination.
#     - The reason might be the US user data i.e. as all the users are from US, older people in US prefer not to travel outside their home country."""
#
def plot_bubble(data, column_name, target, title = None, save = False, show = True):
    df = data.reset_index().groupby([column_name]).agg({target:'mean',
                                        'index':'count'}).reset_index().rename(columns = {'index': 'Total Count'})
    ratio = df['Total Count'].max() / 3000
    fig = sns.relplot(x="Total Count", y=target, size="Total Count", palette="Blues",
                    sizes=(100, 100),
                    data=df, height=PLOT_HEIGHT, aspect=1.618, legend = False, alpha = 0.5)
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.4)
    for line in range(0,df.shape[0]):
         plt.text(df["Total Count"].iloc[line] * 0.97, df[target].iloc[line] * 1.02, df[df.columns[0]].iloc[line],
            bbox=bbox_props, horizontalalignment='left', size='large', color='black', fontsize = 8)
    if title is None:
        title = 'Booking Rate vc Total Count by ' + column_name.replace('_', ' ').replace('.', ' ').title()

    plt.xlim(-df['Total Count'].max()*0.01, df['Total Count'].max()*1.1)
    plt.ylim(0, df[target].max()*1.1)
    x = np.linspace(0, df['Total Count'].max()*1.1, df['Total Count'].max()*1.1)
    y = [data[target].mean()] * len(x)
    plt.ylabel('Booking Rate')
    plt.plot(x, y, linewidth = 2, color = '#2fb7b7')
    plt.text(df["Total Count"].iloc[-1] * 0.8, y[0] * 1.05, 'Average', horizontalalignment='left', size='large', color='#2fb7b7', fontsize = 20)
    plt.title(title, loc = 'center', y=1.3, fontsize = 20)
    plt.tight_layout()

    if save:
        check_dir(IMAGE_BUBBLE_DIRECTORY)
        saved_path = os.path.join(IMAGE_BUBBLE_DIRECTORY, convert_filename(title))
        plt.savefig(saved_path, dpi=200, bbox_inches="tight")
        plt.show()
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()


def plot_catogory_distribution(data, column_name, title = None, percentage = False, rot = 0, save = False, show = True):
    fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    if percentage == False:
        df = data[column_name].value_counts(dropna=False)
    else:
        df = (data[column_name].value_counts(dropna=False) / data.shape[0]*100)

    bars = plt.bar(df.index, list(df), color = 'c')

    for rect in bars:
        height = rect.get_height()
        if percentage == False:
            plt.text(rect.get_x() + rect.get_width()/2.0, height, int(height), ha='center', va='bottom', size = 12)
        else:
            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f %%'%(height), ha='center', va='bottom', size = 12)


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
        saved_path = os.path.join(IMAGE_BAR_DIRECTORY, convert_filename(title))
        plt.savefig(saved_path, dpi=200, bbox_inches="tight")
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()


""" Booking rate vs Age Group """
users['Age Group'] = users['age_bkt'].str.split('-').str.get(0)
plot_pie(users, 'age_bkt', title = "Distribution of Age Bucket", save = True, show = True)
plot_bubble(data = users, column_name = 'Age Group', target = 'has_destination_num', save = True)
plot_bubble(data = users, column_name = 'age_bkt', target = 'has_destination_num', save = True)
plot_catogory_distribution(users, 'age_bkt', title = "Distribution of Age Bucket", percentage = True, save = True)
get_percentage(users, 'age_bkt').to_csv('age_bkt_all.csv')


df = users.groupby(['age_bkt'])['has_destination_num'].mean().reset_index()
df = df.sort_values(['has_destination_num'], ascending = True)

plt.figure(facecolor='w', figsize=(PLOT_HEIGHT, PLOT_WIDTH))
bars = plt.barh(df[df.columns[0]], df[df.columns[1]], color = 'c')
plt.ylabel('Top 5 Accuracy')
plt.xlabel('Model')
# plt.ylim(0.4, 0.9)
title = 'Booking Rate By Age'

plt.title(title, loc = 'center', y=1.1, fontsize = 20)
for rect in bars:
    width = rect.get_width()
    plt.text(width + 0.03, rect.get_y(), '%.3f' % (width), ha='center', va='bottom', size = 12)
y = np.linspace(0,11, 20)
x = [users['has_destination_num'].mean()] * len(y)
plt.xlabel('Booking Rate')
plt.plot(x, y, linewidth = 4, color = '#4682B4')
plt.text(x[0] + 0.03, -0.3, 'Average', verticalalignment='bottom', size='large', color='#4682B4', fontsize = 20)
saved_path = os.path.join(IMAGE_BAR_DIRECTORY, convert_filename(title))
plt.savefig(saved_path, dpi=200, bbox_inches="tight")


# plot_continuous_distribution_as_box(data = users, continuous_column = 'age_fix', category_column = 'has_destination', title = 'Age Distribution across the destinations')
# plot_category_stacked_bar(data = users, x_column = 'age_bkt', y_column = 'has_destination', percentage = True)
# plot_category_clustered_bar(data = users, level_1 = 'age_bkt', level_2 = 'has_destination', title = None, order = sorted(users['age_bkt'].unique()))
# plot_category_clustered_bar(data = users[users['age_bkt'] != 'NA'], level_1 = 'age_bkt', level_2 = 'has_destination', title = None, order = sorted(users['age_bkt'].unique()))
""" #### Finding:
    - Users having no bookings are reletivaely older than those with bookings. Yonger people have higher booking rate
    - Users with unknown ages have higher chance of no bookings. """

# # """ #### booking_active_diff, account_active_diff,  booking_account_diff """
# # plot_continuous_distribution_as_bar(data = users_has_destination, column_name = 'booking_active_diff', bins = 100)
# # users_has_destination['booking_active_diff_clip'] = users_has_destination['booking_active_diff'].clip(0, 400)
# # plot_continuous_distribution_as_box(data = users_has_destination, continuous_column = 'booking_active_diff_clip', category_column = 'country_destination')
#
# """ #### date_account_created_days_to_next_holiday, date_first_active_days_to_next_holiday, date_first_booking_days_to_next_holiday"""
# plot_continuous_distribution_as_bar(data = users, column_name = 'date_account_created_days_to_next_holiday', bins = 20)
# plot_continuous_distribution_as_bar(data = users, column_name = 'date_first_active_days_to_next_holiday', bins = 20)
# plot_continuous_distribution_as_bar(data = users_has_destination, column_name = 'date_first_booking_days_to_next_holiday', bins = 20)
# plot_continuous_distribution_as_box(data = users_has_destination, continuous_column = 'date_first_booking_days_to_next_holiday')
# """ #### Finding:
#     - Users tend to create account, check airbnb and make booking when close to holiday """
#
#
# """ #### Categorical Column """
# categorical_features = ['country_destination', 'has_destination', 'has_destination_num',
#                         'gender', 'language', 'has_age', 'age_bkt',
#                         'affiliate_channel', 'affiliate_provider',
#                         'first_browser', 'first_device_type', 'first_device', 'first_os', 'first_affiliate_tracked', # A user can search before they sign up.
#                         'signup_app', 'signup_method', 'signup_flow', # a key to particular pages - an index for an enumerated list.
#                         'date_account_created_is_holiday', 'date_first_active_is_holiday', 'date_first_booking_is_holiday']
#
""" #### Destination """
plot_catogory_distribution(users, 'country_destination', title = "Distribution of Destination Countries (All)")
plot_catogory_distribution(users, 'country_destination', title = "Distribution of Destination Countries", percentage = True, save = True)
plot_catogory_distribution(users_has_destination, 'country_destination', title = "Distribution of destination countries")
get_percentage(users, 'country_destination').to_csv('country_destination_all.csv')
get_percentage(users_has_destination, 'country_destination').to_csv('country_destination.csv')
plot_catogory_distribution(users_has_destination, 'country_destination', title = "Distribution of Destination Countries (Only Booked)", percentage = True, save = True)
plot_catogory_distribution(users_has_destination_not_US, 'country_destination', title = "Distribution of destination countries")
""" # Finding:
    - Most of our users have never booked.
    - US is the most populor choice as the dataset is from the US users who would likely to prefer travel to nearer place within their home country """

# """#### Gender"""
plot_pie(users, 'gender', title = "Distribution of Gender", save = True, show = True)
plot_catogory_distribution(users, 'gender', title = "Distribution of Gender", percentage = True, save = True)
plot_catogory_distribution(users_has_destination, 'gender', title = "Gender Distribution of users")
plot_category_clustered_bar(data = users, level_1 = 'gender', level_2 = 'has_destination')
plot_category_stacked_bar(data = users, x_column = 'gender', y_column = 'has_destination', percentage = True)
plot_bubble(data = users, column_name = 'gender', target = 'has_destination_num', save = False)
""" # Finding:
    - Female and Male are around the same.
    - It means that the difference between the gender of the users is not significant.
    - Also, around one third of the gender information is missing from the dataset.
    - For users having bookings, female has the largest portion.
    - Users with unknown gender have higher chance of no bookings.
    - More unknown, but with lower booking rate. Male and Female have similar booking rate."""
#
# """#### Language"""
plot_catogory_distribution(users, 'language', title = "Distribution of Language", percentage = True, save = True)
plot_catogory_distribution(users[users['language'] != 'en'], 'language', title = "Language Distribution of users")
plot_bubble(data = users, column_name = 'language', target = 'has_destination_num', save = True)
""" # Finding:
    - Most users use english, followed by zh, fr and es.
    - High booking rate for english speakers --> Why?"""
#
# """#### Sign up method"""
# """ Distribution of user signup method distribution """
# plot_catogory_distribution(users, 'signup_method', title = "Distribution of signup method", percentage = True)
# plot_catogory_distribution(users_has_destination, 'signup_method', title = "Distribution of signup method")
# plot_bubble(data = users, column_name = 'signup_method', target = 'has_destination_num', save = True)
# plot_category_stacked_bar(data = users, x_column = 'signup_method', y_column = 'has_destination')
# plot_category_stacked_bar(data = users, x_column = 'signup_method', y_column = 'has_destination', percentage = True)
# plot_category_clustered_bar(data = users, level_1 = 'signup_method', level_2 = 'has_destination')
# """ # Finding:
#     - Two thirds of the users use the basic signup method to register themselves on Airbnb, followed by Facebook.
#     - Basic signup method is most common among users to signup into Airbnb to book any of the destination countries.
#     - Basic sign up has higher booking rate"""
#
# """#### Sign up App"""
# plot_catogory_distribution(users, 'signup_app', title = "Signup app distribution of users", percentage = True)
# plot_bubble(data = users, column_name = 'signup_app', target = 'has_destination_num', save = True)
# plot_category_clustered_bar(data = users, level_1 = 'signup_app', level_2 = 'has_destination')
# """ # Finding:
#     - More than 80% of the users signup using Web, followed by iOS, Mobile Web and Android."""
#
#
# """#### Sign up flow"""
# users['signup_flow'] = users['signup_flow'].astype(str)
# plot_catogory_distribution(users, 'signup_flow', title = "Signup app distribution of users", percentage = True)
# plot_bubble(data = users, column_name = 'signup_flow', target = 'has_destination_num', save = True)
# plot_category_clustered_bar(data = users, level_1 = 'signup_flow', level_2 = 'has_destination')
# """ # Finding:
#     - Most users came from the same website to sign up"""
#
# """#### Affiliate Channel (which paid marketing)"""
# plot_catogory_distribution(users, 'affiliate_channel', title = "Distribution of Affiliate channels used to attract the users")
# plot_category_stacked_bar(data = users, x_column = 'affiliate_channel', y_column = 'has_destination')
# plot_category_stacked_bar(data = users, x_column = 'affiliate_channel', y_column = 'has_destination', percentage = True)
# plot_category_clustered_bar(data = users, level_1 = 'affiliate_channel', level_2 = 'has_destination')
# plot_bubble(data = users, column_name = 'affiliate_channel', target = 'has_destination_num', save = True)
# """ # Finding:
#     - Direct paid marketing is responsible for attracting most of the users to use Airbnb
#     """
#
#
# """#### Affiliate Provider (where the marketing)"""
# plot_catogory_distribution(users, 'affiliate_provider', title = "Distribution of Affiliate Providers used to attract the users")
# plot_category_stacked_bar(data = users, x_column = 'affiliate_provider', y_column = 'has_destination')
# plot_category_stacked_bar(data = users, x_column = 'affiliate_provider', y_column = 'has_destination', percentage = True)
# plot_category_clustered_bar(data = users, level_1 = 'affiliate_provider', level_2 = 'has_destination')
# plot_bubble(data = users, column_name = 'affiliate_provider', target = 'has_destination_num', save = True)
# """ # Finding:
#     - Direct paid marketing is responsible for attracting most of the users to use Airbnb """
#
#
# """#### First Affiliate Tracked (the first marketing the user interacted with before the signing up)"""
# plot_catogory_distribution(users, 'first_affiliate_tracked', title = "Distribution of First Affiliate Tracked used to attract the users")
# plot_bubble(data = users, column_name = 'first_affiliate_tracked', target = 'has_destination_num', save = True)
#
# """#### First Browser"""
# plot_catogory_distribution(users, 'first_browser', title = "Distribution of first browser", percentage = True)
# plot_category_stacked_bar(data = users, x_column = 'first_browser', y_column = 'has_destination', percentage = True)
# plot_category_clustered_bar(data = users, level_1 = 'first_browser', level_2 = 'has_destination')
# plot_bubble(data = users, column_name = 'first_browser', target = 'has_destination_num', save = True)
# """ # Finding:
#     - 30% of users use Chrome to access Airbnb, followed by Safari and Firefox.
#     - Booking rate of each browser varies a lot"""
#
# """#### First Device Type"""
# plot_catogory_distribution(users, 'first_device_type', title = "Distribution of first device type", percentage = True, rot = 30)
# plot_category_stacked_bar(data = users, x_column = 'first_device_type', y_column = 'has_destination', percentage = True, rot = 30)
# plot_bubble(data = users, column_name = 'first_device_type', target = 'has_destination_num', save = True)
# plot_bubble(data = users, column_name = 'first_device', target = 'has_destination_num', save = True)
# plot_bubble(data = users, column_name = 'first_os', target = 'has_destination_num', save = True)
# """ # Finding:
#     - 32% of the users use Mac Desktop for fist access to Airbnb.
#     - Also, Mac Desktop and Windows Desktop together constitute appoximately 80% of all the users who use Desktop as the first device to access Airbnb.
#     - This supports our earlier result that stated "80% of users use Web as a signup app to register on Airbnb".
#     - With the assuption that users use the same device when signing up and accessing Airbnb for first time
#     - Mac Desktop and Window Desktop has higher sign up rate.
#     - Mac Desktop and Windows Desktop have been the most popular first devices used by users to access Airbnb.
#     - iPhone is used more than iPad as a first device by the users to access Airbnb
#     - But iPad is used more than iPhone as a first device by the users who book their places in countries.
#     - Phones has lower booking rate compared to laptops and pads."""
#
# """ #### Time Trend """
#
# """Monthly Trend"""
# users = users_has_destination
# fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
# df = users['date_first_active' + '_year_month'].value_counts().reset_index()
# sns.lineplot(x="index", y=('date_first_active' + '_year_month'), data = df, lw=2, alpha = 0.6)
# df = users['date_account_created' + '_year_month'].value_counts().reset_index()
# sns.lineplot(x="index", y=('date_account_created' + '_year_month'), data = df, lw=2, alpha = 0.6)
# df = users['date_first_booking' + '_year_month'].value_counts().reset_index()
# sns.lineplot(x="index", y=('date_first_booking' + '_year_month'), data = df, lw=2, alpha = 0.6)
# plt.legend(['First Active', 'Create Account', 'First Booking'])
# plt.ylabel("No. of Users")
# plt.xlabel("Month")
# title = 'Trend of No. of Users'
# plt.title(title, loc = 'center', y=1.1, fontsize = 25)
# saved_path = os.path.join(IMAGE_TIME_DIRECTORY, convert_filename(title))
# fig.savefig(saved_path, dpi=200, bbox_inches="tight")
#
#

def plot_trend(data, time_feature, category_column = None, title = None, save = False, show = True):
    fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH * 2, PLOT_HEIGHT))

    if title is None:
        title = 'Trend of {}'.format(convert_column_name_to_title(time_feature))
        if category_column:
            title = title + ' by ' + category_column

    if not category_column:
        df = data.groupby([time_feature]).size().reset_index().rename(columns = {0: 'Count'})
        df = df.set_index(time_feature)
        df.plot(lw=1.5)

    else:
        df = data.groupby([category_column, time_feature]).size().reset_index().rename(columns = {0: 'Count'})
        df = df.set_index(time_feature)
        df.plot(lw=1.5)

    plt.title(title, loc = 'center', y=1.1, fontsize = 15)
    plt.xlabel('Time')
    plt.ylabel('No. of Users')
    plt.tight_layout()

    if save:
        check_dir(IMAGE_TIME_DIRECTORY)
        saved_path = os.path.join(IMAGE_TIME_DIRECTORY, convert_filename(title))
        plt.savefig(saved_path, dpi=200, bbox_inches="tight")
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()

def plot_continuous_distribution_as_bar(data, column_name, title = None, bins = None, save = False, show = True, xlabel = None, ylabel = None):
    fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    sns.distplot(data[column_name].dropna(), bins = bins, kde = False)

    if title is None:
        title = 'Distribution of ' + column_name
    plt.title(title, loc = 'center', y=1.1, fontsize = 18)
    plt.tight_layout()

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    if save:
        check_dir(IMAGE_BIN_DIRECTORY)
        saved_path = os.path.join(IMAGE_BIN_DIRECTORY, convert_filename(title))
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()

users_has_destination['booking_account_diff_clip'] = users_has_destination['booking_account_diff'].clip(0,50)
users_has_destination[users_has_destination['booking_account_diff_clip']==0].shape[0]
users_has_destination.shape
users_has_destination['booking_account_diff_clip'].max()
365/5
# plot_continuous_distribution_as_box(data = users_has_destination, continuous_column = 'booking_account_diff_clip', title = 'Duration Between Account Creation and First Booking', save = True)
plot_continuous_distribution_as_bar(data = users_has_destination, column_name = 'booking_account_diff_clip',
                                    title = 'Distribution of Duration Between Account Creation and First Booking', save = True,
                                    # bins = 73//2*5,
                                    bins = 50,
                                    xlabel = 'Duration between Account Creation and Booking', ylabel = 'No. of Users')


# users_has_destination['booking_active_diff'].max()
# users_has_destination['booking_active_diff_clip'] = users_has_destination['booking_account_diff'].clip(0,600)
# plot_continuous_distribution_as_bar(data = users_has_destination, column_name = 'booking_active_diff_clip',
#                                     title = 'Distribution of Duration Between First Active and First Booking', save = True, bins = 73//2,
#                                     xlabel = 'Duration between First Active and Booking', ylabel = 'No. of Users')
#
# users_has_destination['account_active_diff'].median()
# users_has_destination['date_first_booking_days_to_next_holiday'].max()
# plot_continuous_distribution_as_bar(data = users_has_destination, column_name = 'date_first_booking_days_to_next_holiday',
#                                     title = 'Duration Between Booking and Next Holiday', bins = 26,
#                                     xlabel = 'Duration between Account Creation and Booking', ylabel = 'No. of Users')

# users_has_destination['date_first_active_days_to_next_holiday'].max()
# plot_continuous_distribution_as_bar(data = users_has_destination, column_name = 'date_first_active_days_to_next_holiday', bins = 13, title = 'Duration Between Active Day and Next Holiday')
# plt.show()

# def plot_percentage_trend(data, time_feature, category_column, title = None, save = False, show = True):
#     fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
#
#     if title is None:
#         title = 'Percentage Trend of {} by {}'.format(convert_column_name_to_title(time_feature), convert_column_name_to_title(category_column))
#
#     df = data.groupby([time_feature, category_column]).size().reset_index().rename(columns = {0: 'Count'})
#     sum_df = data.groupby([time_feature]).size().reset_index().rename(columns = {0: 'Sum'})
#     df = df.merge(sum_df)
#     df['% of Users'] = df['Count'] / df['Sum']
#     df = df.rename(columns = {category_column: convert_column_name_to_title(category_column)})
#     sns.lineplot(x=time_feature, y= '% of Users', data = df, hue = convert_column_name_to_title(category_column), lw=1.5)
#     plt.legend(loc=(1.04,0))
#
#     plt.title(title, loc = 'center', y=1.1, fontsize = 15)
#     plt.xlabel('Time')
#     plt.xlim(first_date, last_date)
#     plt.ylabel('% of Users')
#     plt.tight_layout()
#
#     if save:
#         check_dir(IMAGE_TIME_DIRECTORY)
#         saved_path = os.path.join(IMAGE_TIME_DIRECTORY, convert_filename(title))
#         fig.savefig(saved_path, dpi=200, bbox_inches="tight")
#         print('Saved to {}'.format(saved_path))
#     if show:
#         plt.show()
#
#     plt.close()
#
time_feature_map = {'date_account_created': 'Create Account',
                    'date_first_active': 'First Active',
                    'date_first_booking': 'First Booking'}

# for col in ['date_account_created', 'date_first_active', 'date_first_booking']:
#     plot_trend(data = users, time_feature = col + '_year_month',
#                 category_column = None, title = 'Monthly Trend of No. of Users ({})'.format(time_feature_map[col]), save = True)
users_has_destination[['date_first_booking_year_month']].info()
users_has_destination['date_first_booking_year_month'] = pd.to_datetime(users_has_destination['date_first_booking_year_month'])

users_has_destination_before_2014 = users_has_destination[users_has_destination['date_first_booking_year_month'] <= last_date]
plot_trend(data = users_has_destination_before_2014, time_feature = 'date_first_booking_year_month', category_column = None, title = 'Monthly Trend of No. of Users ({})'.format('First Booking'), save = True)

plot_trend(data = users_has_destination_before_2014[users_has_destination_before_2014['country_destination'] <= 'ES'], time_feature = 'date_first_booking_year_month', category_column = None, title = 'Monthly Trend of No. of Users ({})'.format('First Booking'), save = True)
plot_trend(data = users_has_destination_before_2014[users_has_destination_before_2014['country_destination'] <= 'FR'], time_feature = 'date_first_booking_year_month', category_column = None, title = 'Monthly Trend of No. of Users ({})'.format('First Booking'), save = True)



dest_per_month = users_has_destination.groupby(['date_first_booking_month', 'country_destination']).size().reset_index().rename(columns = {0: 'Count'})
total_per_month = users_has_destination.groupby(['date_first_booking_month']).size().reset_index().rename(columns = {0: 'Sum'})
merge = total_per_month.merge(dest_per_month)
merge['%'] = merge['Count'] / merge['Sum']


sns.lineplot(x = 'date_first_booking_month', y = '%', data = merge[merge['country_destination'] != 'US'], hue = 'country_destination', linewidth = 1)
plt.legend(loc = (1.04, 0.2))


#
# for categorical_feature in categorical_features:
#     for col in ['date_account_created', 'date_first_active', 'date_first_booking']:
#         plot_trend(data = users, time_feature = col + '_year_month',
#             category_column = categorical_feature, save = True,
#             title = 'Monthly Trend of No. of Users by {} ({})'.format(convert_column_name_to_title(categorical_feature), time_feature_map[col]))
#         plot_percentage_trend(data = users, time_feature = col + '_year_month',
#             category_column = categorical_feature, save = True,
#             title = 'Monthly Percentage Trend of No. of Users by {} ({})'.format(convert_column_name_to_title(categorical_feature), time_feature_map[col]))
#
# for col in ['date_account_created', 'date_first_active', 'date_first_booking']:
#     plot_trend(data = users_not_US, time_feature = col + '_year_month',
#         category_column = 'country_destination', save = True,
#         title = 'Monthly Trend of No. of Users by {} ({}) (Not US)'.format(convert_column_name_to_title('country_destination'), time_feature_map[col]))
#
#
#
#
# # """## Ignore"""
# # import holidays
# # holidays.US(years=[2010,2011,2012,2013,2014]).keys()
# # holidays.US(years=[2010,2011,2012,2013,2014]).values()
# # holidays = pd.DataFrame(data = {'date': list(holidays.US(years=[2010,2011,2012,2013,2014]).keys()), 'holiday': list(holidays.US(years=[2010,2011,2012,2013,2014]).values())})
# # holidays
# #
# # # We next take a look at a plot of the count of 'Date First Booking' over time. Note the 3 US summer holidays (Memorial Day, Independence Day, Labor Day) are marked with a 'O' each year (not working here).
# # # What immediately stands out is the seasonality of bookings. Bookings reach a peak around Labor Day each year and then decline into year end before starting to pick up again after New Year's. This suggests the month in the date variables could be useful to separate. Additionally, there appear to be small spikes in bookings right around the summer holidays so perhaps this could be useful, as well. (The spike in early 2014 is possibly the Super Bowl).
# #
# # fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
# # users['date_first_booking'].value_counts().plot(kind='line', ax=axes, lw = 1)
# # import holidays # this is code to plot the 3 major US summer holidays - the package is not available here
# # holidays_tuples = holidays.US(years=[2010,2011,2012,2013,2014])
# # popular_holidays = ['Independence Day', 'Labor Day', 'Memorial Day']
# # # holidays_tuples = {k:v for (k,v) in holidays_tuples.items()}
# # holidays_tuples = {k:v for (k,v) in holidays_tuples.items() if v in popular_holidays}
# # us_holidays = pd.to_datetime([i[0] for i in holidays_tuples.items()])
# # for date in us_holidays:
# #     axes.annotate('O', (date, users[users.date_first_booking == date]['date_first_booking'].value_counts()), xytext=(-35, 145),
# #                 textcoords='offset points', arrowprops=dict(arrowstyle='wedge'))
# # fig.autofmt_xdate()
# # plt.show()
# #
# # """#### Categorical variables
# #
# # There are a lot of categorical variable so this chart is a bit crowded. Below are some quick comments about each.
# #
# # - Starting with gender, it appears users with 'unknown' gender book less frequently than those with a known one while users with gender 'other' book more frequently
# # - Users with the 'google' signup_method book less frequently than 'basic' or 'facebook'
# # - Users with signup_flow 3 book more frequently than any other category while several have nearly 100% 'NDF'
# # - Users with affiliate_channel 'content' book less frequently than other categories
# # - Users with affiliate_provider 'craigslist', direct', and 'google' book more frequently than other categories (this begs the question as to why the google affiliate channel is more effective than the google sign up method)
# # - Users with first_affiliate_tracked 'local ops' book less frequently than other categories
# # - Users with signup_app 'Web' booked the most frequently, while those with 'Android' booked the least
# # - Users with first_device_type 'Mac_Desktop' booked the most frequently, while those with 'Android Phone' booked the least
# # - The chart on first_browser highlights the large number used above all else; it is difficult to glean any meaningful insights beyond that some obscure browsers that are not likely widely used have very high or very low booking frequencies.
# # - The chart on language is somewhat surprising given that all the users were from the US - there are a large number of languages represented and this may warrant further investigation
# # """
# #
# # # bar_order = ['NDF','US','other','FR','IT','GB','ES','CA','DE','NL','AU','PT']
# # # cat_vars = ['gender', 'signup_method', 'signup_flow', 'affiliate_channel', 'affiliate_provider',
# # #             'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'language']
# # # from matplotlib.colors import Colormap
# # # fig, ax4 = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
# # # def pltCatVar(var,axis,ax_num):
# # #     ctab = pd.crosstab([users[var]], users.country_destination).apply(lambda x: x/x.sum(), axis=1)
# # #     ctab[bar_order].plot(kind='bar', stacked=True, ax=axis.reshape(-1)[ax_num],legend=False,
# # #                          colormap='terrain')
# # # for i,var in enumerate(cat_vars[:8]):
# # #     pltCatVar(var,ax4,i)
# # # plt.tight_layout()
# # # fig, ax5 = plt.subplots(nrows=2, ncols=1, figsize=(16, 8), sharey=True)
# # # for i,var in enumerate(cat_vars[8:]):
# # #     pltCatVar(var,ax5,i)
# # # box = ax5[0].get_position()
# # # ax5[0].set_position([box.x0, box.y0 + box.height * 0.4, box.width, box.height * 0.6])
# # # ax5[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=6)
# #
# #
# # # print(countries.shape)
# # # print(countries)
# # # print(countries.describe())
# # # print(countries.nunique())
# # # print(display_null_percentage(countries))
#
#
# """ General Plot """
# for categorical_feature in categorical_features:
#     plot_pie(data = users, column_name = categorical_feature, save = True, show = True)
#
#     plot_catogory_distribution(data = users, column_name = categorical_feature, save = True, show = True)
#     plot_catogory_distribution(data = users, column_name = categorical_feature, percentage = True, save = True, show = True)
#
#     plot_category_stacked_bar(data = users, x_column = categorical_feature, y_column = 'country_destination', percentage = False, rot = 0, save = True)
#     plot_category_stacked_bar(data = users, x_column = categorical_feature, y_column = 'country_destination', percentage = True, rot = 0, save = True)
#     plot_category_clustered_bar(data = users, level_1 = categorical_feature, level_2 = 'country_destination', save = True, show = True)
#
#
# for continuous_feature in continuous_features:
#     plot_continuous_distribution_as_bar(data = users, column_name = continuous_feature, save = True)
#     plot_continuous_distribution_as_box(data = users, continuous_column = continuous_feature, category_column = 'country_destination', save = True)
#
#
# for categorical_feature in ['date_account_created_year', 'date_account_created_month', 'date_account_created_dayofweek',
#                             'date_first_active_year', 'date_first_active_month', 'date_first_active_dayofweek',
#                             'date_first_booking_year', 'date_first_booking_month', 'date_first_booking_dayofweek']:
#     plot_pie(data = users, column_name = categorical_feature, save = True, show = True)
#
#     plot_catogory_distribution(data = users, column_name = categorical_feature, save = True, show = True)
#     plot_catogory_distribution(data = users, column_name = categorical_feature, percentage = True, save = True, show = True)
#
#     plot_category_stacked_bar(data = users, x_column = categorical_feature, y_column = 'country_destination', percentage = False, rot = 0, save = True)
#     plot_category_stacked_bar(data = users, x_column = categorical_feature, y_column = 'country_destination', percentage = True, rot = 0, save = True)
#     plot_category_clustered_bar(data = users, level_1 = categorical_feature, level_2 = 'country_destination', save = True, show = True)
#
#
# # Heatmap
# month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
#              7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
#
# for col in ['date_account_created', 'date_first_active', 'date_first_booking']:
#     users[col + '_month_mapped'] = users[col + '_month'].map(month_map)
#
# df = users.groupby(['date_first_active' + '_month_mapped','date_first_active' + '_dayofweek']).size().reset_index().rename(columns = {0: 'Count'})
# df = df.pivot('date_first_active' + '_dayofweek', 'date_first_active' + '_month_mapped', "Count")
# df = df[[month for month in month_map.values() if month in df.columns]]
# df = df.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
# plot_month_week_heatmap(data = df, title = 'Total No. of Customers (First Active)', save = True)
#
# df = users.groupby(['date_account_created' + '_month_mapped','date_account_created' + '_dayofweek']).size().reset_index().rename(columns = {0: 'Count'})
# df = df.pivot('date_account_created' + '_dayofweek', 'date_account_created' + '_month_mapped', "Count")
# df = df[[month for month in month_map.values() if month in df.columns]]
# df = df.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
# plot_month_week_heatmap(data = df, title = 'Total No. of Customers (Create Account)', save = True)
#
# df = users.groupby(['date_first_booking' + '_month_mapped','date_first_booking' + '_dayofweek']).size().reset_index().rename(columns = {0: 'Count'})
# df = df.pivot('date_first_booking' + '_dayofweek', 'date_first_booking' + '_month_mapped', "Count")
# df = df[[month for month in month_map.values() if month in df.columns]]
# df = df.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
# plot_month_week_heatmap(data = df, title = 'Total No. of Customers (First Booking)', save = True)
#
#
#
# """ country_code_map """
# # country_code_map = {
# #     'NDF': 'No Booking',
# #     'US': 'United States',
# #     'other': 'Others',
# #     'FR': 'France',
# #     'CA': 'Canada',
# #     'GB': 'United Kingdom',
# #     'ES': 'Spain',
# #     'IT': 'Italy',
# #     'PT': 'Portugal',
# #     'DE': 'Germany',
# #     'AU': 'Australia'
# # }
# # users['country_destination'] = users['country_destination'].map(country_code_map)
