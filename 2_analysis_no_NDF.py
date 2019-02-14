"""
#### exploratory-data-analysis:
- Plotting
- Insighes

## Ideas
- Trend of any aspect of ratio
- Map of each destinations
"""

"""
## 1. Environment Setup
"""

import os
os.getcwd()
# if connect to local
# WORKING_DIR = 'C:/Users/Weixian/Documents/Y3S2/Analytics II/airbnb_new_user_booking'
WORKING_DIR = '/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings'
# if connect to hosted
# WORKING_DIR = '/content'
# os.listdir(WORKING_DIR)

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
# users = pd.read_csv(USERS_PATH, keep_default_na=False, na_values=[''])
users = pd.read_csv(USERS_WITH_NDF_PATH, keep_default_na=False, na_values=[''])
users.isnull().sum()
users.columns[users.isnull().any()]
# users = users.sample(5000)

"""#### Useful sub-data and features """
# date_first_active_day_count < date_account_created_day_count < date_first_booking_day_count
users['booking_active_diff'] = users['date_first_booking_day_count'] - users['date_first_active_day_count']
users['account_active_diff'] = users['date_account_created_day_count'] - users['date_first_active_day_count']
users['booking_account_diff'] = users['date_first_booking_day_count'] - users['date_account_created_day_count']
for col in ['date_account_created', 'date_first_active', 'date_first_booking']:
    users[col + '_year_month'] = users.apply(lambda row: date(row[col + '_year'], row[col + '_month'], 1), axis = 1)

users['has_age'] = (users['age_bkt'] != 'NA')

"""#### Useful sub-data and features """
users['has_destination'] = (users['country_destination'] != 'NDF')
users['has_destination_num'] = users['has_destination'].astype(int)
users_has_destination = users[users['country_destination'] != 'NDF']
users['booking_active_diff'] = users['date_first_booking_day_count'] - users['date_first_active_dayofyear']
users['to_US'] = (users['country_destination'] == 'US')
users_not_US = users[users['country_destination'] != 'US']

first_date = users[['date_account_created', 'date_first_active', 'date_first_booking']].min().min()
last_date = users[['date_account_created', 'date_first_active']].max().max()

# original_categorical_features = ['gender', 'language',
#                         'affiliate_channel', 'affiliate_provider',
#                         'first_browser', 'first_device_type', 'first_affiliate_tracked', # A user can search before they sign up.
#                         'signup_app', 'signup_method', 'signup_flow', # a key to particular pages - an index for an enumerated list.
#                         'has_age', 'age_bkt']

categorical_features = ['country_destination', 'to_US',
                        'gender',
                        'language', 'language_map_country',
                        'affiliate_channel',
                        'affiliate_provider', 'affiliate_provider_min_to_other',
                        'first_browser', 'first_browser_min_to_other',
                        'first_device_type', 'first_device', 'first_os',
                        'first_affiliate_tracked', 'first_affiliate_tracked_min_to_other', # A user can search before they sign up.
                        'signup_app', 'signup_method',
                        'signup_flow', 'signup_flow_min_to_other',# a key to particular pages - an index for an enumerated list.
                        'has_age',
                        'age_bkt', 'age_bkt_min_to_other',
                        'date_account_created_is_holiday', 'date_first_active_is_holiday', 'date_first_booking_is_holiday']

continuous_features = ['age', 'age_fix',
                       'booking_active_diff', 'account_active_diff', 'booking_account_diff',
                       'date_account_created_to_next_holiday', 'date_first_active_to_next_holiday', 'date_first_booking_to_next_holiday']

time_features = ['date_account_created',
                'date_first_booking',
                'date_first_active',

                'date_account_created_year',
                'date_account_created_month',
                'date_account_created_dayofweek',
                'date_account_created_dayofyear',
                'date_account_created_year_count',
                'date_account_created_month_count',
                'date_account_created_day_count',

                'date_first_active_year',
                'date_first_active_month',
                'date_first_active_dayofweek',
                'date_first_active_dayofyear',
                'date_first_active_year_count',
                'date_first_active_month_count',
                'date_first_active_day_count',

                'date_first_booking_year',
                'date_first_booking_month',
                'date_first_booking_dayofweek',
                'date_first_booking_dayofyear',
                'date_first_booking_year_count',
                'date_first_booking_month_count',
                'date_first_booking_day_count']

country_columns = ['ASIA', 'AU', 'CA', 'DE', 'ES', 'EU(Other)', 'FR', 'GB', 'IT', 'PT', 'US']

set(users.columns).difference(set(categorical_features)) \
                  .difference(set(continuous_features)) \
                  .difference(set(time_features)) \
                  .difference(set(country_columns))

"""## 3. Plot
- General
"""
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

""" Explanation """
# # data = users
# # activity = 'date_first_booking'
# # category = 'country_destination'
# # df = data.reset_index().groupby([activity  + '_month_count', category]).agg({'index': ['count'],'has_destination': ['sum']}).reset_index().rename(columns = {'index': 'Total Count', 'has_destination': 'Booking Count'})
# # df.columns = df.columns.droplevel(-1)
# # df['Booking Rate'] = df['Booking Count'] / df['Total Count']
# # total_per_month_df = df.groupby([activity  + '_month_count'])['Total Count'].sum().reset_index().rename(columns = {'Total Count': 'Month Count'})
# #
# # df = df.merge(total_per_month_df)
# # df['Month Percentage'] = df['Total Count'] / df['Month Count']
# # df
#
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
#
#
# """### Continuous Column
#
# #### Age
# """
#
# # Distribution of user's age
# plot_continuous_distribution_as_bar(data = users, column_name = 'age', title = "Age Distribution of users", bins = 16)
#
# # Distribution of user's age across the destination countries
# plot_continuous_distribution_as_box(data = users, continuous_column = 'age', category_column = 'country_destination', title = 'Age Distribution across the destinations')
# # Finding: Almost all the countries have a similar median age. Only users tavelling to Spain and Portugal are slightly younger.
# # Users of age 80 and above mostly choose US as their destination.
# # The reason might be the US user data i.e. as all the users are from US, older people in US prefer not to travel outside their home country.
#
# plot_continuous_distribution_as_box(data = users, continuous_column = 'age', category_column = 'has_destination', title = 'Age Distribution across the destinations')
# # Users having no bookings are reletivaely older than those with bookings.
#
# plot_category_stack(data = users, x_column = 'age_bkt', y_column = 'has_destination')
# plot_category_stack(data = users, x_column = 'age_bkt', y_column = 'has_destination', percentage = True)
# plot_category_bar(data = users, level_1 = 'age_bkt', level_2 = 'has_destination', title = None)
# # Users with unknown ages have higher chance of no bookings.
#
# plot_bubble_chart(data = users, column = 'age_bkt', save = False)
#
# """### Categorical Column
#
# #### Destination
# """
#
# # Distribution of user's selection of country
# plot_catogory_distribution(users, 'country_destination', title = "Distribution of destination countries")
# plot_catogory_distribution(users_has_destination, 'country_destination', title = "Distribution of destination countries")
# plot_catogory_distribution(users_has_destination_not_US, 'country_destination', title = "Distribution of destination countries")
# # Finding: Most of our users have never booked.
# # US is the most populor choice as the dataset is from the US users who would likely to prefer travel to nearer place within their home country
#
# plot_trend_chart(data = users, activity = 'date_first_booking', category = 'country_destination', save = False)
#
# """#### Gender"""
#
# # Distribution of user's gender
# plot_catogory_distribution(users, 'gender', title = "Gender Distribution of users")
# # Finding: Female and Male are around the same.
# # It means that the difference between the gender of the users is not significant.
# # Also, around one third of the gender information is missing from the dataset.
#
# plot_catogory_distribution(users_has_destination, 'gender', title = "Gender Distribution of users")
# # For users having bookings, female has the largest portion.
#
# plot_category_stack(data = users, x_column = 'gender', y_column = 'has_destination')
# plot_category_stack(data = users, x_column = 'gender', y_column = 'has_destination', percentage = True)
# # Users with unknown ages have higher chance of no bookings.
# # print(users[users['gender'] == 'OTHER'].shape) # 282
#
# plot_category_bar(data = users, level_1 = 'country_destination', level_2 = 'gender', title = None)
# # The bookings made by females are slightly higher for most of the destination countries
# # except for Canada, Denmark, Netherlands and other(not mentioned) countries where booking by males are slightly more than the females.
#
# plot_category_stack(data = users, x_column = 'country_destination', y_column = 'gender', percentage = True)
# # Users having no bookings are most likely to have unknown ages.
#
# # Distribution of user's gender across the destination countries
# plot_category_stack(data = users, x_column = 'gender', y_column = 'country_destination', percentage = True)
# plot_category_stack(data = users_has_destination, x_column = 'gender', y_column = 'country_destination', percentage = True)
# plot_category_stack(data = users_has_destination_not_US, x_column = 'gender', y_column = 'country_destination', percentage = True)
# # Except for NDF where unknown have higher chance. All other destination shows similar trend for both genders
#
# plot_bubble_chart(data = users, column = 'gender', save = False)
# plot_bubble_chart(data = users, column = 'gender', save = True)
# # More unknown, but with lower booking rate. Male and Female have similar booking rate.
#
# temp_users = users[(users['date_first_booking_month_count'] >18) & (users['date_first_booking_month_count'] < 64) & (users['gender'] != 'OTHER')]
# plot_trend_chart(data = temp_users, activity = 'date_first_booking', category = 'gender', save = False)
# # Ignore the first 18 months (less than 100) and OTHERS
# # temp_users.groupby(['date_first_booking_month_count', 'gender']).size()
#
# """#### Language"""
#
# plot_catogory_distribution(users, 'language', title = "Language Distribution of users")
# plot_catogory_distribution(users[users['language'] != 'en'], 'language', title = "Language Distribution of users")
# # Most users use english, followed by zh, fr and es.
#
# # users['language'].value_counts()
#
# plot_category_stack(data = users_has_destination, x_column = 'language', y_column = 'country_destination', percentage = True)
# plot_category_stack(data = users_has_destination_not_US, x_column = 'language', y_column = 'country_destination', percentage = True)
# # Users having no bookings are most likely to have unknown ages.
#
# # languages_more_than_100 = list(users['language'].value_counts().reset_index().iloc[:10]['index'])
# # users_popular_languages = users[users['language'].isin(languages_more_than_100)]
# # users_popular_languages_has_destination = users_has_destination[users_has_destination['language'].isin(languages_more_than_100)]
# # users_popular_languages_has_destination_not_US = users_has_destination_not_US[users_has_destination_not_US['language'].isin(languages_more_than_100)]
#
# # plot_category_stack(data = users_popular_languages_has_destination, x_column = 'language', y_column = 'country_destination', percentage = True)
# # plot_category_stack(data = users_popular_languages_has_destination_not_US, x_column = 'language', y_column = 'country_destination', percentage = True)
# # # Users having no bookings are most likely to have unknown ages.
#
# plot_bubble_chart(data = users, column = 'language', save = False)
#
# """#### Sign up method"""
#
# # Distribution of user signup method distribution
# plot_catogory_distribution(users, 'signup_method', title = "Distribution of signup method", percentage = True)
# plot_catogory_distribution(users_has_destination, 'signup_method', title = "Distribution of signup method")
# # Two thirds of the users use the basic signup method to register themselves on Airbnb, followed by Facebook.
#
# users['signup_method'].value_counts()
#
# # Booking rate of each signup method
# plot_category_stack(data = users, x_column = 'signup_method', y_column = 'has_destination', percentage = True)
# # Basic sign up has higher booking rate
#
# # Distribution of user's signup method across the destination countries
# plot_category_stack(data = users, x_column = 'country_destination', y_column = 'signup_method', percentage = True)
# # Basic signup method is most common among users to signup into Airbnb to book any of the destination countries.
#
# plot_bubble_chart(data = users, column = 'signup_method', save = False)
#
# """#### Sign up App"""
#
# # Distribution of user signup app distribution
# plot_catogory_distribution(users, 'signup_app', title = "Signup app distribution of users")
# # More than 80% of the users signup using Web, followed by iOS, Mobile Web and Android.
#
# # Distribution of user's signup app across the destination countries
# plot_category_stack(data = users, x_column = 'country_destination', y_column = 'signup_app', percentage = True)
# plot_category_bar(data = users, level_1 = 'country_destination', level_2 = 'signup_app', title = None)
# # users mostly use web irrespective of booking any of the destination countries.
#
# plot_bubble_chart(data = users, column = 'signup_app', save = False)
#
# """#### Sign up flow"""
#
# # Distribution of user sign up flow distribution
# plot_catogory_distribution(users, 'signup_flow', title = "Distribution of sign up flow")
# plot_catogory_distribution(users_has_destination, 'signup_flow', title = "Distribution of sign up flow")
# # Most users came from the same website to sign up
#
# users['signup_flow'].value_counts()
#
# # Booking rate of each signup flow page
# plot_category_stack(data = users, x_column = 'signup_flow', y_column = 'has_destination', percentage = True)
#
# # Distribution of user's signup app across the destination countries
# plot_category_stack(data = users, x_column = 'country_destination', y_column = 'signup_flow', percentage = True)
# # Dispite its least effectiveness, Direct paid marketing is still responsible for attracting most of the users to use Airbnb
# # due its ability to attract many users
#
# """#### Affiliate Channel (which paid marketing)"""
#
# # Distribution of user signup app distribution
# plot_catogory_distribution(users, 'affiliate_channel', title = "Distribution of Affiliate channels used to attract the users")
# plot_catogory_distribution(users_has_destination, 'affiliate_channel', title = "Distribution of Affiliate channels used to attract the users")
# # Direct paid marketing is responsible for attracting most of the users to use Airbnb
#
# # Booking rate of each channel
# plot_category_stack(data = users, x_column = 'affiliate_channel', y_column = 'has_destination', percentage = True)
# # However, it is the least effective in making users book travel experience
#
# # Distribution of user's signup app across the destination countries
# plot_category_stack(data = users, x_column = 'affiliate_channel', y_column = 'country_destination', percentage = True)
# plot_category_stack(data = users_has_destination, x_column = 'affiliate_channel', y_column = 'country_destination', percentage = True)
# plot_category_stack(data = users_has_destination_not_US, x_column = 'affiliate_channel', y_column = 'country_destination', percentage = True)
# # Dispite its least effectiveness, Direct paid marketing is still responsible for attracting most of the users to use Airbnb
# # due its ability to attract many users
#
# plot_bubble_chart(data = users, column = 'affiliate_channel', save = False)
#
# """#### Affiliate Provider (where the marketing)"""
#
# plot_catogory_distribution(users, 'affiliate_provider', title = "Distribution of Affiliate Providers used to attract the users")
# plot_catogory_distribution(users_has_destination, 'affiliate_provider', title = "Distribution of Affiliate channels used to attract the users")
#
# users['affiliate_provider'].value_counts()
#
# # Distribution of user's signup app across the destination countries
# plot_category_stack(data = users, x_column = 'affiliate_provider', y_column = 'country_destination', percentage = True, rot = 45)
# plot_category_stack(data = users_has_destination, x_column = 'affiliate_provider', y_column = 'country_destination', percentage = True)
# plot_category_stack(data = users_has_destination_not_US, x_column = 'affiliate_provider', y_column = 'country_destination', percentage = True)
# # Dispite its least effectiveness, Direct paid marketing is still responsible for attracting most of the users to use Airbnb
# # due its ability to attract many users
#
# """#### First Affiliate Tracked (the first marketing the user interacted with before the signing up)"""
#
# plot_catogory_distribution(users, 'first_affiliate_tracked', title = "Distribution of First Affiliate Tracked used to attract the users")
# plot_catogory_distribution(users_has_destination, 'first_affiliate_tracked', title = "Distribution of Affiliate channels used to attract the users")
#
# # Distribution of user's signup app across the destination countries
# plot_category_stack(data = users, x_column = 'first_affiliate_tracked', y_column = 'country_destination', percentage = True, rot = 45)
# plot_category_stack(data = users_has_destination, x_column = 'first_affiliate_tracked', y_column = 'country_destination', percentage = True)
# plot_category_stack(data = users_has_destination_not_US, x_column = 'first_affiliate_tracked', y_column = 'country_destination', percentage = True)
# # Dispite its least effectiveness, Direct paid marketing is still responsible for attracting most of the users to use Airbnb
# # due its ability to attract many users
#
# """#### First Device Type"""
#
# # Distribution of user first device type
# plot_catogory_distribution(users, 'first_device_type', title = "Distribution of first device type", percentage = True, rot = 30)
# plot_catogory_distribution(users_has_destination, 'first_device_type', title = "Distribution of first device type", percentage = True, rot = 30)
# # 30% of the users use Mac Desktop for fist access to Airbnb.
# # More than 40% of the users who book used Mac Desktop for fist access to Airbnb.
# # Also, Mac Desktop and Windows Desktop together constitute appoximately 80% of all the users who use Desktop as the first device to access Airbnb.
# # This supports our earlier result that stated "80% of users use Web as a signup app to register on Airbnb".
# # With the assuption that users use the same device when signing up and accessing Airbnb for first time
#
# # Booking rate of each signup device
# plot_category_stack(data = users, x_column = 'first_device_type', y_column = 'has_destination', percentage = True, rot = 30)
# # Mac Desktop and Window Desktop has higher sign up rate.
# # Mac Desktop and Windows Desktop have been the most popular first devices used by users to access Airbnb.
# # iPhone is used more than iPad as a first device by the users to access Airbnb
# # But iPad is used more than iPhone as a first device by the users who book their places in countries.
#
# # Distribution of user's first device type across the destination countries
# plot_category_stack(data = users, x_column = 'country_destination', y_column = 'first_device_type', percentage = True)
# # Phones has lower booking rate compared to laptops and pads.
#
# # Distribution of user's signup app across the destination countries
# plot_category_stack(data = users, x_column = 'first_device_type', y_column = 'country_destination', percentage = True, rot = 45)
# plot_category_stack(data = users_has_destination, x_column = 'first_device_type', y_column = 'country_destination', percentage = True)
# plot_category_stack(data = users_has_destination_not_US, x_column = 'first_device_type', y_column = 'country_destination', percentage = True)
# # Dispite its least effectiveness, Direct paid marketing is still responsible for attracting most of the users to use Airbnb
# # due its ability to attract many users
#
# """#### First Browser"""
#
# # Distribution of user first browser
# plot_catogory_distribution(users, 'first_browser', title = "Distribution of first browser", percentage = True, rot = 60)
# plot_catogory_distribution(users_has_destination, 'first_browser', title = "Distribution of first browser", percentage = True, rot = 60)
# plot_catogory_distribution(users_has_destination, 'first_browser', title = "Distribution of first browser", percentage = False, rot = 60)
# # 25% of users use Chrome to access Airbnb, followed by Safari and Firefox.
#
# unpopular_browsers = list((users.first_browser.value_counts()[7:]).index)
#
# # Booking rate of each browser varies a lot
# plot_category_stack(data = users[~users['first_browser'].isin(unpopular_browsers)], x_column = 'first_browser', y_column = 'has_destination', percentage = True, rot = 60)
#
# # Distribution of user's signup app across the destination countries
# plot_category_stack(data = users[~users['first_browser'].isin(unpopular_browsers)], x_column = 'first_browser', y_column = 'country_destination', percentage = True, rot = 45)
# plot_category_stack(data = users_has_destination[~users_has_destination['first_browser'].isin(unpopular_browsers)], x_column = 'first_browser', y_column = 'country_destination', percentage = True)
# plot_category_stack(data = users_has_destination_not_US[~users_has_destination_not_US['first_browser'].isin(unpopular_browsers)], x_column = 'first_browser', y_column = 'country_destination', percentage = True)
# # Dispite its least effectiveness, Direct paid marketing is still responsible for attracting most of the users to use Airbnb
# # due its ability to attract many users
#

"""### Time Trend
"""

"""Monthly Trend"""
fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
df = users['date_first_active' + '_year_month'].value_counts().reset_index()
sns.lineplot(x="index", y=('date_first_active' + '_year_month'), data = df, lw=2, alpha = 0.6)
df = users['date_account_created' + '_year_month'].value_counts().reset_index()
sns.lineplot(x="index", y=('date_account_created' + '_year_month'), data = df, lw=2, alpha = 0.6)
df = users['date_first_booking' + '_year_month'].value_counts().reset_index()
sns.lineplot(x="index", y=('date_first_booking' + '_year_month'), data = df, lw=2, alpha = 0.6)
plt.legend(['First Active', 'Create Account', 'First Booking'])
plt.ylabel("No. of Users")
plt.xlabel("Month")
title = 'Trend of No. of Users'
plt.title(title, loc = 'center', y=1.1, fontsize = 25)
saved_path = os.path.join(IMAGE_TIME_DIRECTORY, convert_title_to_filename(title))
fig.savefig(saved_path, dpi=200, bbox_inches="tight")


def plot_trend(data, time_feature, category_column = None, title = None, save = False, show = True):
    fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    if title is None:
        title = 'Trend of {}'.format(convert_column_name_to_title(time_feature))
        if category_column:
            title = title + ' by ' + category_column

    if not category_column:
        df = data.groupby([time_feature]).size().reset_index().rename(columns = {0: 'Count'})
        sns.lineplot(x=time_feature, y= 'Count', data = df, lw=1.5)

    else:
        df = data.groupby([category_column, time_feature]).size().reset_index().rename(columns = {0: 'Count'})
        sns.lineplot(x=time_feature, y= 'Count', data = df, hue=category_column, lw=1.5)

    plt.title(title, loc = 'center', y=1.1, fontsize = 15)
    plt.xlabel('Time')
    plt.xlim(first_date, last_date)
    plt.ylabel('No. of Users')
    plt.tight_layout()

    if save:
        check_dir(IMAGE_TIME_DIRECTORY)
        saved_path = os.path.join(IMAGE_TIME_DIRECTORY, convert_title_to_filename(title))
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()

def plot_percentage_trend(data, time_feature, category_column, title = None, save = False, show = True):
    fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    if title is None:
        title = 'Percentage Trend of {} by {}'.format(convert_column_name_to_title(time_feature), convert_column_name_to_title(category_column))

    df = data.groupby([time_feature, category_column]).size().reset_index().rename(columns = {0: 'Count'})
    sum_df = data.groupby([time_feature]).size().reset_index().rename(columns = {0: 'Sum'})
    df = df.merge(sum_df)
    df['% of Users'] = df['Count'] / df['Sum']
    df = df.rename(columns = {category_column: convert_column_name_to_title(category_column)})
    sns.lineplot(x=time_feature, y= '% of Users', data = df, hue = convert_column_name_to_title(category_column), lw=1.5)
    plt.legend(loc=(1.04,0))

    plt.title(title, loc = 'center', y=1.1, fontsize = 15)
    plt.xlabel('Time')
    plt.xlim(first_date, last_date)
    plt.ylabel('% of Users')
    plt.tight_layout()

    if save:
        check_dir(IMAGE_TIME_DIRECTORY)
        saved_path = os.path.join(IMAGE_TIME_DIRECTORY, convert_title_to_filename(title))
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        print('Saved to {}'.format(saved_path))
    if show:
        plt.show()

    plt.close()

# time_feature_map = {'date_account_created': 'Create Account',
#                     'date_first_active': 'First Active',
#                     'date_first_booking': 'First Booking'}
#
# for col in ['date_account_created', 'date_first_active', 'date_first_booking']:
#     plot_trend(data = users, time_feature = col + '_year_month',
#                 category_column = None, title = 'Monthly Trend of No. of Users ({})'.format(time_feature_map[col]), save = True)
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


plot_percentage_trend(users, 'date_account_created_month', 'country_destination', title = None, save = False, show = True)

# """## Ignore"""
# import holidays
# holidays.US(years=[2010,2011,2012,2013,2014]).keys()
# holidays.US(years=[2010,2011,2012,2013,2014]).values()
# holidays = pd.DataFrame(data = {'date': list(holidays.US(years=[2010,2011,2012,2013,2014]).keys()), 'holiday': list(holidays.US(years=[2010,2011,2012,2013,2014]).values())})
# holidays
#
# # We next take a look at a plot of the count of 'Date First Booking' over time. Note the 3 US summer holidays (Memorial Day, Independence Day, Labor Day) are marked with a 'O' each year (not working here).
# # What immediately stands out is the seasonality of bookings. Bookings reach a peak around Labor Day each year and then decline into year end before starting to pick up again after New Year's. This suggests the month in the date variables could be useful to separate. Additionally, there appear to be small spikes in bookings right around the summer holidays so perhaps this could be useful, as well. (The spike in early 2014 is possibly the Super Bowl).
#
# fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
# users['date_first_booking'].value_counts().plot(kind='line', ax=axes, lw = 1)
# import holidays # this is code to plot the 3 major US summer holidays - the package is not available here
# holidays_tuples = holidays.US(years=[2010,2011,2012,2013,2014])
# popular_holidays = ['Independence Day', 'Labor Day', 'Memorial Day']
# # holidays_tuples = {k:v for (k,v) in holidays_tuples.items()}
# holidays_tuples = {k:v for (k,v) in holidays_tuples.items() if v in popular_holidays}
# us_holidays = pd.to_datetime([i[0] for i in holidays_tuples.items()])
# for date in us_holidays:
#     axes.annotate('O', (date, users[users.date_first_booking == date]['date_first_booking'].value_counts()), xytext=(-35, 145),
#                 textcoords='offset points', arrowprops=dict(arrowstyle='wedge'))
# fig.autofmt_xdate()
# plt.show()
#
# """#### Categorical variables
#
# There are a lot of categorical variable so this chart is a bit crowded. Below are some quick comments about each.
#
# - Starting with gender, it appears users with 'unknown' gender book less frequently than those with a known one while users with gender 'other' book more frequently
# - Users with the 'google' signup_method book less frequently than 'basic' or 'facebook'
# - Users with signup_flow 3 book more frequently than any other category while several have nearly 100% 'NDF'
# - Users with affiliate_channel 'content' book less frequently than other categories
# - Users with affiliate_provider 'craigslist', direct', and 'google' book more frequently than other categories (this begs the question as to why the google affiliate channel is more effective than the google sign up method)
# - Users with first_affiliate_tracked 'local ops' book less frequently than other categories
# - Users with signup_app 'Web' booked the most frequently, while those with 'Android' booked the least
# - Users with first_device_type 'Mac_Desktop' booked the most frequently, while those with 'Android Phone' booked the least
# - The chart on first_browser highlights the large number used above all else; it is difficult to glean any meaningful insights beyond that some obscure browsers that are not likely widely used have very high or very low booking frequencies.
# - The chart on language is somewhat surprising given that all the users were from the US - there are a large number of languages represented and this may warrant further investigation
# """
#
# # bar_order = ['NDF','US','other','FR','IT','GB','ES','CA','DE','NL','AU','PT']
# # cat_vars = ['gender', 'signup_method', 'signup_flow', 'affiliate_channel', 'affiliate_provider',
# #             'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'language']
# # from matplotlib.colors import Colormap
# # fig, ax4 = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
# # def pltCatVar(var,axis,ax_num):
# #     ctab = pd.crosstab([users[var]], users.country_destination).apply(lambda x: x/x.sum(), axis=1)
# #     ctab[bar_order].plot(kind='bar', stacked=True, ax=axis.reshape(-1)[ax_num],legend=False,
# #                          colormap='terrain')
# # for i,var in enumerate(cat_vars[:8]):
# #     pltCatVar(var,ax4,i)
# # plt.tight_layout()
# # fig, ax5 = plt.subplots(nrows=2, ncols=1, figsize=(16, 8), sharey=True)
# # for i,var in enumerate(cat_vars[8:]):
# #     pltCatVar(var,ax5,i)
# # box = ax5[0].get_position()
# # ax5[0].set_position([box.x0, box.y0 + box.height * 0.4, box.width, box.height * 0.6])
# # ax5[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=6)
#
#
# # print(countries.shape)
# # print(countries)
# # print(countries.describe())
# # print(countries.nunique())
# # print(display_null_percentage(countries))
