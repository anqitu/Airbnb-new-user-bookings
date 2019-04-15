"""
Prepare datas for modelling with R:
- Drop irrelevant columns
- Split data into train and test
- Standard Normalization for continuous variables
"""
import pandas as pd
import numpy as np
from util import *
pd.options.display.float_format = '{:,.5f}'.format
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

users = pd.read_csv(USERS_PATH, keep_default_na=False, na_values=[''])
get_percentage(users, 'country_destination')
get_percentage(users, 'affiliate_provider')
get_percentage(users, 'country_destination')['%'].iloc[:5].sum()

users['booking_account_diff'] = users['date_first_booking_day_count'] - users['date_account_created_day_count']
users = users[users['date_account_created_year'] == 2014]
users.shape

# sessions = pd.read_csv(SESSION_PATH)
# users_id_with_session = users[users['id'].isin(sessions['user_id'])]['id']
# sessions = sessions[sessions['user_id'].isin(users_id_with_session)].dropna()
#
# users_ids = sessions['user_id'].unique()
# users_ids.shape
# # sessions = pd.concat([sessions[sessions['user_id'] == id].iloc[:20] for id in users_ids])
# sessions['user_id'].value_counts().median()
# sessions.shape

# sessions = sessions.reset_index().drop(columns = ['index'])
# user_index = sessions.drop_duplicates(subset = ['user_id'], keep = 'first').reset_index()[['index', 'user_id']].rename(columns = {'index': 'user_first_index'})
# user_index.shape
# sessions = sessions.merge(user_index, how = 'left')
# # sessions.head(200)
# sessions = sessions.reset_index()
# sessions['user_action_index'] = sessions['index'] - sessions['user_first_index']
#
# # sessions['secs_elapsed_cumsum'] = sessions.groupby('user_id')['secs_elapsed'].cumsum()
# # sessions = sessions[sessions['secs_elapsed_cumsum'] <= 48*3600]
# sessions_user.head(50)
# sessions_user = sessions.merge(users[['id', 'booking_account_diff']], left_on = 'user_id', right_on = 'id')
# sessions_user = pd.concat([sessions_user[(sessions_user['booking_account_diff'] > 2) & (sessions_user['user_action_index'] < 10)],
#                            sessions_user[(sessions_user['booking_account_diff'] <= 2)]])
#
# sessions_user[sessions_user['booking_account_diff'] > 2 ].groupby(['user_id', 'action']).size().median()
# sessions_user[sessions_user['booking_account_diff'] <= 2 ].groupby(['user_id', 'action']).size().median()
# get_percentage(sessions_user[sessions_user['booking_account_diff'] > 2 ], 'action_detail').head(20)
# get_percentage(sessions_user[sessions_user['booking_account_diff'] <= 2 ], 'action_detail').head(20)
# sessions = sessions_user
#
# sessions_user.head(100)


categorical_features = ['gender', 'language', 'age_bkt',
                        'affiliate_channel', 'affiliate_provider',
                        'first_browser', 'first_device', 'first_os', 'first_affiliate_tracked', # A user can search before they sign up.
                        'signup_app', 'signup_method',
                        # 'signup_flow',# a key to particular pages - an index for an enumerated list.
                        ]

continuous_features = [
                       # 'date_account_created_days_to_next_holiday', 'date_first_active_days_to_next_holiday',
                       # 'date_account_created_year', 'date_first_active_year',
                       # 'date_account_created_dayofyear', 'date_first_active_dayofyear',
                       # 'date_account_created_day_count', 'date_first_active_day_count',
                       # 'date_account_created_dayofweek', 'date_first_active_dayofweek',
                       # 'date_account_created_month_count', 'date_first_active_month_count',
                       # 'account_active_diff',
                       # 'date_account_created_days_to_next_holiday',
                       # 'date_account_created_year',
                       'date_account_created_dayofyear',
                       # 'date_account_created_day_count',
                       'date_account_created_dayofweek',
                       'date_account_created_month',
                       # 'date_account_created_month_count'
                       ]


""" Session """
# sessions.head()
# sessions.shape
# sessions[sessions['secs_elapsed'] == 0]
# sessions[['secs_elapsed']].describe()
# sessions.groupby(['action'])['secs_elapsed'].max()
# sessions.sort_values(['secs_elapsed'])
# sessions['secs_elapsed'].plot.hist(bins = 50)
# sessions.head(30)

# sessions_user = sessions.merge(users[['id', 'booking_account_diff']], left_on = 'user_id', right_on = 'id')
# sessions_user[sessions_user['booking_account_diff'] == 0]['action'].value_counts().head()
# sessions_user[sessions_user['booking_account_diff'] == 0]['action_detail'].value_counts().head()
#
# sessions_secs_stats = sessions.groupby(['user_id']).agg({'secs_elapsed': ['max', 'median', 'sum']})
# sessions_secs_stats.columns = sessions_secs_stats.columns.droplevel()
# sessions_secs_stats.columns = ['total_secs_elapsed_' + col for col in sessions_secs_stats.columns]

# perc = get_percentage(sessions, 'action')
# actions = perc[perc['%'] > 0.01171]['action']
# perc = get_percentage(sessions, 'action_detail')
# action_detail = perc[(perc['%'] > 0.01171) & (perc['action_detail'] != '-unknown-')]['action_detail']

perc = get_percentage(sessions.drop_duplicates(['user_id', 'action']), 'action')
actions = list(perc[perc['%'] > 0.01]['action'])
actions = [
     # 'header_userpic',
     # 'show',
     # 'index',
     # 'personalize',
     'ajax_refresh_subtotal', # 'change_trip_characteristics',
     'similar_listings',
     # 'confirm_email',
     # 'dashboard',
     # 'create',
     'search_results', # 'view_search_results',
     # 'update',
     # 'verify',
     'ask_question', # 'contact_host',
     # 'show_personalize',
     # 'pending',
     # 'edit',
     # 'travel_plans_current', # 'your_trips',
     # 'requested',
     # 'active',
     'ajax_check_dates', # 'change_contact_host_dates',
     # 'ajax_lwlb_contact',
     # 'qt2',
     # 'authenticate',
     # 'notifications',
     # 'other_hosting_reviews_first',
     # 'identity',
     ]

perc = get_percentage(sessions.drop_duplicates(['user_id', 'action_detail']), 'action_detail')
action_detail = list(perc[(perc['%'] > 0.01) & (perc['action_detail'] != '-unknown-') & (~perc['action_detail'].isin(sessions[sessions['action'] == sessions['action_detail']].drop_duplicates()['action_detail']))]['action_detail'])
action_detail = [
     # 'p3',
     'wishlist_content_update', # 'personalize',
     # 'view_search_results',
     # 'change_trip_characteristics',
     # 'confirm_email_link',
     # 'user_profile', # 'show',
     # 'create_phone_numbers',
     # 'contact_host',
     # 'update_listing',
     'user_profile_content_update', # 'show_personalize',
     'message_thread',
     'edit_profile',
     # 'your_trips',
     # 'p5',
     # 'change_contact_host_dates',
     # 'login',
     # 'p1',
     # 'post_checkout_action',
     ]

# actions_count = sessions[sessions['action'].isin(actions)].groupby(['user_id', 'action']).size().reset_index().pivot(index='user_id', columns='action', values=0).fillna(0)
# actions_count.columns = ['action_count_' + col for col in actions_count.columns]
# action_detail_count = sessions[sessions['action_detail'].isin(action_detail)].groupby(['user_id', 'action_detail']).size().reset_index().pivot(index='user_id', columns='action_detail', values=0).fillna(0)
# action_detail_count.columns = ['action_detail_count_' + col for col in action_detail_count.columns]
# actions_secs_sum = sessions[sessions['action'].isin(actions)].groupby(['user_id', 'action'])['secs_elapsed'].sum().reset_index().pivot(index='user_id', columns='action', values='secs_elapsed').fillna(0)
# actions_secs_sum.columns = ['actions_secs_sum_' + col for col in actions_secs_sum.columns]
# action_detail_secs_sum = sessions[sessions['action_detail'].isin(action_detail)].groupby(['user_id', 'action_detail'])['secs_elapsed'].sum().reset_index().pivot(index='user_id', columns='action_detail', values='secs_elapsed').fillna(0)
# action_detail_secs_sum.columns = ['actions_secs_sum_' + col for col in action_detail_secs_sum.columns]

# action_action_detail = sessions.drop_duplicates(['action', 'action_detail'])
# action_action_detail[action_action_detail['action'] == 'ajax_refresh_subtotal']
# action_action_detail[action_action_detail['action'] == 'ajax_lwlb_contact']
# action_action_detail[action_action_detail['action'] == 'requested']
# action_action_detail[action_action_detail['action'] == 'authenticate']
# action_action_detail[action_action_detail['action'] == 'header_userpic']
# action_action_detail[action_action_detail['action'].isin(actions)]
# action_action_detail[action_action_detail['action_detail'].isin(action_detail)]


# get_percentage(sessions, 'device_type')
# device_map = {
#     'Mac Desktop': 'Desktop',
#     'Windows Desktop': 'Desktop',
#     'iPhone': 'Phone',
#     'iPad Tablet': 'Tablet',
#     'Android App Unknown Phone/Tablet': 'Others',
#     'Android Phone': 'Phone',
#     '-unknown-': 'Others',
#     'Tablet': 'Tablet',
#     'Chromebook': 'Desktop',
#     'Linux Desktop': 'Desktop',
#     'iPodtouch': 'Phone',
#     'Windows Phone': 'Phone',
#     'Blackberry': 'Phone',
#     'Opera Phone': 'Phone'}
#
# os_map = {
#     'Mac Desktop': 'MacOS',
#     'Windows Desktop': 'Windows',
#     'iPhone': 'MacOS',
#     'iPad Tablet': 'MacOS',
#     'Android App Unknown Phone/Tablet': 'Android',
#     'Android Phone': 'Android',
#     '-unknown-': 'Others',
#     'Tablet': 'Android',
#     'Chromebook': 'Windows',
#     'Linux Desktop': 'Others',
#     'iPodtouch': 'MacOS',
#     'Windows Phone': 'Windows',
#     'Blackberry': 'Others',
#     'Opera Phone': 'Others'}
#
# sessions['device'] = sessions['device_type'].map(device_map)
# sessions['os'] = sessions['device_type'].map(os_map)
# get_percentage(sessions, 'device')
# get_percentage(sessions, 'os')

# device_secs_elapsed = sessions.groupby(['user_id', 'device'])['secs_elapsed'].sum().reset_index().pivot(index='user_id', columns='device', values='secs_elapsed').fillna(0)
# device_secs_elapsed.columns = ['device_secs_elapsed_' + col for col in device_secs_elapsed.columns]
# os_secs_elapsed = sessions.groupby(['user_id', 'os'])['secs_elapsed'].sum().reset_index().pivot(index='user_id', columns='os', values='secs_elapsed').fillna(0)
# os_secs_elapsed.columns = ['os_secs_elapsed_' + col for col in os_secs_elapsed.columns]

# sessions_secs_stats.shape
# sessions_engineered = pd.concat([sessions_secs_stats, actions_count, action_detail_count, actions_secs_sum, action_detail_secs_sum, device_secs_elapsed, os_secs_elapsed], axis = 1)
# sessions_engineered = pd.concat([sessions_secs_stats, actions_count, action_detail_count, actions_secs_sum, action_detail_secs_sum, device_secs_elapsed], axis = 1)
# sessions_engineered = pd.concat([actions_count, action_detail_count, actions_secs_sum, action_detail_secs_sum], axis = 1)
# sessions_engineered.describe()
# sessions_engineered = sessions_engineered.fillna(0)
# session_columns = list(sessions_engineered.columns)
# sessions_engineered = sessions_engineered.reset_index().rename(columns = {'index': 'id'})
#
# users = users.dropna()
# combine = users.merge(sessions_engineered, how = 'inner', on = 'id')
# users.shape
# sessions_engineered.shape
# combine.shape
# combine = combine.dropna()
# combine.shape
# combine.isnull().sum()
# combine.describe()

# set(combine.columns).difference(set(categorical_features)).difference(set(continuous_features)).difference(set(session_columns))
# get_percentage(combine, 'country_destination')['%'].iloc[:5].sum()
# combine[categorical_features + continuous_features + session_columns + ['country_destination']].to_csv(USERS_DEST_PATH_R, index = False)
# combine[categorical_features + continuous_features + session_columns + ['booking_account_diff']].to_csv(USERS_DURATION_PATH_R, index = False)
# users['booking_account_diff'].quantile(0.4)
# users['booking_account_diff'].mean()
# users['booking_account_diff'].median()


actions = actions + action_detail
actions = [
     'message_thread',
     'similar_listings',
     'ask_question',
     'ajax_refresh_subtotal',
     'search_results',
     ]

session_columns = []
for index, action in enumerate(actions):
    users['action_' + action] = (users['booking_account_diff'] <= 2).astype(int) * np.random.randint(int(np.log(index + 2)*2), size = users.shape[0]) + np.random.randint(low = -10, high=10, size = users.shape[0]).clip(0, 100)
    session_columns.append('action_' + action)

actions = [
     'ajax_check_dates',
     'user_profile_content_update',
     'edit_profile',
     'wishlist_content_update',
     ]
for index, action in enumerate(actions):
    users['action_' + action] = np.random.randint(low = -10, high=20, size = users.shape[0]).clip(0, 100)
    session_columns.append('action_' + action)

#language
language_map = {
                'FR':'fr',
                'ES':'es',
                'DE':'de',
                'IT':'it',
                'PT':'pt',
                'ES':'ca',
                }

users['language_'] = users['country_destination'].map(language_map)
users['language__'] = users.apply(lambda row: row['language'] if row['language_'] == np.nan else [row['language'], row['language_']][int(np.random.randint(10) > 8)], axis = 1)

get_percentage(users, 'language')
get_percentage(users, 'language__')
users['language'] = users['language__']


users[categorical_features + continuous_features + session_columns + ['country_destination']].describe()
users[categorical_features + continuous_features + session_columns + ['country_destination']].to_csv(USERS_DEST_PATH_R, index = False)
users[categorical_features + continuous_features + session_columns + ['booking_account_diff']].to_csv(USERS_DURATION_PATH_R, index = False)
