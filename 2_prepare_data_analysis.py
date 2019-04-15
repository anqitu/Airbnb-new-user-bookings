"""
#### Prepare data for Analysis:
- Create extra features
- Filter out features useful for analysis and plotting
"""

"""#### Loading data"""
import pandas as pd
from datetime import date
from util import *
users = pd.read_csv(USERS_WITH_NDF_ANALYSIS_PATH, keep_default_na=False, na_values=[''])
users.shape
get_percentage(users[users['country_destination'] != 'NDF'], 'country_destination')['%'].iloc[:5].sum()

"""#### Create extra features """
# date_first_active_day_count < date_account_created_day_count < date_first_booking_day_count
users['booking_active_diff'] = users['date_first_booking_day_count'] - users['date_first_active_day_count']
users['account_active_diff'] = users['date_account_created_day_count'] - users['date_first_active_day_count']
users['booking_account_diff'] = users['date_first_booking_day_count'] - users['date_account_created_day_count']
for col in ['date_account_created', 'date_first_active', 'date_first_booking']:
    users[col + '_year_month'] = users.apply(lambda row: np.nan if pd.isna(row[col])
                    else date(int(row[col + '_year']), int(row[col + '_month']), 1), axis = 1)
users['has_age'] = (users['age_bkt'] != 'NA')
users['has_destination'] = (users['country_destination'] != 'NDF')
users['has_destination_num'] = users['has_destination'].astype(int)

"""#### Select useful features """
categorical_features = ['country_destination', 'has_destination', 'has_destination_num',
                        'gender', 'language',
                        'affiliate_channel', 'affiliate_provider',
                        'first_browser', 'first_device_type', 'first_device', 'first_os', 'first_affiliate_tracked', # A user can search before they sign up.
                        'signup_app', 'signup_method', 'signup_flow', # a key to particular pages - an index for an enumerated list.
                        'has_age', 'age_bkt',
                        'date_account_created_is_holiday', 'date_first_active_is_holiday', 'date_first_booking_is_holiday']

continuous_features = ['age', 'age_fix',
                       'booking_active_diff', 'account_active_diff', 'booking_account_diff',
                       'date_account_created_days_to_next_holiday',
                       'date_first_active_days_to_next_holiday',
                       'date_first_booking_days_to_next_holiday']

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
                'date_account_created_year_month',

                'date_first_active_year',
                'date_first_active_month',
                'date_first_active_dayofweek',
                'date_first_active_dayofyear',
                'date_first_active_year_count',
                'date_first_active_month_count',
                'date_first_active_day_count',
                'date_first_active_year_month',

                'date_first_booking_year',
                'date_first_booking_month',
                'date_first_booking_dayofweek',
                'date_first_booking_dayofyear',
                'date_first_booking_year_count',
                'date_first_booking_month_count',
                'date_first_booking_day_count',
                'date_first_booking_year_month',]

selected_columns = categorical_features + continuous_features + time_features
set(users.columns).difference(set(selected_columns))
# users.isnull().sum()
users[selected_columns].to_csv(USERS_PLOT_PATH, index = False)
