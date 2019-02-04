import os
os.getcwd()
# if connect to local
WORKING_DIR = '/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings'
# if connect to hosted
# WORKING_DIR = '/content'

# ------------------------------------------------------------------------------
import pandas as pd

USERS_PATH = os.path.join(WORKING_DIR, 'raw_data/users.csv')
SESSIONS_PATH = os.path.join(WORKING_DIR, 'raw_data/sessions.csv')

users = pd.read_csv(USERS_PATH)
sessions = pd.read_csv(SESSIONS_PATH)
countries = pd.read_csv(COUNTRIES_PATH)

users = users[users['id'].isin(list(sessions['user_id']))]
sessions = sessions[sessions['user_id'].isin(list(users['id']))]

users.to_csv('data/users_all.csv', index = False)
sessions.to_csv('data/sessions_all.csv', index = False)
