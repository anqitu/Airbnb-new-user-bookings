# @TODO actions

# action -----------------------------------------------------------------------
# convert '-unknown-' to NA
# sessions[sessions['action'] == '-unknown-'].shape
# sessions[sessions['action_type'] == '-unknown-'].shape
# sessions[sessions['action_detail'] == '-unknown-'].shape


# action_combo_df = sessions[['action', 'action_type', 'action_detail']].drop_duplicates()
# action_combo_df.sample(10)
# 'action', 'action_type', 'action_detail' can be overlap, Every row has at least one
# sessions[sessions['action'] ==  sessions['action_type']]
# sessions[sessions['action'] ==  sessions['action_detail']]
# sessions[sessions['action_type'] ==  sessions['action_detail']]
# sessions[(pd.isna(sessions['action'])) & (pd.isna(sessions['action_type'])) & (pd.isna(sessions['action_detail']))]
#
#
# sessions_concat = pd.concat([sessions[['id', 'action']],
#                             sessions[['id', 'action_type']].rename(columns = {'action_type': 'action'}),
#                             sessions[['id', 'action_detail']].rename(columns = {'action_detail': 'action'})])
# sessions_concat = sessions_concat.dropna()
# sessions_concat = sessions_concat.drop_duplicates()
# action_count_df = get_percentage(data = sessions_concat, column = 'action')
# action_count_df.to_csv('prepared_data/actions.csv', index = False)
# pd.read_csv('prepared_data/actions.csv')['action_category'].value_counts()

# Map actions to self-defined categories
# action_count_df.iloc[119:]['%'].sum()
# convert those out of the 95 percentile to others
# convert_minority_to_others(data = sessions_concat, column_name = 'action', minority_counts = 308)
# test_df = get_percentage(sessions_concat, 'action_min_to_other')
# test_df
# test_df[test_df['action_min_to_other'] == 'other']
# test_df.sample(10)
#
# action_concat_df = sessions_concat.groupby(['id'])['action_min_to_other'].apply(lambda x: '|'.join(x)).reset_index()
# action_encoded_df = action_concat_df['action_min_to_other'].str.get_dummies(sep='|')
# action_encoded_df.columns = ['action_' + column for column in action_encoded_df.columns]
# action_encoded_df['id'] = action_concat_df['id']


# Age --------------------------------------------------------------------------
# users[~pd.isna(users['age'])].sort_values(['age'])
# users[users['age'] > 100].sort_values(['age'])
# users[users['age'] > 100]['age'].hist()
# users[users['age'] > 1500].sort_values(['age'])
# users[users['age'] > 1500]['age'].describe()
# users[users['age'] < 20]['age'].hist()
# users[users['age'] < 15].sort_values(['age'])
# users[(users['age'] > 15) & (users['age'] < 100)]['age'].hist()
# users[(users['age'] > 15) & (users['age'] < 100)]['age'].describe()
# users[(users['age'] > 15) & (users['age'] < 100)].shape
# users[~pd.isna(users['age'])].shape
