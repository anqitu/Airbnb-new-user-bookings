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



# # RFECV: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
# import time
# start_total = time.perf_counter()
#
# estimators = {clf.__class__.__name__: clf for clf in [LogisticRegression(random_state = SEED, solver = 'sag', multi_class = 'ovr'),
#                                                     DecisionTreeClassifier(random_state = SEED),
#                                                     RandomForestClassifier(random_state = SEED),
#                                                     # XGBClassifier(seed = SEED)
#                                                     ]}
# estimators_RFECV = {}
# for estimatorName, estimator in estimators.items():
#     print(current_time() + ': Start training ' + estimatorName)
#     start = time.perf_counter()
#     rfecv = RFECV(estimator = estimator, scoring = 'accuracy', cv = 5, n_jobs = -1)
#     rfecv_result = rfecv.fit(x_train, y_train)
#     run = time.perf_counter() - start
#
#     print('{} runs for {:.2f} seconds.'.format(estimator.__class__.__name__, run))
#     estimators_RFECV[estimatorName] = rfecv_result
#
#     rfe_result_rank = pd.DataFrame(data = {'Ranking': rfecv_result.ranking_, 'Column': x_train.columns}).sort_values('Ranking')
#     check_dir(TRAIN_RESULT_PATH)
#     rfe_result_rank.to_csv(os.path.join(TRAIN_RESULT_PATH,'RFECV_Ranking_for_{}.csv').format(estimator.__class__.__name__), index = False)
#
#     plot_estimator_no_feature_vs_accuracy_score(rfecv_result, estimator)
#
# run_total = time.perf_counter() - start_total
# print('Total running time was {:.2f} minutes.'.format(run_total/60))


# estimator = LogisticRegression(random_state = SEED)
#
# estimatorName = estimator.__class__.__name__
# selector = RFECV(estimator, step=1, cv=3, scoring='accuracy', n_jobs = -1)
# grid_searcher = GridSearchCV(selector, param_grid_lr, **gridsearch_param)
#
# print(current_time() + ': Start training ' + estimatorName)
# start = time.perf_counter()
# grid_searcher.fit(x_train, y_train)
# run = time.perf_counter() - start
# print('{} runs for {:.2f} seconds.'.format(estimatorName, run))
#
# rfecv_result = grid_searcher.best_estimator_
# estimators_RFECV[estimatorName] = rfecv_result
# rfe_result_rank = pd.DataFrame(data = {'Ranking': rfecv_result.ranking_, 'Column': x_train.columns}).sort_values('Ranking')
# plot_estimator_no_feature_vs_accuracy_score(rfecv_result, estimatorName)
#
# estimators_params[estimatorName]= grid_searcher.best_params_

# xgb = XGBClassifier(max_depth= 6, learning_rate=0.3, n_estimators=25,
#                     objective='multi:softprob', subsample=0.5, colsample_bytree=0.5)


# """#### Ensembling"""
# test_prob_files = [file for file in os.listdir(WORKING_DIR) if file.startswith('Prob_Test_') and 'transformed' not in file]
# prob_dfs = []
# for prob_file in test_prob_files:
#     df = pd.read_csv(prob_file)
#     model_name = prob_file.replace('Prob_Test_', '').replace('.csv', '')
#     df.columns = [model_name + '_' + str(i) for i in range(12)]
#     prob_dfs.append(df)
#
# x_test_meta = pd.concat(prob_dfs, axis = 1)
# y_test_meta = y_test
#
# xgb_stacking = XGBClassifier(max_depth= 6, learning_rate=0.3, n_estimators=25,
#                             objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=SEED)
# xgb_stacking.fit(x_test_meta, y_test_meta)
#
# y_test_pred_meta = xgb_stacking.predict(x_test_meta)
# print(accuracy_score(y_test_meta, y_test_pred_meta))
# save_model(xgb_stacking, 'XGBClassifier_Stacking')
#
#
# val_prob_files = [file for file in os.listdir(WORKING_DIR) if file.startswith('Prob_Val_') and 'transformed' not in file]
# prob_dfs = []
# for prob_file in val_prob_files:
#     df = pd.read_csv(prob_file)
#     model_name = prob_file.replace('Prob_Val_', '').replace('.csv', '')
#     df.columns = [model_name + '_' + str(i) for i in range(12)]
#     prob_dfs.append(df)
#
# x_val_meta = pd.concat(prob_dfs, axis = 1)
# x_val_meta = x_val_meta[x_test_meta.columns]
# y_val_pred = xgb_stacking.predict(x_val_meta)
# print(accuracy_score(y_val, y_val_pred))
#
# get_matrix(y_test = y_val, y_test_pred = y_val_pred,
#             y_train = y_test_meta, y_train_pred = y_test_pred_meta,
#             estimator_name = 'XGBClassifier_Stacking', label_encoder = label_encoder)
#
# # 0.6529168990518683
# # 0.6287713161346743
