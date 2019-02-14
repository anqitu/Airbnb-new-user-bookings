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


# def plot_estimator_no_feature_vs_accuracy_score(rfecv_result, estimatorName, title = None, save = False, show = True):
#     plt.figure()
#     plt.xlabel("Number of features selected")
#     plt.ylabel("Cross validation score (Accuracy)")
#     plt.plot(range(1, len(rfecv_result.grid_scores_) + 1), rfecv_result.grid_scores_, figsize=(PLOT_WIDTH, PLOT_HEIGHT))
#     plt.ylim([0.4,0.7])
#
#     if title is None:
#         title = 'Number of Features vs Accuracy Score for ' + estimatorName
#     plt.title(title, loc = 'center', y=1.1, fontsize = 25)
#
#     if save:
#         check_dir(IMAGE_GENERAL_DIRECTORY)
#         saved_path = os.path.join(IMAGE_GENERAL_DIRECTORY, convert_title_to_filename(title))
#         plt.savefig(saved_path, dpi=200, bbox_inches="tight")
#         print('Saved to {}'.format(saved_path))
#     if show:
#         plt.show()
#
#     plt.close()




# y, X = dmatrices( 'country_destination ~ ' + ' + '.join(cols), data=x_train, return_type='dataframe')
# X = sm.add_constant(X, prepend = False)
# mdl = MNLogit(X, y)
# mdl_fit = mdl.fit(method='bfgs')


# """#### Light GBM"""
# import lightgbm as lgbm
#
# lgb_params =  {
#     'task': 'train',
#     'nthread': -1,
#     'boosting_type': 'gbdt',
#     'objective': 'multiclass',
#     'num_class': 11,
#     'seed': SEED,
#     'metric': ['multi_logloss'],
#     'max_depth': 8}
#
# x_train_lgb = lgbm.Dataset(x_train, label=y_train, categorical_feature = [])
# model = LGBMClassifier(boosting_type='gbdt', objective='multiclass',
#                        num_class=9,early_stopping = 50,num_iteration=10000,num_leaves=31,
#                        is_enable_sparse='true',tree_learner='data',min_data_in_leaf=600,max_depth=4,
#                        learning_rate=0.01, n_estimators=675, max_bin=255, subsample_for_bin=50000,
#                        min_split_gain=5, min_child_weight=5, min_child_samples=10, subsample=0.995,
#                        subsample_freq=1, colsample_bytree=1, reg_alpha=0,
#                        reg_lambda=0, seed=0, nthread=-1, silent=True)
#
# #Fit to training data
# model.fit(x_train_lgb)
# #Generate Predictions
# y_pred=model.predict_proba(X_test)
#
#
# cv_result_lgb = lgbm.cv(params = lgb_params,
#                         train_set = x_train_lgb,
#                         num_boost_round=1000,
#                         nfold=5,
#                         stratified=True,
#                         early_stopping_rounds=50,
#                         verbose_eval=100,
#                         show_stdv=True)
#
# num_boost_rounds_lgb = len(cv_result_lgb['multi_logloss-mean'])
# print('num_boost_rounds_lgb=' + str(num_boost_rounds_lgb))
#
# x_train_lgb = lgbm.Dataset(x_train, label=y_train, categorical_feature = [])
# lgb.fit(lgb_params, x_train_lgb, num_boost_round=num_boost_rounds_lgb)
#
# lgb.feature_importances_
# y_test_prob = lgb.predict(x_test)
# np.array([accuracy_score(get_prob_top_n(y_test_prob, n), y_test) for n in range(11)]).cumsum()
#
#
# param_grid_lg = {"max_depth": [25,50, 75],
#               "learning_rate" : [0.01,0.05,0.1],
#               "num_leaves": [300,900,1200],
#               "n_estimators": [200]
#              }
# param_grid_lg = {"max_depth": [10], "num_leaves": [300], "n_estimators": [20]}
# gridsearcher_rfc = gridsearch_rfecv_estimator(estimator = lgb, param_grid = param_grid_lg, rfecv = False)
#
# lgb = LGBMClassifier(**load_obj('GridSearch_Best_Params_LGBMClassifier'))
# lgb.fit(x_train, y_train, categorical_feature=[])
# lgb.predict_proba(x_test)
#
#
# x_train_lgb = lgbm.Dataset(x_train, label=y_train, categorical_feature = [])
# cv_result_lgb = lgbm.cv(param_grid_lg,
#                        x_train_lgb,
#                        num_boost_round=1000,
#                        nfold=5,
#                        stratified=True,
#                        early_stopping_rounds=50,
#                        verbose_eval=100,
#                        show_stdv=True, objective= 'multiclass',
#                        num_class = 11,
#                        metric = 'multi_logloss')
# num_boost_rounds_lgb = len(cv_result_lgb['multi_logloss-mean'])
# print('num_boost_rounds_lgb=' + str(num_boost_rounds_lgb))
# # train model
# model_lgb = lgbm.train(lgb_params, x_train_lgb, num_boost_round=num_boost_rounds_lgb)
# model_lgb.feature_importance
# y_pred=model_lgb.predict(test)
# classes = "class1,class2,class3,class4,class5,class6,class7,class8,class9".split(',')
# subm = pd.DataFrame(y_pred, columns=classes)
# subm['ID'] = pid
# model_lgb.feature_importance
#
#
# # Without Categorical Features
# model2 = lgb.train(params, d_train)
# lg.predict_proba(x_test)
#
# lgb_cv = lgbm.cv(params, d_train, num_boost_round=10000, nfold=3, shuffle=True, stratified=True, verbose_eval=20, early_stopping_rounds=100)
#
# nround = lgb_cv['multi_logloss-mean'].index(np.min(lgb_cv['multi_logloss-mean']))
# print(nround)
#
# model = lgbm.train(params, d_train, num_boost_round=nround)
#
# #With Catgeorical Features
# cate_features_name = ["MONTH","DAY","DAY_OF_WEEK","AIRLINE","DESTINATION_AIRPORT",
#                  "ORIGIN_AIRPORT"]
# model2 = lgb.train(params, d_train, categorical_feature = cate_features_name)
# auc2(model2, train, test)
#


# from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
# import itertools
# def plot_confusion_matrix(cm, classes, normalize=False, title=None, cmap=plt.cm.Blues, save = False, show = True):
#     fig = plt.figure(facecolor='w', figsize=(PLOT_WIDTH, PLOT_HEIGHT))
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap, )
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     if title is None:
#         title = 'Confusion Matrix'
#         if normalize:
#             title = title + ' (Normalized)'
#     plt.title(title, loc = 'center', y=1.15, fontsize = 25)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#     if save:
#         check_dir(IMAGE_MATRIX_DIRECTORY)
#         saved_path = os.path.join(IMAGE_MATRIX_DIRECTORY, convert_title_to_filename(title))
#         plt.savefig(saved_path, dpi=200, bbox_inches="tight")
#         plt.show()
#         print('Saved to {}'.format(saved_path))
#     if show:
#         plt.show()
#
#     plt.close()
#
#
# def get_matrix(y_test, y_test_pred, y_train, y_train_pred, estimator_name, label_encoder):
#     check_dir(TRAIN_RESULT_PATH)
#
#     title = 'Confusion Matrix for ' + estimator_name + ' Test'
#     df_confusion = pd.crosstab(pd.Series(label_encoder.inverse_transform(y_test), name='True'), pd.Series(label_encoder.inverse_transform(y_test_pred), name='Predict'))
#     df_confusion.to_csv(os.path.join(TRAIN_RESULT_PATH, convert_title_to_filename(title) + '.csv'))
#     plot_confusion_matrix(confusion_matrix(y_test, y_test_pred), label_encoder.classes_, title = title, save = True)
#
#     title = 'Confusion Matrix for ' + estimator_name  + ' Train'
#
#     df_confusion = pd.crosstab(pd.Series(label_encoder.inverse_transform(y_train), name='True'), pd.Series(label_encoder.inverse_transform(y_train_pred), name='Predict'))
#     df_confusion.to_csv(os.path.join(TRAIN_RESULT_PATH, convert_title_to_filename(title) + '.csv'))
#     plot_confusion_matrix(confusion_matrix(y_train, y_train_pred), label_encoder.classes_, title = title, save = True)
#
#
#
#
#
# get_matrix(y_test = y_test, y_test_pred = y_test_pred, y_train = y_train, y_train_pred = y_train_pred, estimator_name = clf_name, label_encoder = label_encoder)


# #language
# language_map = {'en': 'AU|CA|GB|US',
#                 'zh': 'ASIA',
#                 'ko': 'ASIA',
#                 'fr': 'FR',
#                 'es': 'ES',
#                 'de': 'DE',
#                 'ru': 'ASIA',
#                 'it': 'IT',
#                 'ja': 'ASIA',
#                 'pt': 'PT',
#                 'sv': 'EU(Other)',
#                 'nl': 'EU(Other)',
#                 'pl': 'EU(Other)',
#                 'tr': 'EU(Other)',
#                 'da': 'EU(Other)',
#                 'th': 'ASIA',
#                 'cs': 'EU(Other)',
#                 'id': 'ASIA',
#                 'el': 'EU(Other)',
#                 'no': 'EU(Other)',
#                 'fi': 'EU(Other)',
#                 'hu': 'EU(Other)',
#                 'is': 'EU(Other)',
#                 'ca': 'ES',
#                 'hr': 'EU(Other)',
#                 }
#
# users['language_map_country'] = users['language'].map(language_map)
# language_df = users[['id', 'language_map_country']].groupby(['id'])['language_map_country'].apply(lambda x: '|'.join(x)).reset_index()
# language_encoded_df = language_df['language_map_country'].str.get_dummies(sep='|')
# languages_mapped = language_encoded_df.columns
# language_encoded_df['id'] = language_df['id']
# users = users.merge(language_encoded_df, how = 'left')
