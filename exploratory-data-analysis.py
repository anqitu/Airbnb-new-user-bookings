# Exploratory Data Analysis ----------------------------------------------------
def plot_catogory_distribution(data, column_name, title = None):
    data[column_name].value_counts(dropna=False).plot(kind='bar', rot=0, color = 'c')
    plt.title(title)
    plt.show()

def plot_continuous_distribution(data, column_name, title = None, bins = None):
    sns.distplot(combine_users['age'].dropna(), bins = bins)
    plt.title(title)
    plt.show()


def plot_category_stack(data, column_name):
    pivot_df = data.groupby(['country_destination', column_name])['date_account_created'].count().reset_index().pivot(index='country_destination', columns=column_name, values='date_account_created')
    pivot_df.plot.bar(stacked=True)

def plot_category_bar(data, column_name):
    sns.catplot(data = data, x = "country_destination", hue = column_name, kind = "count", height = 8, aspect = 1.5)

def plot_train_test_diff(data, column_name):
    pivot_df = data.groupby([column_name, 'data'])['date_account_created'].count().reset_index().pivot(index=column_name, columns='data', values='date_account_created')
    pivot_df.plot.bar(stacked=True)


# for categorical_feature in categorical_features:
#     plot_catogory_distribution(data = combine_users, column_name = categorical_feature, title = categorical_feature)
# plot_catogory_distribution(combine_users[combine_users['data'] == 'train'], 'country_destination', title = None)

# plot_continuous_distribution(data = combine_users, column_name = 'age', title = 'Age', bins = 20)

# for categorical_feature in categorical_features:
#     plot_category_stack(data = combine_users, column_name = categorical_feature)

# for categorical_feature in ['gender', 'age_missing']:
#     plot_category_bar(data = combine_users, column_name = categorical_feature)

# for categorical_feature in categorical_features:
#     plot_train_test_diff(data = combine_users, column_name = categorical_feature)

# combine_users.date_account_created.value_counts().plot(kind='line', linewidth=1)
# combine_users.date_first_active.value_counts().plot(kind='line', linewidth=1)


# Datatime Trend ---------------------------------------------------------------
# users_2013 = combine_users[combine_users['date_first_active'] > pd.to_datetime(20130101, format='%Y%m%d')]
# users_2013 = users_2013[users_2013['date_first_active'] < pd.to_datetime(20140101, format='%Y%m%d')]
# users_2013.date_first_active.value_counts().plot(kind='line', linewidth=2, color='#FD5C64')
# plt.show()
#
# weekdays = []
# for date in combine_users.date_account_created:
#     weekdays.append(date.weekday())
# weekdays = pd.Series(weekdays)
# sns.barplot(x = weekdays.value_counts().index, y=weekdays.value_counts().values, order=range(0,7))
# plt.xlabel('Week Day')
# sns.despine()
#
#
# users = combine_users
# date = pd.to_datetime(20140101, format='%Y%m%d')
#
# before = sum(users.loc[users['date_first_active'] < date, 'country_destination'].value_counts())
# after = sum(users.loc[users['date_first_active'] > date, 'country_destination'].value_counts())
# before_destinations = users.loc[users['date_first_active'] < date,
#                                 'country_destination'].value_counts() / before * 100
# after_destinations = users.loc[users['date_first_active'] > date,
#                                'country_destination'].value_counts() / after * 100
# before_destinations.plot(kind='bar', width=5, color='#63EA55', position=0, label='Before 2014', rot=0)
# after_destinations.plot(kind='bar', width=5, color='#4DD3C9', position=1, label='After 2014', rot=0)
#
# plt.legend()
# plt.xlabel('Destination Country')
# plt.ylabel('Percentage')
#
# sns.despine()
# plt.show()
