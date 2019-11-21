import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def data_division():
    # data stratified
    housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
    housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
    # stratified splitting
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    strat_train_set = strat_test_set = pd.DataFrame
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    # restore data
    strat_train_set.drop(['income_cat'], axis=1, inplace=True)
    strat_test_set.drop(['income_cat'], axis=1, inplace=True)
    return strat_train_set, strat_test_set


if __name__ == '__main__':
    housing = pd.read_csv('housing.csv')
    # randomly create train/test set
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    # data stratified
    housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
    housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
    # print(housing['income_cat'].value_counts(normalize=True))

    # stratified splitting
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    strat_train_set = strat_test_set = pd.DataFrame
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # restore data
    # for set in (strat_train_set, strat_test_set):
    #     set.drop(["income_cat"], axis=1, inplace=True)
    strat_train_set.drop(['income_cat'], axis=1, inplace=True)
    strat_test_set.drop(['income_cat'], axis=1, inplace=True)

    # observe correlation between attributions
    housing = strat_train_set
    corr_matrix = housing.corr()
    print(corr_matrix['median_house_value'].sort_values(ascending=False))
    housing.plot(kind='scatter', x='median_income', y='median_house_value')

    # attributes combination
    housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
    housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
    housing['population_per_household'] = housing['population'] / housing['households']
    corr_matrix = housing.corr()
    print()
    print(corr_matrix['median_house_value'].sort_values(ascending=False))

    plt.show()
