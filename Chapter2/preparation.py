import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def data_division():
    housing = pd.read_csv('housing.csv')
    # data stratified
    housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
    housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
    # stratified splitting
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_set = test_set = pd.DataFrame
    for train_index, test_index in split.split(housing, housing['income_cat']):
        train_set = housing.loc[train_index]
        test_set = housing.loc[test_index]
    # restore data
    train_set.drop(['income_cat'], axis=1, inplace=True)
    test_set.drop(['income_cat'], axis=1, inplace=True)
    return train_set, test_set


if __name__ == '__main__':
    housing = pd.read_csv('housing.csv')
    # randomly create train/test set
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    # create train/test set by stratifying
    strat_train_set, strat_test_set = data_division()

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
