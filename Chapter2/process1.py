import numpy as np
import pandas as pd
from Chapter2.preparation import data_division
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X: np, y=None):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.household_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def pipeline():
    # data acquisition
    train_set, test_set = data_division()
    X_train_num = train_set.drop(['median_house_value', 'ocean_proximity'], axis=1)
    X_train_cat = train_set[['ocean_proximity']]
    X_test_num = test_set.drop(['median_house_value', 'ocean_proximity'], axis=1)
    X_test_cat = test_set[['ocean_proximity']]

    # data clean
    imputer = SimpleImputer(strategy='median')
    # print(imputer.statistics_)
    imputer.fit(X_train_num)
    X_train_num = imputer.transform(X_train_num)
    imputer.fit(X_test_num)
    X_test_num = imputer.transform(X_test_num)

    # encoder = LabelEncoder()
    encoder = OneHotEncoder(sparse=False)
    X_train_cat = encoder.fit_transform(X_train_cat)
    X_test_cat = encoder.transform(X_test_cat)

    # avoid heavy-tail
    attr_adder = CombinedAttributesAdder()
    X_train_num = attr_adder.transform(X_train_num)
    X_test_num = attr_adder.transform(X_test_num)

    # data scale
    std_scaler = StandardScaler()
    X_train_num = std_scaler.fit_transform(X_train_num)
    X_test_num = std_scaler.transform(X_test_num)

    X_train: np.ndarray = np.c_[X_train_num, X_train_cat]
    y_train: pd.Series = train_set['median_house_value'].copy()

    X_test: np.ndarray = np.c_[X_test_num, X_test_cat]
    y_test: pd.Series = test_set['median_house_value'].copy()

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    housing_prepared = pipeline()[0]