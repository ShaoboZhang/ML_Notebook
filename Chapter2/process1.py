import numpy as np
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


if __name__ == '__main__':
    # data acquisition
    housing, test = data_division()
    housing_num = housing.drop(labels='ocean_proximity', axis=1)
    housing_cat = housing[['ocean_proximity']]

    # data clean
    imputer = SimpleImputer(strategy='median')
    imputer.fit(housing_num)
    # print(imputer.statistics_)
    X = imputer.transform(housing_num)

    # encoder = LabelEncoder()
    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat)

    # avoid heavy-tail
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    X = attr_adder.transform(X)

    # data scale
    std_scaler = StandardScaler()
    X = std_scaler.fit_transform(X)