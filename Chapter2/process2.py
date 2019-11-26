import numpy as np
import pandas as pd
from Chapter2.preparation import data_division
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X: np, y=None) -> np.ndarray:
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
    housing = train_set.drop('median_house_value', axis=1)
    housing_num = housing.drop('ocean_proximity', axis=1)

    # attributes acquisition
    num_attribs = list(housing_num)
    cat_attribs = ['ocean_proximity']

    # pipeline for numbers
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    # pipeline for all
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs),
    ])

    X_train: np.ndarray = full_pipeline.fit_transform(housing)
    y_train: pd.Series = train_set['median_house_value'].copy()

    X_test = test_set.drop('median_house_value', axis=1)
    X_test: np.ndarray = full_pipeline.transform(X_test)
    y_test: pd.Series = test_set['median_house_value'].copy()

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    housing_prepared = pipeline()[0]