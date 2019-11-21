# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
from Chapter2.preparation import data_division
# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    housing, test = data_division()
    # Data clean
    # housing_num = housing.drop(labels='ocean_proximity', axis=1)
    # imputer = SimpleImputer(strategy='median')
    # imputer.fit(housing_num)
    # # print(imputer.statistics_)
    # X = imputer.transform(housing_num)
    # print(imputer.strategy)

    # one-hot encoding
    housing_cat = housing['ocean_proximity']
    encoder = OneHotEncoder(sparse=True)
    housing_cat_1hot = encoder.fit_transform([housing_cat])
    print(housing_cat_1hot)
    pass