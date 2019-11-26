# from Chapter2.process2 import pipeline
from Chapter2.process1 import pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
from time import time


def display_scores(scores: np.ndarray):
    print('mean: %.3f' % scores.mean())
    print('std:  %.3f' % scores.std())


def model_selection():
    # data acquisition
    X_train, X_test, y_train, y_test = pipeline()

    # --------------------------------linear model-------------------------------- #
    t0 = time()
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    # performance
    lin_predict = lin_reg.predict(X_train)
    lin_mse = mean_squared_error(y_train, lin_predict)
    print('linear model'.center(25, '-'))
    print('rmse: %.3f' % np.sqrt(lin_mse))
    # cross validation
    lin_scores = cross_val_score(lin_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    print('10-cross validation')
    display_scores(lin_rmse_scores)
    # process time
    print('processing time: %.3fs\n' % (time() - t0))
    # --------------------------------linear model-------------------------------- #

    # # -------------------------------decision tree-------------------------------- #
    # t0 = time()
    # tree_reg = DecisionTreeRegressor(random_state=42)
    # tree_reg.fit(X_train, y_train)
    # # performance
    # tree_predict = tree_reg.predict(X_train)
    # tree_mse = mean_squared_error(y_train, tree_predict)
    # print('decision tree'.center(25, '-'))
    # print('rmse: %.3f' % np.sqrt(tree_mse))
    # # cross validation
    # tree_scores = cross_val_score(tree_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
    # tree_rmse_scores = np.sqrt(-tree_scores)
    # print('10-cross validation')
    # display_scores(tree_rmse_scores)
    # # process time
    # print('processing time: %.3fs\n' % (time() - t0))
    # # -------------------------------decision tree-------------------------------- #
    #
    # # -------------------------------random forest-------------------------------- #
    # t0 = time()
    # forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    # forest_reg.fit(X_train, y_train)
    # # performance
    # forest_predict = forest_reg.predict(X_train)
    # forest_mse = mean_squared_error(y_train, forest_predict)
    # print('random forest'.center(25, '-'))
    # print('rmse: %.3f' % np.sqrt(forest_mse))
    # # cross validation
    # forest_scores = cross_val_score(forest_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
    # forest_rmse_scores = np.sqrt(-forest_scores)
    # print('10-cross validation')
    # display_scores(forest_rmse_scores)
    # # process time
    # print('processing time: %.3fs' % (time() - t0))
    # # -------------------------------random forest-------------------------------- #


if __name__ == '__main__':
    # model_selection()
    X_train, X_test, y_train, y_test = pipeline()
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    # performance
    lin_predict = lin_reg.predict(X_test)
    lin_mse = mean_squared_error(y_test, lin_predict)
    print('linear model'.center(25, '-'))
    print('rmse: %.3f' % np.sqrt(lin_mse))