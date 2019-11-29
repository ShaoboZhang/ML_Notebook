from Chapter2.process2 import pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
from time import time


def display_scores(scores: np.ndarray):
    print('mean: %.3f' % scores.mean())
    print('std:  %.3f' % scores.std())


def model_comparision():
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
    print('train set rmse: %.3f' % np.sqrt(lin_mse))
    lin_predict = lin_reg.predict(X_test)
    lin_mse = mean_squared_error(y_test, lin_predict)
    print('test set rmse: %.3f ' % np.sqrt(lin_mse))
    # cross validation
    lin_scores = cross_val_score(lin_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    print('10-fold cv'.center(25))
    display_scores(lin_rmse_scores)
    # process time
    print('processing time: %.3fs\n' % (time() - t0))
    # --------------------------------linear model-------------------------------- #

    # -------------------------------decision tree-------------------------------- #
    t0 = time()
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X_train, y_train)
    # performance
    tree_predict = tree_reg.predict(X_train)
    tree_mse = mean_squared_error(y_train, tree_predict)
    print('decision tree'.center(25, '-'))
    print('train set rmse: %.3f' % np.sqrt(tree_mse))
    tree_predict = tree_reg.predict(X_test)
    tree_mse = mean_squared_error(y_test, tree_predict)
    print('test set rmse: %.3f ' % np.sqrt(tree_mse))
    # cross validation
    tree_scores = cross_val_score(tree_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
    tree_rmse_scores = np.sqrt(-tree_scores)
    print('10-fold cv'.center(25))
    display_scores(tree_rmse_scores)
    # process time
    print('processing time: %.3fs\n' % (time() - t0))
    # -------------------------------decision tree-------------------------------- #

    # -------------------------------random forest-------------------------------- #
    t0 = time()
    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(X_train, y_train)
    # performance
    forest_predict = forest_reg.predict(X_train)
    forest_mse = mean_squared_error(y_train, forest_predict)
    print('random forest'.center(25, '-'))
    print('train set rmse: %.3f' % np.sqrt(forest_mse))
    forest_predict = forest_reg.predict(X_test)
    forest_mse = mean_squared_error(y_test, forest_predict)
    print('test set rmse: %.3f ' % np.sqrt(forest_mse))
    # cross validation
    forest_scores = cross_val_score(forest_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    print('10-fold cv'.center(25))
    display_scores(forest_rmse_scores)
    # process time
    print('processing time: %.3fs' % (time() - t0))
    # -------------------------------random forest-------------------------------- #


def grid_search():
    # data acquisition
    X_train, X_test, y_train, y_test = pipeline()

    t0 = time()
    # grid search
    forest_reg = RandomForestRegressor(random_state=42)
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train, y_train)

    # best score
    print('best parameters: ', grid_search.best_params_)
    print('best score: '.ljust(17), np.sqrt(-grid_search.best_score_))

    # prediction performance
    forest_predict = grid_search.predict(X_test)
    forest_mse = mean_squared_error(y_test, forest_predict)
    print('test set rmse:'.ljust(17), '%.3f ' % np.sqrt(forest_mse))

    # model save
    joblib.dump(grid_search.best_estimator_, 'forest_model.pkl')

    # process time
    print('processing time:'.ljust(17) ,'%.3fs' % (time() - t0))


if __name__ == '__main__':
    # model_comparision()

    # grid_search()

    X_train, X_test, y_train, y_test = pipeline()
    forest_reg = joblib.load('forest_model.pkl')
    # performance
    forest_predict = forest_reg.predict(X_test)
    forest_mse = mean_squared_error(y_test, forest_predict)
    print('test set rmse: %.3f ' % np.sqrt(forest_mse))
    pass
