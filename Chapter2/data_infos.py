import pandas as pd


if __name__ == '__main__':
    housing = pd.read_csv('housing.csv')
    print(housing.head())
    print(housing.info())
    print(housing['ocean_proximity'].value_counts())