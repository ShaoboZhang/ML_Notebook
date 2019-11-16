import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    housing = pd.read_csv('housing.csv')
    housing.hist(bins=50, figsize=(12,8))
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"] / 100,
                 label="population", c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.legend()
    plt.show()