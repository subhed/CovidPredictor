
import numpy as np
import pickle
import pandas as pd
import pandas.util.testing as tm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


if __name__== "__main__":

    df = pd.read_csv('data/Data12.csv')

    x = df.iloc[:, :-1].values
    y = df.iloc[:, 1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=101)


    gbc = GradientBoostingRegressor(n_estimators=200, learning_rate=0.5, max_depth=10)
    gbc.fit(x_train, y_train)

    file = open('model/model_GBR.pkl','wb')

    pickle.dump(gbc, file)
    file.close()


