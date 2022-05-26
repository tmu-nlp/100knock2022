import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

X_train = pd.read_table("train.feature.txt")
X_valid = pd.read_table("valid.feature.txt")
X_test = pd.read_table("test.feature.txt")
Y_train = pd.read_table("train.txt")["CATEGORY"]
Y_valid = pd.read_table("valid.txt")["CATEGORY"]
Y_test = pd.read_table("test.txt")["CATEGORY"]
train = pd.read_table("train.txt")
valid = pd.read_table("valid.txt")
test = pd.read_table("test.txt")
param_grid = {'C': [i for i in range(1, 21)]}

grid_search = GridSearchCV(LogisticRegression(
    random_state=20010101, max_iter=1000), param_grid, cv=5)
grid_search.fit(X_train, Y_train)

print("Best parameters : {}".format(grid_search.best_params_))
print("Best cross-validation score : {:.3f}".format(grid_search.best_score_))

"""
Best parameters : {'C': 5}
Best cross-validation score : 0.882
"""
