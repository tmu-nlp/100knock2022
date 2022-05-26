from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
import pandas as pd


def score_lr(lr, x):
    return [np.max(lr.predict_proba(x), axis=1), lr.predict(x)]


lr = pickle.load(open("logreg.pkl", 'rb'))

X_train = pd.read_table("train.feature.txt")
X_test = pd.read_table("test.feature.txt")
train = pd.read_table("train.txt")
test = pd.read_table("test.txt")

train_pred = score_lr(lr, X_train)
test_pred = score_lr(lr, X_test)

if __name__ == "__main__":
    print(train_pred)

"""
[array([0.95639054, 0.95150829, 0.7172343 , ..., 0.41618661, 0.79069576,
       0.40435474]), array(['e', 'e', 't', ..., 'b', 'b', 'b'], dtype=object)]

"""
