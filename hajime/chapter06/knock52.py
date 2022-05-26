import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

lr = LogisticRegression(random_state=20010101, max_iter=1000)

X_train = pd.read_table("train.feature.txt")
Y_train = pd.read_table("train.txt")["CATEGORY"]

# print(len(X_train))
# print(len(Y_train))
lr.fit(X_train, Y_train)

with open("logreg.pkl", "wb") as f:
    pickle.dump(lr, f)
