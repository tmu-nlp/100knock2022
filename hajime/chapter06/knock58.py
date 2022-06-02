import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def score_lr(lr, x):
    return [np.max(lr.predict_proba(x), axis=1), lr.predict(x)]


def learning_model(x_train, y_train, c):
    lr = LogisticRegression(random_state=20010101, max_iter=1000, C=c)
    lr.fit(x_train, y_train)
    model_name = "logreg" + str(c) + ".pkl"
    with open(model_name, "wb") as f:
        pickle.dump(lr, f)
    return lr


def open_model(c):
    model_name = "logreg" + str(c) + ".pkl"
    with open(model_name, "rb") as f:
        model = pickle.load(f)
    return model


if __name__ == "__main__":
    X_train = pd.read_table("train.feature.txt")
    X_valid = pd.read_table("valid.feature.txt")
    X_test = pd.read_table("test.feature.txt")
    Y_train = pd.read_table("train.txt")["CATEGORY"]
    Y_valid = pd.read_table("valid.txt")["CATEGORY"]
    Y_test = pd.read_table("test.txt")["CATEGORY"]
    train = pd.read_table("train.txt")
    valid = pd.read_table("valid.txt")
    test = pd.read_table("test.txt")
    # 逆数が正則化項に対応するのに注意
    regular = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    train_acu = []
    valid_acu = []
    test_acu = []
    for c in regular:
        # model = learning_model(X_train, Y_train, c)
        model = open_model(c)
        train_pred = score_lr(model, X_train)
        valid_pred = score_lr(model, X_valid)
        test_pred = score_lr(model, X_test)
        train_acu.append(accuracy_score(train["CATEGORY"], train_pred[1]))
        valid_acu.append(accuracy_score(valid["CATEGORY"], valid_pred[1]))
        test_acu.append(accuracy_score(test["CATEGORY"], test_pred[1]))
    plt.plot(regular, train_acu,  marker="o", color="red", label="train")
    plt.plot(regular, valid_acu,  marker="o", color="blue", label="valid")
    plt.plot(regular, test_acu,  marker="o", color="green", label="test")
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.savefig("regular_parameter")
