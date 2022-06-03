"""
ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，学習時の過学習（overfitting）の度合いを制御できる．
異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，および評価データ上の正解率を求めよ．
実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．
"""

from sklearn.linear_model import LogisticRegression
from knock51 import X_train, X_test, X_valid, train, test, valid
from knock52 import lr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

score_train = []
score_test = []
score_valid = []

C_list = np.logspace(-5, 4, 10, base=10)
for C in C_list:
    lr = LogisticRegression(C=C, max_iter=10000)
    lr.fit(X_train, train["CATEGORY"])
    score_train.append(lr.score(X_train, train["CATEGORY"]))
    score_test.append(lr.score(X_test, test["CATEGORY"]))
    score_valid.append(lr.score(X_valid, valid["CATEGORY"]))

fig, ax = plt.subplots()
ax.plot(C_list, score_train)
ax.plot(C_list, score_test)
ax.plot(C_list, score_valid)
plt.xscale('log')
ax.set_xlabel('C')
ax.set_ylabel('Accuracy')
plt.savefig('Figure_1.png')
plt.show()
