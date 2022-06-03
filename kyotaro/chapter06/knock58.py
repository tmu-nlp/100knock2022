"""
ロジスティック回帰モデルを学習するとき,正則化パラメータを調整することで,学習時の過学習（overfitting）の度合いを制御できる.
異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，および評価データ上の正解率を求めよ．
実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．
"""

from cProfile import label
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from knock52 import X_train, train
from knock53 import score_lr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# テストデータと訓練データの特徴量を取得
train_feture = pd.read_table("train.feture.txt")
valid_feture = pd.read_table("valid.feture.txt")
test_feture = pd.read_table("test.feture.txt")

# 正則化係数のリスト
c_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
result = []

for c in c_list:
    lrc = LogisticRegression(random_state=1013, max_iter=1000, C=c)
    lrc.fit(X_train, train["CATEGORY"])

    # 予測確率
    train_predict_c = score_lr(lrc, train_feture)
    valid_predict_c = score_lr(lrc, valid_feture)
    test_predict_c = score_lr(lrc, test_feture)

    # 正解のタグ
    train_ans = pd.read_csv("train.txt", sep="\t")["CATEGORY"]
    valid_ans = pd.read_csv("valid.txt", sep="\t")["CATEGORY"]
    test_ans = pd.read_csv("test.txt", sep="\t")["CATEGORY"]

    # 自分で予測したタグ
    train_myans_c = train_predict_c[1]
    valid_myans_c = valid_predict_c[1]
    test_myans_c = test_predict_c[1]

    # 正解率の計算
    train_accuracy_c = accuracy_score(train_ans, train_myans_c)
    valid_accuracy_c = accuracy_score(valid_ans, valid_myans_c)
    test_accuracy_c = accuracy_score(test_ans, test_myans_c)

    # 結果をまとめる
    # 0番目に正則化係数、1番目に訓練、2番目に検証、3番目にテスト
    result.append([c, train_accuracy_c, valid_accuracy_c, test_accuracy_c])

# グラフで出力
result = np.array(result).T
plt.plot(result[0], result[1], label="train")
plt.plot(result[0], result[2], label="valid")
plt.plot(result[0], result[3], label="test")
plt.xscale('log')
plt.xlabel("C")
plt.ylim(0, 1.1)
plt.ylabel("Accuracy")
plt.legend()
plt.show()
