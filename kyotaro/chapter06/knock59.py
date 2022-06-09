"""
学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．
検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．
また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．
"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from knock52 import X_train, train
from knock53 import score_lr
import matplotlib.pyplot as plt
import pandas as pd

# テストデータと訓練データの特徴量を取得
train_feture = pd.read_table("train.feture.txt")
valid_feture = pd.read_table("valid.feture.txt")
test_feture = pd.read_table("test.feture.txt")

# kernelの設定のリスト
kernel_names = ['linear', 'rbf', 'poly', 'sigmoid']

for kernel in kernel_names:
    # SVM
    svm = SVC(C=1.0, kernel=kernel, gamma=0.01, max_iter=1000)
    svm.fit(X_train, train["CATEGORY"])

    # 予測確率
    train_predict_svm = score_lr(svm, train_feture)
    valid_predict_svm = score_lr(svm, valid_feture)
    test_predict_svm = score_lr(svm, test_feture)

    # 正解のタグ
    train_ans = pd.read_csv("train.txt", sep="\t")["CATEGORY"]
    valid_ans = pd.read_csv("valid.txt", sep="\t")["CATEGORY"]
    test_ans = pd.read_csv("test.txt", sep="\t")["CATEGORY"]

    # 自分で予測したタグ
    train_myans_svm = train_predict_svm[1]
    valid_myans_svm = valid_predict_svm[1]
    test_myans_svm = test_predict_svm[1]

    # 正解率の計算
    train_accuracy_svm = accuracy_score(train_ans, train_myans_svm)
    valid_accuracy_svm = accuracy_score(valid_ans, valid_myans_svm)
    test_accuracy_svm = accuracy_score(test_ans, test_myans_svm)

    # 出力
    print(kernel)
    print(f'train accuracy = {train_accuracy_svm}%')
    print(f'valid accuracy = {valid_accuracy_svm}%')
    print(f'test accuracy = {test_accuracy_svm}%')


"""
train accuracy = 0.9263493253373314%
valid accuracy = 0.8860569715142429%
test accuracy = 0.8620689655172413%
"""
