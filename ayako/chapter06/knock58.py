# knock58
# ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，
# 学習時の過学習（overfitting）の度合いを制御できる．
# 異なる正則化パラメータでロジスティック回帰モデルを学習し，
# 学習データ，検証データ，および評価データ上の正解率を求めよ．
# 実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import japanize_matplotlib

if __name__ == "__main__":
    #学習データ
    X_train = pd.read_table("train.feature.txt", header=None)
    Y_train = pd.read_table("train.txt", header=None)[1]
    #検証データ
    X_valid = pd.read_table("valid.feature.txt", header=None)
    Y_valid = pd.read_table("valid.txt", header=None)[1]
    #評価データ
    X_test = pd.read_table("test.feature.txt", header=None)
    Y_test = pd.read_table("test.txt", header=None)[1]

    #正則化パラメータのリスト
    C_list = [1e-2, 1e-1, 1.0, 1e+1, 1e+2]
    
    #各データの正解率を格納するリスト
    train_ac = []
    valid_ac = []
    test_ac = []

    for C in C_list:
        #正則化パラメータを指定して学習（L2正則化）
        lr = LogisticRegression(random_state=0, max_iter=1000, C=C)
        lr.fit(X_train, Y_train)
        #それぞれのデータで正解率を計算してリストに格納
        train_ac.append(accuracy_score(Y_train, lr.predict(X_train)))
        valid_ac.append(accuracy_score(Y_valid, lr.predict(X_valid)))
        test_ac.append(accuracy_score(Y_test, lr.predict(X_test)))

    #結果をグラフに表示
    plt.plot(C_list, train_ac, label="train")
    plt.plot(C_list, valid_ac, label="valid")
    plt.plot(C_list, test_ac, label="test")
    plt.xlabel("正則化パラメータC")
    plt.ylabel("正解率")
    plt.legend()#凡例を表示
    plt.xscale("log")#x軸はlogスケールにする
    plt.savefig("output58.png", format="png")