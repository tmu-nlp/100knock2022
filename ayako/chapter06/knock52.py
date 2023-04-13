# knock52
# 51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ.
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    #X="TITLE"からY="CATEGORY"を予測
    X_train = pd.read_table("train.feature.txt", header=None) #各単語の特徴量
    Y_train = pd.read_table("train.txt", header=None)[1]#分類の正解ラベル

    #デフォルトだと警告出た（収束しなかった？）からmax_iter=1000にした
    lr = LogisticRegression(random_state=0, max_iter=1000)#ロジスティック回帰モデルのインスタンスを生成
    lr.fit(X_train, Y_train)#ロジスティック回帰モデルの重みを学習

    #モデルを保存
    with open("model.pkl", "wb") as f:
        pickle.dump(lr, f)