import pickle
from sys import displayhook
import pandas as pd
import numpy as np

# ロジスティック回帰の結果をロード
lr = pickle.load(open("logistic.sav", "rb"))
# 訓練データの特徴量をロード
X_train = pd.read_table("train.feture.txt")
# 特徴量がついている名前を抽出
fetures = X_train.columns.values


for c, coef in zip(lr.classes_, lr.coef_):
    best10 = pd.DataFrame(fetures[np.argsort(
        coef)[::-1][:10]], columns=["重要特徴量TOP10"], index=[i for i in range(1, 11)]).T
    low10 = pd.DataFrame(fetures[np.argsort(coef)[:10]], columns=[
                         "非重要特徴量TOP10"], index=[i for i in range(1, 11)]).T
    print(c)
    print(pd.concat([best10, low10], axis=0))
