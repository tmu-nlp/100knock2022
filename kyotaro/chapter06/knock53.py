import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

# train, test の特徴量とロジスティック回帰の結果があれば良い


def score_lr(model, data):
    """確率の計算"""
    return [np.max(lr.predict_proba(data), axis=1), model.predict(data)]  # predict_probaで各データがそのクラスに属する確率を取得、predictでそのクラスを表示


# テストデータと訓練データの特徴量を取得
test_feture = pd.read_table("test.feture.txt")
train_feture = pd.read_table("train.feture.txt")

# ロジスティック回帰のモデルをリロード
lr = pickle.load(open("logistic.sav", 'rb'))

train_predict = score_lr(lr, train_feture)
test_predict = score_lr(lr, test_feture)

# print(train_predict)
# print(test_predict)
