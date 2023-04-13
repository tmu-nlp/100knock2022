"""
52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．
カテゴリごとに適合率，再現率，F1スコアを求め，カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ．
"""
from knock51 import test
from knock53 import test_pred
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd


def calculate_scores(y_true, y_pred):
    # 適合率
    precision = precision_score(
        test["CATEGORY"], test_pred[1], average=None, labels=["b", "e", "t", "m"]
    )  # Noneを指定するとクラスごとの精度をndarrayで返す
    precision = np.append(
        precision, precision_score(y_true, y_pred, average="micro")
    )  # 末尾にマイクロ平均を追加
    precision = np.append(
        precision, precision_score(y_true, y_pred, average="macro")
    )  # 末尾にマクロ平均を追加

    # 再現率
    recall = recall_score(
        test["CATEGORY"], test_pred[1], average=None, labels=["b", "e", "t", "m"]
    )
    recall = np.append(recall, recall_score(y_true, y_pred, average="micro"))
    recall = np.append(recall, recall_score(y_true, y_pred, average="macro"))

    # F1スコア
    f1 = f1_score(
        test["CATEGORY"], test_pred[1], average=None, labels=["b", "e", "t", "m"]
    )
    f1 = np.append(f1, f1_score(y_true, y_pred, average="micro"))
    f1 = np.append(f1, f1_score(y_true, y_pred, average="macro"))

    # 結果を結合してデータフレーム化
    scores = pd.DataFrame(
        {"適合率": precision, "再現率": recall, "F1スコア": f1},
        index=["b", "e", "t", "m", "マイクロ平均", "マクロ平均"],
    )

    return scores

print(calculate_scores(test["CATEGORY"], test_pred[1]))
