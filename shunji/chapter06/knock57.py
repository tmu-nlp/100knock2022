"""
52で学習したロジスティック回帰モデルの中で，重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．
"""

from knock51 import X_train
from knock52 import lr
import numpy as np
import pandas as pd


features = X_train.columns.values
print(X_train.columns)

index = [i for i in range(1, 11)]

for c, coef in zip(lr.classes_, lr.coef_):  # lr.classes_:クラスラベル, lr.coef_:重み
    print(f"【カテゴリ】{c}")
    
    # np.argsortで重みが昇順になるようにインデックスのndarrayが返される．
    # 転置して行データ状にする
    best10 = pd.DataFrame(
        features[np.argsort(coef)[::-1][:10]], columns=["重要度上位"], index=index
    ).T
    worst10 = pd.DataFrame(
        features[np.argsort(coef)[:10]], columns=["重要度下位"], index=index
    ).T
    print(pd.concat([best10, worst10], axis=0))  # 上下方向にconcatする
    print("\n")
