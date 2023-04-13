'''
51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．
'''

from sklearn.linear_model import LogisticRegression
from knock51 import X_train, train

# モデルの学習
lr = LogisticRegression(max_iter=1000)  # デフォルトのmax_iter=100だとエラーになる
lr.fit(X_train, train["CATEGORY"])  # 今回のテーマがカテゴリ分類だからラベルはtrain['CATEGORY']
