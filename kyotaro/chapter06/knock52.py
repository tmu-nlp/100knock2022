from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

# ロジスティック回帰の設定
lr = LogisticRegression(random_state=1013, max_iter=1000)

# 訓練データと訓練データの特徴量を表したテーブルを取得
X_train = pd.read_table("train.feture.txt")
train = pd.read_table("train.txt")

# ロジスティック回帰を実行
lr.fit(X_train, train["CATEGORY"])

# 結果を保存
file_name = "logistic.sav"
pickle.dump(lr, open(file_name, "wb"))
