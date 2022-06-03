from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pickle

X_train = pd.read_table("./output/train.feature.txt", header=0)
y_train = pd.read_table("./output/train.txt", header=None)[0] #カテゴリラベルのみを格納

# -注意- sklearnのfitの第一引数（特徴量）に文字列は使えないのでベクトル化しなきゃいけない（knock51でやった）
lr = LogisticRegression(random_state=0, max_iter = 10000)
lr.fit(X_train, y_train)
with open("./output/knock52_lr_model", "wb") as f_out:
    pickle.dump(lr, f_out)
