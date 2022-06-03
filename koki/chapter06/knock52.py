from sklearn.linear_model import LogisticRegression

model = LogisticRegression() #ロジスティック回帰モデルのインスタンス生成
model.fit(x_train, y_train) #ロジスティック回帰モデルの重みを学習
