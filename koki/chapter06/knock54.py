from sklearn.metrics import accuracy_score  # 正解率計算用のメソッド

y_train_pred = model.predict(x_train)  #予測
y_test_pred = model.predict(x_test)

print(f'Accuracy (train) : {accuracy_score(y_train, y_train_pred)}')
print(f'Accuracy (test) : {accuracy_score(y_test, y_test_pred)}')
# Accuracyは正解してたらsum+=1とかしてデータ数で割れば出せる　面倒なのでメソッド使った
