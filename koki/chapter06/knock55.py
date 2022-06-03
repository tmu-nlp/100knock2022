from sklearn.metrics import confusion_matrix

# 対角成分の数だけ正解

print(f'Confusion matrix (train)\n: {confusion_matrix(y_train, y_train_pred)}')
print(f'Confusion matrix (train)\n : {confusion_matrix(y_test, y_test_pred)}')
