'''
52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，学習データおよび評価データ上で作成せよ．
'''

from knock51 import X_train, X_test, train, test
from knock52 import lr
from sklearn.metrics import confusion_matrix

pred_train = lr.predict(X_train)
pred_test = lr.predict(X_test)

cm_train = confusion_matrix(pred_train, train['CATEGORY'])
cm_test = confusion_matrix(pred_test, test['CATEGORY'])

print(cm_train)
print(cm_test)