'''
52で学習したロジスティック回帰モデルの正解率を，学習データおよび評価データ上で計測せよ．
'''
from knock51 import X_train, X_test, train, test
from knock52 import lr

print('訓練データ正解率：', lr.score(X_train, train['CATEGORY']))
print('評価データ正解率：', lr.score(X_test, test['CATEGORY']))