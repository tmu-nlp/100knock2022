'''
54. 正解率の計測
52で学習したロジスティック回帰モデルの正解率を，
学習データおよび評価データ上で計測せよ．

method used:
metrics.accuracy_score(y_true, y_pred): return accuracy score.
labels predicted must exactly match the corresponding gt
'''


import pandas as pd
from sklearn.metrics import accuracy_score
from knock53 import *


if __name__ == '__main__':
    X_train = pd.read_table('./X_train_features.txt')
    X_test = pd.read_table('./X_test_features.txt')
    train_re = pd.read_table('./train_re.txt', names=['CATEGORY', 'TITLE'])
    test_re = pd.read_table('./test_re.txt', names=['CATEGORY', 'TITLE'])
    LogRe = LogRe(X_train, train_re['CATEGORY'],)

    train_pred = cal_score(LogRe, X_train)
    test_pred = cal_score(LogRe, X_test)

    print(accuracy_score(train_re['CATEGORY'], train_pred[1]))
    print(accuracy_score(test_re['CATEGORY'], test_pred[1]))


'''
train:0.9236319340329835 
test: 0.8680659670164917'''