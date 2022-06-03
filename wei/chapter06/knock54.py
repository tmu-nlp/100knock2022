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
from knock51 import get_features
from knock52 import LogRe
from knock53 import cal_score


if __name__ == '__main__':

    train_re = pd.read_table('./train_re.txt', names=['CATEGORY', 'TITLE'])
    valid_re = pd.read_table('./valid_re.txt', names=['CATEGORY', 'TITLE'])
    test_re = pd.read_table('./test_re.txt', names=['CATEGORY', 'TITLE'])
    X_train = get_features(train_re, valid_re, test_re)[0]
    X_test = get_features(test_re, valid_re, test_re)[2]

    LogRe_train = LogRe(X_train, train_re)
    LogRe_test = LogRe(X_test, test_re)

    train_pred = cal_score(LogRe_train, X_train)
    test_pred = cal_score(LogRe_test, X_test)

    print(accuracy_score(train_re['CATEGORY'], train_pred[1]))
    print(accuracy_score(test_re['CATEGORY'], test_pred[1]))


