'''
53. 予測
52で学習したロジスティック回帰モデルを用い，
与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．

methods used:
predict(X): predict class labels for samples in X
predict_proba(X): probability estimates
'''


import pandas as pd
from knock51 import get_features
from knock52 import LogRe
import numpy as np

def cal_score(lgr, X):
    # [max prob per sample, pred_label]
    return [np.max(lgr.predict_proba(X), axis=1), lgr.predict(X)]


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

    print(train_pred)




