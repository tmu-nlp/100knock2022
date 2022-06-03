'''
56. 適合率precision，再現率recall，F1スコアの計測
52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．カテゴリごとに適合率，再現率，F1スコアを求め，
カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ

document:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
'''

from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
from knock53 import *



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

    train_metrics = precision_recall_fscore_support(train_re['CATEGORY'], train_pred[1],
                                          average=None, labels=['b', 'e', 't', 'm'])[:3]
    test_metrics = precision_recall_fscore_support(test_re['CATEGORY'], test_pred[1],
                                          average=None, labels=['b', 'e', 't', 'm'])[:3]



    train_micro = precision_recall_fscore_support(train_re['CATEGORY'], train_pred[1], average='micro', labels=['b', 'e', 't', 'm'])[:3]
    test_micro = precision_recall_fscore_support(test_re['CATEGORY'], test_pred[1], average='micro',
                                                  labels=['b', 'e', 't', 'm'])[:3]

    print(train_metrics)
    #print(train_res)





