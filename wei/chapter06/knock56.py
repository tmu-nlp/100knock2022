'''
56. 適合率precision，再現率recall，F1スコアの計測
52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．カテゴリごとに適合率，再現率，F1スコアを求め，
カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ

document:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
'''

from sklearn.metrics import classification_report
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

    #print(classification_report(train_re['CATEGORY'], train_pred[1]))
    '''
                  precision    recall  f1-score   support

           b       0.94      0.97      0.96      4502
           e       0.94      0.99      0.96      4223
           m       0.98      0.72      0.83       728
           t       0.94      0.76      0.84      1219

    accuracy                           0.94     10672
   macro avg       0.95      0.86      0.90     10672
weighted avg       0.94      0.94      0.94     10672'''
    print(classification_report(test_re['CATEGORY'], test_pred[1]))
    '''
                  precision    recall  f1-score   support

           b       0.88      0.95      0.91       563
           e       0.85      0.98      0.91       528
           m       1.00      0.44      0.61        91
           t       0.96      0.51      0.66       152

    accuracy                           0.88      1334
   macro avg       0.92      0.72      0.77      1334
weighted avg       0.89      0.88      0.86      1334'''




