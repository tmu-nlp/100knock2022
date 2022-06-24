'''
56. 適合率precision，再現率recall，F1スコアの計測
52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．カテゴリごとに適合率，再現率，F1スコアを求め，
カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ

document:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
'''

from sklearn.metrics import classification_report
import pandas as pd
from knock53 import *



if __name__ == '__main__':
    X_train = pd.read_table('./X_train_features.txt')
    X_test = pd.read_table('./X_test_features.txt')
    train_re = pd.read_table('./train_re.txt', names=['CATEGORY', 'TITLE'])
    test_re = pd.read_table('./test_re.txt', names=['CATEGORY', 'TITLE'])
    LogRe = LogRe(X_train, train_re['CATEGORY'], )

    train_pred = cal_score(LogRe, X_train)
    test_pred = cal_score(LogRe, X_test)

    print(classification_report(train_re['CATEGORY'], train_pred[1]))
    '''
                  precision    recall  f1-score   support

           b       0.93      0.96      0.94      4502
           e       0.92      0.98      0.95      4223
           m       0.97      0.70      0.81       728
           t       0.91      0.72      0.80      1219

    accuracy                           0.92     10672
   macro avg       0.93      0.84      0.88     10672
weighted avg       0.92      0.92      0.92     10672'''
    print(classification_report(test_re['CATEGORY'], test_pred[1]))
    '''
              precision    recall  f1-score   support

           b       0.88      0.92      0.90       563
           e       0.86      0.96      0.91       528
           m       0.87      0.49      0.63        91
           t       0.86      0.57      0.69       152

    accuracy                           0.87      1334
   macro avg       0.87      0.74      0.78      1334
weighted avg       0.87      0.87      0.86      1334'''




