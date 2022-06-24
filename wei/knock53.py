'''
53. 予測
52で学習したロジスティック回帰モデルを用い，
与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．

methods used:
predict(X): predict class labels for samples in X
predict_proba(X): probability estimates
'''

import numpy as np
import pandas as pd
from knock52 import *


def cal_score(lgr, X):
    # [max prob per sample, pred_label]
    return [np.max(lgr.predict_proba(X), axis=1), lgr.predict(X)]


if __name__ == '__main__':
    X_train = pd.read_table('./X_train_features.txt')
    X_test = pd.read_table('./X_test_features.txt')
    train_re = pd.read_table('./train_re.txt', names=['CATEGORY', 'TITLE'])
    LogRe = LogRe(X_train, train_re['CATEGORY'])

    train_pred = cal_score(LogRe, X_train)
    test_pred = cal_score(LogRe, X_test)

    print(f'pred proba on train:{train_pred}')
    print(f'pred proba on test:{test_pred}')

'''
pred proba on train:[array([0.68265556, 0.7315841 , 0.8333757 , ..., 0.77320911, 0.88480182,
       0.90610809]), array(['e', 'b', 'e', ..., 'e', 'e', 'b'], dtype=object)]
pred proba on test:[array([0.6061314 , 0.50418264, 0.72998405, ..., 0.60211282, 0.88243351,
       0.9778531 ]), array(['e', 'e', 'b', ..., 'e', 'e', 'b'], dtype=object)]'''


