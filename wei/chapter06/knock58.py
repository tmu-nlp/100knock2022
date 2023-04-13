'''
58.正則化パラメータの変更
正則化パラメータを調整することで、過学習の度合いを制御できる。
異なる正則化パラメータでロジスティック回帰モデルを学習し、学習データ、検証データ、評価データ上で正解率を求める
結果は、正則化パラメータを横軸、正解率を縦軸としたグラフにまとめる
'''
from sklearn.linear_model import LogisticRegression
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from knock54 import *

if __name__ == '__main__':
    X_train = pd.read_table('./X_train_features.txt')
    X_valid = pd.read_table('./X_valid_features.txt')
    X_test = pd.read_table('./X_test_features.txt')
    train_re = pd.read_table('./train_re.txt', names=['CATEGORY', 'TITLE'])
    valid_re = pd.read_table('./valid_re.txt', names=['CATEGORY', 'TITLE'])
    test_re = pd.read_table('./test_re.txt', names=['CATEGORY', 'TITLE'])

    start = time.time()
    res = []
    regular = list(np.logspace(-6, 3, 10))   # [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for c in tqdm(regular):
        # C: Inverse of regularization strength,and smaller values specify stronger regularization
        lr = LogisticRegression(random_state=886, max_iter=10000,multi_class='auto', C=c)
        lrfit_T= lr.fit(X_train, train_re['CATEGORY'])
        lrfit_V = lr.fit(X_valid, valid_re['CATEGORY'])
        lrfit_Te = lr.fit(X_test, test_re['CATEGORY'])

        train_pred = cal_score(lrfit_T, X_train)
        train_acc = accuracy_score(train_re['CATEGORY'], train_pred[1])

        valid_pred = cal_score(lrfit_V, X_valid)
        valid_acc = accuracy_score(valid_re['CATEGORY'], valid_pred[1])

        test_pred = cal_score(lrfit_Te, X_test)
        test_acc = accuracy_score(test_re['CATEGORY'], test_pred[1])

        res.append([c, train_acc, valid_acc, test_acc])
    print(f'acc on training set : {train_acc:.3f}')
    print(f'acc on valid set : {valid_acc:.3f}')
    print(f'acc on test set : {test_acc:.3f}')
    end = time.time()
    print(f'time used : {end-start}')


    '''
    acc on training set : 0.820
    acc on valid set : 0.829
    acc on test set : 1.000
    time used : 318.0961112976074'''

    result = np.array(res).T
    plt.plot(result[0], result[1], label='train')
    plt.plot(result[0], result[2], label='valid')
    plt.plot(result[0], result[3], label='test')
    plt.ylabel('accuracy')
    plt.xlabel('regularization')
    plt.xscale('log')
    plt.legend()
    plt.savefig('./knock58.png')



