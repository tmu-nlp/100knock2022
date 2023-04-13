'''
52で学習したロジスティック回帰モデルを用い，与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．
'''
import numpy as np
from knock51 import X_train, X_test, X_valid
from knock52 import lr


def score_lr(lr, X):
    '''第一要素を予測確率のndarray, 第二要素を予測されるラベルのリストとするリストを返す'''
    return [np.max(lr.predict_proba(X), axis=1), lr.predict(X)]


train_pred = score_lr(lr, X_train)
test_pred = score_lr(lr, X_test)
valid_pred = score_lr(lr, X_valid)

if __name__ == "__main__":
    print(train_pred)