'''
57. 特徴量の重みの確認
52で学習したロジスティック回帰モデルの中で、
重みの高い特徴量トップ10と、重みの低い特徴量トップ10を確認せよ。
'''

import pandas as pd
import numpy as np
from IPython import display
from knock53 import *



if __name__ == '__main__':
    X_train = pd.read_table('./X_train_features.txt')
    X_test = pd.read_table('./X_test_features.txt')
    train_re = pd.read_table('./train_re.txt', names=['CATEGORY', 'TITLE'])
    LogRe = LogRe(X_train, train_re['CATEGORY'])

    features = X_train.columns.values
    # print(features.shape)   # (2814,)
    # print(features[:10])
    index = [i for i in range(1,11)]
    # (n_classes, n_features)
    for cls, coef in zip(LogRe.classes_, LogRe.coef_):
        print(f'【カテゴリ】{cls}')
        best10 = pd.DataFrame(features[np.argsort(-coef)[:10]],
                              columns=['重要度上位'], index=index).T
        worst10 = pd.DataFrame(features[np.argsort(coef)[:10]], columns=['重要度下位'], index=index).T
        display.display(pd.concat([best10, worst10], axis=0))
        print('\n')
'''
【カテゴリ】b
        1    2      3      4       5     6          7      8       9       10
重要度上位  fed  ecb  china   bank  stocks  euro  obamacare    oil  yellen  dollar
重要度下位  and  her  study  ebola   video  star   facebook  virus     she     fda


【カテゴリ】e
               1      2       3      4   ...       7         8         9      10
重要度上位  kardashian  chris     her   star  ...      kim       she  jennifer   paul
重要度下位      update     us  google  china  ...  billion  facebook      data  study

[2 rows x 10 columns]


【カテゴリ】m
          1         2       3       4   ...     7    8           9       10
重要度上位  ebola       fda  cancer   study  ...  virus  cdc  cigarettes  health
重要度下位     gm  facebook   apple  amazon  ...  sales   tv        deal     ceo

[2 rows x 10 columns]


【カテゴリ】t
           1         2       3          4   ...   7        8       9        10
重要度上位  google  facebook   apple  microsoft  ...   gm    tesla  mobile  comcast
重要度下位  stocks       fed  cancer       drug  ...  day  percent   money  ukraine

[2 rows x 10 columns]'''