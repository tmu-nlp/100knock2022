# knock55
# 52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，学習データおよび評価データ上で作成せよ．
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    X_train = pd.read_table("train.feature.txt", header=None)
    Y_train = pd.read_table("train.txt", header=None)[1]
    X_test = pd.read_table("test.feature.txt", header=None)
    Y_test = pd.read_table("test.txt", header=None)[1]

    lr = pickle.load(open("model.pkl", "rb"))

    #混同行列を作成
    #b = business, t = science and technology, e = entertainment, m = health
    labels = ["b","t","e","m"]#ラベルの順序指定
    train_cm = confusion_matrix(Y_train, lr.predict(X_train), labels=labels)
    test_cm = confusion_matrix(Y_test, lr.predict(X_test), labels=labels)

    #見やすく出力
    colums_labels = ["pred_" + str(l) for l in labels]
    index_labels = ["act_" + str(l) for l in labels]
    train_cm = pd.DataFrame(train_cm, columns=colums_labels, index=index_labels)
    test_cm = pd.DataFrame(test_cm, columns=colums_labels, index=index_labels)
    print("訓練データの混同行列")
    print(train_cm)
    print("評価データの混同行列")
    print(test_cm)
"""
訓練データの混同行列
       pred_b  pred_t  pred_e  pred_m
act_b    4420      34      55       2
act_t     134     995     117       2
act_e      15       2    4184       0
act_m      85       3     128     508

評価データの混同行列
       pred_b  pred_t  pred_e  pred_m
act_b     539       9      25       2
act_t      29      93      16       0
act_e       7       1     532       1
act_m      16       2      17      47
"""