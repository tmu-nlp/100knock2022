'''
55. 混同行列の作成
52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，
学習データおよび評価データ上で作成せよ

document:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
'''

from sklearn.metrics import confusion_matrix

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


    print(confusion_matrix(train_re['CATEGORY'], train_pred[1]))
    print(confusion_matrix(test_re['CATEGORY'], test_pred[1]))


'''
confusion matrix of training:
[[4389   64    5   44]
 [  36 4180    1    6]
 [  80  114  526    8]
 [ 169  112    6  932]]
 confusion matrix of test:
 [[533  27   0   3]
 [ 10 518   0   0]
 [ 22  29  40   0]
 [ 43  32   0  77]]

'''
