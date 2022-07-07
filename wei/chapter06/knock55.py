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
    X_train = pd.read_table('./X_train_features.txt')
    X_test = pd.read_table('./X_test_features.txt')
    train_re = pd.read_table('./train_re.txt', names=['CATEGORY', 'TITLE'])
    test_re = pd.read_table('./test_re.txt', names=['CATEGORY', 'TITLE'])
    LogRe = LogRe(X_train, train_re['CATEGORY'], )

    train_pred = cal_score(LogRe, X_train)
    test_pred = cal_score(LogRe, X_test)



    print(confusion_matrix(train_re['CATEGORY'], train_pred[1]))
    print(confusion_matrix(test_re['CATEGORY'], test_pred[1]))


'''
confusion matrix of training:
[[4329  108    5   60]
 [  62 4149    2   10]
 [  82  128  506   12]
 [ 203  135    8  873]]
 confusion matrix of test:
[[519  28   5  11]
 [ 19 507   0   2]
 [ 19  26  45   1]
 [ 35  28   2  87]]

'''
