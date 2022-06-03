'''
52. 学習
ロジスティック回帰モデルを学習

document: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
methods used:
fit(X, y): fit model according to the given training data
'''


import pandas as pd
from knock51 import get_features



def LogRe(X_train, train):
    from sklearn.linear_model import LogisticRegression

    LogRe = LogisticRegression(random_state=886, max_iter=1000)
    # モデルを学習
    # X_train: training vector, shape(n_samples, n_features);
    # train['CATEGORY']:target vector relative to X,shape (n_samples)

    LogRe.fit(X_train, train['CATEGORY'])
    return LogRe



if __name__ == '__main__':

    train_re = pd.read_table('./train_re.txt', names=['CATEGORY', 'TITLE'])
    valid_re = pd.read_table('./valid_re.txt', names=['CATEGORY', 'TITLE'])
    test_re = pd.read_table('./test_re.txt', names=['CATEGORY', 'TITLE'])
    X_train = get_features(train_re, valid_re, test_re)[0]
    LogRe = LogRe(X_train, train_re)

    print(LogRe.classes_)
    # ['b' 'e' 'm' 't']
    # .classes_: an attribute of classifier. ndarrarry of shape:A list of class labels known to the classifier.