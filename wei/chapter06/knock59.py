'''
59.ハイパーパラの探索
学習アルゴリズムや学習パラメータを変えながら、カテゴリ分類モデルを学習
検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めて、
それを使って評価データ上の正解率を求める

DOC:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
class sklearn.model_selection.GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)
realize exhaustive search over specified parameter values for an estimator. estimator's params are optimized by cross-validated grid-search over a param grid.
param_grid:dict or list of dicts. like:{parameter name: list of param settings,...}
'''


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from prettyprinter import cpprint

def acc(lr, x, y):
    y_pred = lr.predict(x)
    return (y == y_pred).mean()

# load features
X_train = pd.read_table('./X_train_features.txt')
X_valid = pd.read_table('./X_valid_features.txt')
X_test = pd.read_table('./X_test_features.txt')
train = pd.read_table('./train.txt', names=['CATEGORY', 'TITLE'])
valid = pd.read_table('./valid.txt', names=['CATEGORY', 'TITLE'])
test = pd.read_table('./test.txt', names=['CATEGORY', 'TITLE'])

params = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
clf = GridSearchCV(LogisticRegression(max_iter=1000), params)
clf.fit(X_train, train['CATEGORY'])

cpprint(clf.cv_results_)
print(f'best estimator: {clf.best_estimator_}')
print(f'best_score: {clf.best_score_}')
print(f'best_params: {clf.best_params_}')

acc = acc(clf, X_test, test['CATEGORY'])
print(f'the acc: {acc}')

'''
{
    'mean_fit_time': array([ 7.86890988,  8.59406538,  0.5847578 , 17.76938844, 15.30056338]),
    'std_fit_time': array([0.38260192, 0.73061453, 0.07462585, 1.16549918, 0.82270624]),
    'mean_score_time': array([0.0315249 , 0.03837147, 0.03008537, 0.02847042, 0.03504901]),
    'std_score_time': array([0.00257325, 0.00376446, 0.00394831, 0.00152249, 0.01041497]),
    'param_solver': masked_array(data=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
             mask=[False, False, False, False, False],
       fill_value='?',
            dtype=object),
    'params': [
        {'solver': 'newton-cg'},
        {'solver': 'lbfgs'},
        {'solver': 'liblinear'},
        {'solver': 'sag'},
        {'solver': 'saga'}
    ],
    'split0_test_score': array([0.87728337, 0.87728337, 0.87025761, 0.87728337, 0.87728337]),
    'split1_test_score': array([0.8763466 , 0.8763466 , 0.86978923, 0.8763466 , 0.8763466 ]),
    'split2_test_score': array([0.87300843, 0.87300843, 0.86597938, 0.87300843, 0.87300843]),
    'split3_test_score': array([0.86597938, 0.86597938, 0.8552015 , 0.86597938, 0.86597938]),
    'split4_test_score': array([0.87956888, 0.87956888, 0.8683224 , 0.87956888, 0.87956888]),
    'mean_test_score': array([0.87443734, 0.87443734, 0.86591002, 0.87443734, 0.87443734]),
    'std_test_score': array([0.00472561, 0.00472561, 0.00555782, 0.00472561, 0.00472561]),
    'rank_test_score': array([1, 1, 5, 1, 1])
}
best estimator: LogisticRegression(max_iter=1000, solver='newton-cg')
best_score: 0.8744373355223448
best_params: {'solver': 'newton-cg'}
the acc: 0.8680659670164917'''

