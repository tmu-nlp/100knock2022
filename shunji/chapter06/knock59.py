"""
学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．
検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．
また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．
"""

from cmath import log
from knock51 import X_train, X_valid, train, valid
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import optuna


# 最適化関数
def objective(trial):

    # ハイパーパラメータのセット
    C = trial.suggest_float("C", 1e-4, 1e4, log=True)

    # モデルの学習
    svc = SVC(random_state=1, max_iter=10000, C=C, gamma="auto")
    svc.fit(X_train, train["CATEGORY"])

    # 正解率の算出
    valid_accuracy = svc.score(X_valid, valid["CATEGORY"])

    return valid_accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=4)

print("最適値：", study.best_value)
print("最適パラメータ", study.best_params)
