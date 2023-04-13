# knock59
# 学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．
# 検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．
# また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

if __name__ == "__main__":
    #学習データ
    X_train = pd.read_table("train.feature.txt", header=None)
    Y_train = pd.read_table("train.txt", header=None)[1]
    #検証データ
    X_valid = pd.read_table("valid.feature.txt", header=None)
    Y_valid = pd.read_table("valid.txt", header=None)[1]
    #評価データ
    X_test = pd.read_table("test.feature.txt", header=None)
    Y_test = pd.read_table("test.txt", header=None)[1]

    #学習パラメータはsolverを変える
    params = {"solver": ["newton-cg", "lbfgs", "liblinear", "sag"]}

    #グリッドサーチのインスタンス生成
    gscv = GridSearchCV(LogisticRegression(random_state=0, max_iter=1000), params, scoring="accuracy")
    gscv.fit(X_train, Y_train)

    #スコアの一覧を取得
    gs_result = pd.DataFrame.from_dict(gscv.cv_results_)
    gs_result.to_csv("gs_result.csv")
    
    #最も精度が良いモデルの正解率とパラメータを出力
    print(f"最高精度モデルの正解率:{gscv.best_score_}")
    print(f"最高精度モデルのパラメータ:{gscv.best_params_}")

    #最も精度が良いモデルを取得し，評価データの正解率を出力
    best_lr = gscv.best_estimator_
    print(f"評価データの正解率:{accuracy_score(Y_test, best_lr.predict(X_test))}")

"""
最高精度モデルの正解率:0.8713968179691157
最高精度モデルのパラメータ:{'solver': 'newton-cg'}
評価データの正解率:0.907185628742515
"""