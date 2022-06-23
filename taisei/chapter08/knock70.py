from gensim.models import KeyedVectors
import pickle
import pandas as pd
import numpy as np
import joblib


path = "../chapter06/output/"
with open("../chapter07/output/knock60_word2vec", "rb") as f:
    model = pickle.load(f)

def get_text_vec(line):
    """入力文の特徴量ベクトルを取得"""
    words = line.strip().split()
    sum_vecs = np.zeros(300)  # modelは300次元の単語ベクトルを持つので
    for word in words:
        if word in model.index_to_key:
            sum_vecs += model[word]
    return sum_vecs / len(words)


if __name__ == "__main__":

    train_data = pd.read_csv(f'{path}train.txt', sep='\t', names=("CATEGORY", "TEXT"))
    valid_data = pd.read_csv(f'{path}valid.txt', sep='\t', names=("CATEGORY", "TEXT"))
    test_data = pd.read_csv(f'{path}test.txt', sep='\t', names=("CATEGORY", "TEXT"))

    Y_train = train_data["CATEGORY"]
    X_train_text = train_data["TEXT"]
    Y_valid = valid_data["CATEGORY"]
    X_valid_text = valid_data["TEXT"]
    Y_test = test_data["CATEGORY"]
    X_test_text = test_data["TEXT"]

    #カテゴリ名を数字に置換
    Y_train = Y_train.map({"b":0, "t":1, "e":2, "m":3})
    Y_valid = Y_valid.map({"b":0, "t":1, "e":2, "m":3})
    Y_test = Y_test.map({"b":0, "t":1, "e":2, "m":3})

    #特徴量ベクトル
    X_train = X_train_text.map(get_text_vec)
    X_valid = X_valid_text.map(get_text_vec)
    X_test = X_test_text.map(get_text_vec)

    #ファイル書き込み(pandasなどの情報ごとそのまま保存してる)
    joblib.dump(X_train, "./output/X_train.joblib")
    joblib.dump(X_valid, "./output/X_valid.joblib")
    joblib.dump(X_test, "./output/X_test.joblib")
    joblib.dump(Y_train, "./output/Y_train.joblib")
    joblib.dump(Y_valid, "./output/Y_valid.joblib")
    joblib.dump(Y_test, "./output/Y_test.joblib")