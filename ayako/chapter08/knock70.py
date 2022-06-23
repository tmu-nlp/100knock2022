import pandas as pd
from gensim.models import KeyedVectors
import numpy as np

#knock50で作成したデータを読み込む
train = pd.read_csv("../chapter06/train.txt", sep="\t", header=None)
valid = pd.read_csv("../chapter06/valid.txt", sep="\t", header=None)
test = pd.read_csv("../chapter06/test.txt", sep="\t", header=None)

#knock60のデータから単語ベクトルを取得
model = KeyedVectors.load_word2vec_format("../chapter07/GoogleNews-vectors-negative300.bin.gz", binary=True)

#ラベルを数字に置換する用の辞書{元の値: 置換後の値}
labels = {"b":0, "t":1, "e":2, "m":3}
#ラベルを置換
y_train = train.iloc[:,1].replace(labels)
y_valid = valid.iloc[:,1].replace(labels)
y_test = test.iloc[:,1].replace(labels)
#それぞれのデータのラベルベクトルを作成して保存
y_train.to_csv("./data/y_train.txt", header=False, index=False)
y_valid.to_csv("./data/y_valid.txt", header=False, index=False)
y_test.to_csv("./data/y_test.txt", header=False, index=False)

#特徴量行列を作成
def write_x(f_name, df):
    with open(f_name, "w") as f:
        for title in df.iloc[:,0]:#見出しを一個ずつ見ていく
            vectors = []
            for word in title.split():
                if word in model.index_to_key:
                    vectors.append(model[word])
            if len(vectors) == 0:
                vector = np.zeros(300)#ベクトルが空の時は300次元のベクトル入れとく
            else:
                vectors = np.array(vectors)#300次元×単語数の配列
                """
                タイトルの単語数Nのとき
                [[単語1の300次元ベクトル1, ..., 単語1の300次元ベクトル300]
                 [単語2の300次元ベクトル1, ..., 単語2の300次元ベクトル300]
                  ...
                 [単語Nの300次元ベクトル1, ..., 単語Nの300次元ベクトル300]]
                ->列ごとに平均を求めて300次元のベクトル1つでタイトルの特徴を表現
                """
                vector = vectors.mean(axis=0)#列ごとに平均した値
            vector = vector.astype("str").tolist()#joinで出力するために文字列にしてからリスト変換
            output = " ".join(vector)
            print(output, file=f)

write_x("./data/x_train.txt", train)
write_x("./data/x_valid.txt", valid)
write_x("./data/x_test.txt", test)
