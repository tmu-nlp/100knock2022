"""
> 国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．
"""

import pickle
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

model = pickle.load(open("model.pkl", "rb"))

# 国名の取得
countries = set()
with open("./questions-words-add.txt") as f:
    for line in f:
        line = line.split()
        if line[0] in ["capital-common-countries", "capital-world"]:
            countries.add(line[2])
        elif line[0] in ["currency", "gram6-nationality-adjective"]:
            countries.add(line[1])
countries = list(countries)

# 単語ベクトルの取得
countries_vec = [model[country] for country in countries]
df = pd.DataFrame(countries_vec)

# 確認用
# print(df)

if __name__ == "__main__":
    # k-meansクラスタリング
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df)
    for i in range(5):
        cluster = np.where(kmeans.labels_ == i)[0]
        print("cluster", i)
        print(", ".join([countries[k] for k in cluster]))
