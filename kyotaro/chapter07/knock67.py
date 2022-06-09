"""
国名に関する単語ベクトルを抽出し, k-meansクラスタリングをクラスタ数k=5として実行せよ.
"""

from itertools import count
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pickle
import re

def preprocess(x):
    pattern = r'[^a-zA-Z]'

model = pickle.load(open("model.sav", "rb"))

countries = []  # 国が入るリスト
countries_vec = []  # 国のベクトルが入るリスト
existed_countries = []  # 68用

# 国の抽出
with open("country.csv", "r") as data:
    for line in data:
        line = line.strip().split(",")
        if line[0] != "ID":
            countries.append(line[1])

# ベクトルが必要なので、抽出した国名のベクトルを生成（モデルに含まれるものだけ）
for country in countries:
    if country in model:
        countries_vec.append(model[country])
        existed_countries.append(country)

# k-meansの定義
kmeans = KMeans(n_clusters=5)

# k-meansの実行
result = kmeans.fit(countries_vec)

# 出力
# for i in range(5):
#     claster = np.where(result.labels_ == i)[0]  # 各クラスタに存在している国を抽出
#     print(f'claster = {i}')
#     print(", ".join([countries[k] for k in claster]))