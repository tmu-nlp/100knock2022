from gensim.models import keyedvectors
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

with open('GoogleNews-vectors.pkl', 'rb') as f:
    model = pickle.load(f)
#model = keyedvectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

# 国名の入手: https://www.fao.org/nocs/en/
# xlsxファイルをダウンロードし、国名の情報のみのcsvに整形
# df = pd.read_csv(./country-names.csv')
# df.head()

countries = []
countries_name = []
vec_countries = []

with open('./country-name.csv', 'r', encoding='utf-8_sig') as f:
    for line in f:
        country = line.rstrip()  # 改行を削除
        countries.append(country)

countries.pop(0)  # ヘッダの削除

# モデルに含まれる国名のみを抽出
for country in countries:
    if country in model:
        countries_name.append(country)
        vec_countries.append(model[country])  # 単語ベクトルの取得
    
# k-meansモデル作成
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)  # n_clusters..クラスタ数(デフォルト8)

# クラスタリング実行
kmeans.fit(vec_countries)

for i in range(k):
    cluster = np.where(kmeans.labels_ == i)[0]
    print(f'cluster:{i}')
    print(", ".join([countries[j] for j in cluster]))
    print('-'*100)
