# knock67
# 国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．
from gensim.models import keyedvectors
from sklearn.cluster import KMeans
import numpy as np

model = keyedvectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
countries = []#国名のリスト
country_names = []#モデるに含まれる国名リスト，knock68用
vec_countries = []#モデルに含まれる国名のベクトル
with open("country.csv") as f:
    for line in f:
        country = line.strip().split(",")[1]#２列目のSHORTNAMEを使用
        countries.append(country)
countries.pop(0)
#モデルに含まれる国名のベクトルを取得
for country in countries:
    if country in model:
        country_names.append(country)
        vec_countries.append(model[country])

#k-meansクラスタリング
k = 5
Kmeans = KMeans(n_clusters=k, random_state=0)#インスタンス生成
Kmeans.fit(vec_countries)#クラスタリングの計算を実行

if __name__ == "__main__":
    for i in range(k):
        cluster = np.where(Kmeans.labels_==i)[0]#配列の要素に対して条件に応じた処理を行う，要素1つのタプルが返るのに注意
        print(f"----cluster{i}----")
        print(", ".join([countries[j] for j in cluster]))