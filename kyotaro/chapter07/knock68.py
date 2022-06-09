"""
国名に関する単語ベクトルに対し, Ward法による階層型クラスタリングを実行せよ. さらに, クラスタリング結果をデンドログラムとして可視化せよ. 
"""

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from knock67 import countries_vec, existed_countries

linkage_result = linkage(countries_vec, method="ward", metric="euclidean")

fig = plt.figure(figsize=(16, 9))
dendrogram(linkage_result, labels=existed_countries)
fig.savefig("68.png")
#plt.show()