"""
> 国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
さらに，クラスタリング結果をデンドログラムとして可視化せよ．

# Ward法
全てのデータが別々のクラスタである状態から始めて，「今あるクラスタの中で最も距離が近い2つのクラスタをまとめていく」
という操作を目標のクラスタ数になるまで続ける手法

# デンドログラム
日本語で言うと樹形図．Ward法によるクラスタ連結を示すことができる

- https://www.albert2005.co.jp/knowledge/data_mining/cluster/hierarchical_clustering
- https://mathwords.net/wardmethod
"""

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from knock67 import df

linkage_result = linkage(df, method="ward", metric="euclidean")
plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor="w", edgecolor="k")
dendrogram(linkage_result, labels=df.index)
plt.show()
