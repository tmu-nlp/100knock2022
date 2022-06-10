"""
> 国名に関する単語ベクトルのベクトル空間をt-SNEで可視化せよ．

# t-SNE
t-SNEは高次元データを2次元または3次元に変換して可視化するための次元削減アルゴリズムである．
"""

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from knock67 import countries, countries_vec
import numpy as np

tsne = TSNE()
tsne.fit(countries_vec)

plt.figure(figsize=(15, 15), dpi=300)
plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1])
for (x, y), name in zip(tsne.embedding_, countries):
    plt.annotate(name, (x, y))
plt.savefig("tsne.png")
plt.show()
