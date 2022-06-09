"""
ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ.
"""

from sklearn.manifold import TSNE
from knock67 import countries_vec, countries
import matplotlib.pyplot as plt
import numpy as np

tsne = TSNE(random_state=0, perplexity=30, n_iter=1000)
embed = tsne.fit_transform(countries_vec)

fig = plt.figure(figsize=(16, 9))
plt.scatter(np.array(embed).T[0], np.array(embed).T[1])
for (x, y), name in zip(embed, countries):
    plt.annotate(name, (x, y))
plt.show()
fig.savefig("69.png")
