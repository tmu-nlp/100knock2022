from sklearn.manifold import TSNE  # t-SNE 高次元データの可視化に適した次元削減アルゴリズム
import numpy as np

tsne = TSNE(n_components=2, random_state=42)
vec_embedded = tsne.fit_transform(vec_countries)  # (167, 2)
vec_embedded = np.array(vec_embedded).T  # 転置(2, 167)

fig, ax = plt.subplots(figsize=(16, 14))
plt.scatter(vec_embedded[0], vec_embedded[1])
for i, country in enumerate(countries_name):
    ax.annotate(country, (vec_embedded[0][i], vec_embedded[1][i]))  # plt.annotate パラメータのアノテーションを行う
plt.show()
