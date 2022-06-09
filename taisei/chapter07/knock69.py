import matplotlib.pyplot as plt
from knock67 import in_countrys, in_countrys_vec
from sklearn.manifold import TSNE
# 参考サイト
# https://qiita.com/g-k/items/120f1cf85ff2ceae4aba

if __name__ == "__main__":
    tsne = TSNE(random_state=0)
    embedded = tsne.fit_transform(in_countrys_vec)
    plt.figure(figsize=(9, 9))
    plt.scatter(embedded[:, 0], embedded[:, 1])
    for i, country in enumerate(in_countrys):
        plt.annotate(country, (embedded[i, 0], embedded[i, 1]))
    plt.savefig("./output/knock69_output.png")
