import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from knock67 import in_countrys, in_countrys_vec
# 参考サイト
# https://qiita.com/sho-matsuo/items/cd7f2b66e572bf60048b

if __name__ == "__main__":
    Z = linkage(in_countrys_vec, method="ward", metric="euclidean")
    plt.figure(figsize=(16, 9))
    dendrogram(Z, labels=in_countrys)
    plt.savefig("./output/knock68_output.png")
