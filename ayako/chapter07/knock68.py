# knock68
# 国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
# さらに，クラスタリング結果をデンドログラムとして可視化せよ
import knock67
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

plt.figure(figsize=(15, 5))
Z = linkage(knock67.vec_countries, method="ward")#linkageモジュールでクラスタリングを行う
dendrogram(Z, labels=list(knock67.country_names))#デンドログラムで可視化
plt.savefig("output/output68.png")