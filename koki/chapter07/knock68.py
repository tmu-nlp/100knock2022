from knock67 import vec_countries
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# 階層型クラスタリングの実行
# linkage...凝縮型クラスタリングのメソッド、今回はウォード法を用いる、その他群平均法、重み付き平均法、メディアン法など
# Z = linkage(X, 'method_name')  Xに座標データを入力、結合手順を示したリストZを返す
Z = linkage(vec_countries, method='ward')
# Z = linkage(vec_countries, method='average')

# デンドログラムの描画
plt.figure(figsize=(20, 10))
dendrogram(Z, labels=countries_name)  # Zに結合手順を示すリスト、labelsに葉のラベルのリスト
plt.savefig('./results/output68.png')
plt.show()
