from matplotlib import pyplot as plt
from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import dendrogram, linkage

# 単語ベクトルを読み込む
m = KeyedVectors.load_word2vec_format('./100knock2022/DUAN/chapter07/GoogleNews-vectors-negative300.bin', binary=True)

# 国名を取得する
countries = set()
with open('./100knock2022/DUAN/chapter07/output.txt') as f:
    for line in f:
        line = line.split()
        if line[0] in ['capital-common-countries', 'capital-world']:
            countries.add(line[2])
        elif line[0] in ['currency', 'gram6-nationality-adjective']:
            countries.add(line[1])
countries = list(countries)

# 単語ベクトルを取得する
countries_vec = [m[country] for country in countries]

plt.figure(figsize=(15, 5)) # サーズを指定する
Z = linkage(countries_vec, method='ward') # Ward法でのクラスタリング
dendrogram(Z, labels=countries) # デンドログラムを表示する
plt.show() # 可視化
