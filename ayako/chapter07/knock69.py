# knock69
# ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ
import knock67
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

if __name__ == "__main__":
    #t-SNEで単語ベクトルを二次元に圧縮
    #ref:https://qiita.com/g-k/items/120f1cf85ff2ceae4aba

    tsne = TSNE(random_state=0)#インスタンス作成
    vec_embedded = tsne.fit_transform(knock67.vec_countries)#ベクトルを埋め込み
    vec_embedded_t = list(zip(*vec_embedded)) #転置:https://jackee777.hatenablog.com/entry/2019/05/03/223646

    plt.figure(figsize=(10,10))
    plt.scatter(*vec_embedded_t)
    for i, name in enumerate(knock67.country_names):
        plt.annotate(name, (vec_embedded[i][0], vec_embedded[i][1]))#annotateでデータに注釈(name)を付与
    plt.savefig("output/output69.png")
