# knock66
# The WordSimilarity-353 Test Collectionの評価データをダウンロードし，
# 単語ベクトルにより計算される類似度のランキングと，人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．

"""
スピアマン相関係数:ノンパラメトリックな(特定の分布に従わない)相関係数，それぞれの順位で評価する．
                一緒に変化しているかどうかはわかるがどれくらい一緒かはわからない．
                pandasで計算できる．
"""
import pandas as pd
from gensim.models import keyedvectors
from tqdm import tqdm
#プログレスバーを表示してくれるらしい，もっと前から使えばよかった.
#ref:https://blog.imind.jp/entry/2019/03/06/111152
tqdm.pandas()

def calc_sim(row):
    """行ごとに類似度を計算する"""
    word1 = row["Word 1"]
    word2 = row["Word 2"]
    if word1 in model and word2 in model:
        sim = model.similarity(word1, word2)
    else:
        sim = None
    return sim


if __name__ == "__main__":
    df = pd.read_csv("wordsim353/combined.csv")
    model = keyedvectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
    
    #類似度の計算結果を最後の列に追加，横方向に連結するときaxis=1
    #progress_applyで進捗をプログレスバーで見れる
    df["Similarity"] = df.progress_apply(calc_sim, axis=1)

    #corrでスピアマン係数を計算
    #二重括弧だとDataframeとして抽出できる
    #一重だとSeries型になってしまうらしい
    spearman = df[["Human (mean)", "Similarity"]].corr(method="spearman")
    print(spearman)

"""
              Human (mean)  Similarity
Human (mean)      1.000000    0.700017
Similarity        0.700017    1.000000
"""