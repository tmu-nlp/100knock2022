"""
> The WordSimilarity-353 Test Collectionの評価データをダウンロードし，
単語ベクトルにより計算される類似度のランキングと，人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．

# スピアマン相関係数(スピアマンの順位相関係数)
2変数間に、どの程度、順位づけの直線関係があるかを調べる際に使う分析手段がスピアマンの順位相関です。
データが順位尺度のとき（順位しか付けられないとき）に使用すべき手法です。 
https://qiita.com/dacciinfo/items/88debe69f9f4e927aafc

1. 今回のようなケースでは使うべきではなく，純粋に相関求めた方がいい？
2. scipyのspearmanrの引数にはパラメータそのまま入れる？順位入れる？ -> 内部で順位付けするrankdataが使われていた
    - https://github.com/scipy/scipy/blob/v1.3.0/scipy/stats/stats.py#L3613-L3764
"""

from scipy.stats import spearmanr, pearsonr
import pickle
import pandas as pd

# 保存したモデルをロード
model = pickle.load(open("model.pkl", "rb"))

# 評価データの読み込み
df = pd.read_csv("./wordsim353/combined.csv", header=0)

# w2v_simカラムの作成
df["w2v_sim"] = 0
for i in range(len(df)):
    df.iloc[i, 3] = model.similarity(w1=df.iloc[i, 0], w2=df.iloc[i, 1])

# Human(mean)とw2v_simカラムを順位データに置き換えたdf_rankを作る
df_rank = df.rank(numeric_only=True, ascending=False)

# 確認用
# print(df)
# print(df_rank)

# 相関係数とp値を計算
spe_cor_rank, spe_pval_rank = spearmanr(
    df_rank["Human (mean)"].values, df_rank["w2v_sim"].values
)
spe_cor, spe_pval = spearmanr(df["Human (mean)"].values, df["w2v_sim"].values)
pear_cor, pear_pval = pearsonr(df["Human (mean)"].values, df["w2v_sim"].values)

print("スピアマン相関係数(順位引数):", spe_cor_rank)
print("スピア万相関係数(パラメータ引数):", spe_cor)
print("ピアソン相関係数:", pear_cor)


"""
スピアマン相関係数(順位引数): 0.7000166486272194
スピア万相関係数(パラメータ引数): 0.7000166486272194
ピアソン相関係数: 0.652534960755016
"""
