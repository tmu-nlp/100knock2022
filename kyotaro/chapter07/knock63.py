"""
“Spain”の単語ベクトルから”Madrid”のベクトルを引き, ”Athens”のベクトルを足したベクトルを計算し, 
そのベクトルと類似度の高い10語とその類似度を出力せよ.
"""

from gensim.models import KeyedVectors
from numpy import negative, positive
from knock60 import model

# ベクトルの生成
vec = model["Spain"] - model["Madrid"] + model["Athens"]

# 類似度の高い順
ans = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'])

# 出力
print(ans)
