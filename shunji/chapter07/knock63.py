"""
> “Spain”の単語ベクトルから”Madrid”のベクトルを引き，
”Athens”のベクトルを足したベクトルを計算し，そのベクトルと類似度の高い10語とその類似度を出力せよ．
"""

import pickle

model = pickle.load(open("model.pkl", "rb"))

print(model.most_similar(positive=["Spain", "Athens"], negative=["Madrid"], topn=10))
