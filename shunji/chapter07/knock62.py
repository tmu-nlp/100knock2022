"""
> “United States”とコサイン類似度が高い10語と，その類似度を出力せよ．
"""

import pickle

model = pickle.load(open("model.pkl", "rb"))

print(
    model.most_similar("United_States", topn=10)
)  # https://radimrehurek.com/gensim/models/keyedvectors.html?highlight=keyedvectors#gensim.models.keyedvectors.KeyedVectors.most_similar
