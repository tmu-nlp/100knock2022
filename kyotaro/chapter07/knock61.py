"""
“United States”と”U.S.”のコサイン類似度を計算せよ.
"""
from gensim.models import KeyedVectors

# gensimでファイルを読み込む
model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin.gz', binary=True)

# コサイン類似度
print(model.similarity("United_States", "U.S."))
