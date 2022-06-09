"""
“United States”とコサイン類似度が高い10語と, その類似度を出力せよ.
"""

from gensim.models import KeyedVectors
from knock60 import model

# 単語の類似度を高い順に表示
print(model.most_similar('United_States'))