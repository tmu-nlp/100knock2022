"""
> Google Newsデータセット（約1,000億単語）での学習済み単語ベクトル（300万単語・フレーズ，300次元）をダウンロードし，
”United States”の単語ベクトルを表示せよ．ただし，”United States”は内部的には”United_States”と表現されていることに注意せよ．
"""

from gensim.models import KeyedVectors
import pickle

"""
save_word2vec_formatで保存したモデルはload_word2vec_formatで読み込めて、
saveで保存したモデルはloadで読み込めるらしい https://qiita.com/omuram/items/6570973c090c6f0cb060
"""

model = KeyedVectors.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin.gz", binary=True
)
filename = "model.pkl"
pickle.dump(model, open(filename, "wb"))

if __name__ == "__main__":
    print(model["United_States"])
