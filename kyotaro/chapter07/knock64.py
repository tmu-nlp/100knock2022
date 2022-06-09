"""
単語アナロジーの評価データをダウンロードし, vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
そのベクトルと類似度が最も高い単語と，その類似度を求めよ．求めた単語と類似度は，各事例の末尾に追記せよ
"""
from gensim.models import KeyedVectors
import pickle
from tqdm import tqdm

model = pickle.load(open("model.sav", 'rb'))

with open("questions-words.txt", "r") as data:
    for line in tqdm(data):
        line = line.strip().split()
        if line[0] == ":":
            print(" ".join(line))
        else:
            ans = model.most_similar(
                positive=[line[1], line[2]], negative=[line[0]], topn=1)
            original = ' '.join(line)
            print(f'{original} {ans[0][0]} {ans[0][1]}')
