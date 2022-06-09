"""
The WordSimilarity-353 Test Collectionの評価データをダウンロードし, 単語ベクトルにより計算される類似度のランキングと，
人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．
"""

import numpy as np
import pickle
from scipy.stats import spearmanr

model = pickle.load(open("model.sav", "rb"))

human_similarity = []
word_similality = []


with open("combined.csv", "r") as data:
    for line in data:
        line = line.strip().split(",")
        if " " not in line[0]:
            human_similarity.append(line[2])
            word_similality.append(model.similarity(line[0], line[1]))
    
human = np.array(human_similarity).T
word = np.array(word_similality).T
correlation, pvalue = spearmanr(human, word)
print(f'スピアマン相関係数：{correlation:.3f}')