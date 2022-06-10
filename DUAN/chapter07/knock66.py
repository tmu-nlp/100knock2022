import numpy as np
from scipy.stats import spearmanr
from gensim.models import KeyedVectors

m = KeyedVectors.load_word2vec_format('./100knock2022/DUAN/chapter07/GoogleNews-vectors-negative300.bin', binary=True)
w = []

with open('./100knock2022/DUAN/chapter07/combined.csv') as f:
    next(f)
    for line in f:
        line = [s.strip() for s in line.split(',')]
        line.append(m.similarity(line[0], line[1]))
        w.append(line)

human = np.array(w).T[2]
w2v = np.array(w).T[3]
cor, p = spearmanr(human, w2v)
print(f'スピアマン相関係数: {cor:.3f}')