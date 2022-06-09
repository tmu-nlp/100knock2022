'''
66. wordsimilarity -353での評価
単語ベクトルにより計算される類似度のランキングと、
人間の類似度判定のランキングの間のスピアマン相関係数を計算
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
'''
from knock60 import *
import numpy as np
from scipy.stats import spearmanr


if __name__ == '__main__':
    wv = load_wv()
    sim = []
    with open('../data/wordsim353/combined.csv', 'r', encoding='utf-8') as f:
        next(f)  # 最初の行を読み込まない
        data = f.readlines()
    for line in data:
        words = line.split(',')
        wv_sim = wv.similarity(words[0], words[1])    # return float
        words.append(wv_sim)
        sim.append(words)



    human_score = np.array(sim).T[2]
    w2v_score = np.array(sim).T[3]
    correlation, pvalue = spearmanr(human_score, w2v_score)  # float, float

    print(f'スピアマン相関係数は{correlation:.4f}, pvalue は{pvalue:.4f}')


    # スピアマン相関係数は0.6850, pvalue は0.0000


