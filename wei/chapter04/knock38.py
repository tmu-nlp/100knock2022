'''
38. ヒストグラム
単語の出現頻度のヒストグラムを描け。ただし、横軸は出現頻度を表し、1から単語の出現
頻度の最大値までの線形目盛とする。縦軸はｘ軸で示される出現頻度となった
単語の異なり数(種類数)である
'''

from knock30 import load_result
from collections import defaultdict

import matplotlib.pyplot as plt
import japanize_matplotlib

def frequency(morphs_result):
    dic = defaultdict(int)
    for morphs in morphs_result:
        for morph in morphs:
            if morph['pos'] != '記号':
                dic[morph['surface']] += 1

    return dic

if __name__ == "__main__":
    path = '../data/neko.txt.mecab'
    results = load_result(path)

    counts = frequency(results).values()

    plt.hist(counts, bins=100)
    plt.xlabel('出現頻度')
    plt.ylabel('異なり数')
    plt.title('単語出現頻度ヒストグラム')
    plt.show()