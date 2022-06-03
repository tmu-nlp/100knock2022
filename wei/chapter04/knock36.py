'''
36. 頻度上位10語
出現頻度が高い10語とその出現頻度をグラフ(例えば棒グラフなど)で表示せよ。
'''

from knock30 import load_result
from knock35 import sort_frequency

import matplotlib.pyplot as plt
import japanize_matplotlib


if __name__ == "__main__":
    path = '../data/neko.txt.mecab'
    results = load_result(path)
    words = []
    counts = []
    for w,c in sort_frequency(results)[:9]:
        words.append(w)
        counts.append(c)

    plt.bar(words, counts)
    plt.xlabel('語')
    plt.ylabel('出現頻度')
    plt.title('頻度上位10語')
    plt.show()
