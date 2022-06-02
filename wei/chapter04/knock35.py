'''
35. 単語の出現頻度
文章中に出現する単語とその出現頻度を求め、出現頻度の高い順に並べよ。

defaultdict() to count word frequency
'''

from collections import defaultdict
from knock30 import load_result

def sort_frequency(morphs_result):
    dic = defaultdict(int)
    for morphs in morphs_result:
        for morph in morphs:
            if morph['pos'] != '記号':
                dic[morph['surface']] += 1

    fre_list = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    return fre_list




if __name__ == "__main__":
    path = '../data/neko.txt.mecab'
    results = load_result(path)
    for i in sort_frequency(results)[:9]:
        print(i)