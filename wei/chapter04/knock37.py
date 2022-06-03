'''
37. 「猫」と共起頻度の高い上位10語
「猫」とよく共起する(共起頻度が高い)10語とその出現頻度をグラフで表示せよ。
'''

from knock30 import load_result
from collections import defaultdict
import matplotlib.pyplot as plt
import japanize_matplotlib

def cooccured_tops(morphs_result):
    dic = defaultdict(int)
    for morphs in morphs_result:
        for i in range(1, len(morphs)-1):
            if morphs[i]['surface'] == '猫':
                dic[morphs[i-1]['surface']] += 1
                dic[morphs[i+1]['surface']] += 1
    sort_list = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    return sort_list   # list of tuple



if __name__ == "__main__":
    path = '../data/neko.txt.mecab'
    results = load_result(path)
    words = []
    counts = []
    for w,c in cooccured_tops(results)[:9]:
        words.append(w)
        counts.append(c)

    plt.bar(words,counts)
    plt.title('「猫」と共起頻度上位10語')
    plt.xlabel('語')
    plt.ylabel('共起頻度')
    plt.show()




