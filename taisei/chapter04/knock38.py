import knock30
import japanize_matplotlib #pltで日本語を使うため
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

count_dict = defaultdict(lambda: 0)
for line in knock30.morpheme_line_list:
    for morpheme in line:
        if morpheme["品詞"] == "記号":
            continue
        count_dict[morpheme["基本形"]] += 1

count_dict_sort = sorted(count_dict.items(), key=lambda x:x[1], reverse=True)
#ここより上はknock35と同じ

count_np_sort = np.array(count_dict_sort)
frec_list = count_np_sort[:, 1] #各単語の出現頻度だけ抽出

plt.hist(frec_list.astype(np.int64), range=(1,max(frec_list.astype(np.int64)))) #int化
"""
numpy使わなくても下のでも抽出できる
hoge, frec_list = zip(*count_dict_sort)
plt.hist(frec_list, range=(1,max(frec_list)))
"""
plt.xlabel("出現頻度")
plt.ylabel("単語の異なり数")
plt.title("単語の出現頻度分布")
plt.savefig("./output/knock38_output")