# knock-36
# 出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

import knock30
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import defaultdict #default-set
from collections import OrderedDict
word_dict = defaultdict(lambda: 0)

for line in knock30.sentence_list:
    for morph in line:
        word_dict[morph["surface"]] += 1

sort_word_dict = OrderedDict(sorted(word_dict.items(), key = lambda x:x[1], reverse=True)[:10])

label = []
data = []

for k,v in sort_word_dict.items():
    label.append(k)
    data.append(v)
x = [1,2,3,4,5,6,7,8,9,10]
# print(label)
plt.bar(x,data)
plt.xticks(x,label)
plt.savefig("36.png")