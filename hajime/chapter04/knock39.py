# knock-39
# 単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．

import knock30
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import defaultdict #default-set
from collections import OrderedDict
word_dict = defaultdict(lambda: 0)

for line in knock30.sentence_list:
    for morph in line:
        if morph["pos"] == "記号":
            continue
        word_dict[morph["base"]] += 1

sort_word_dict = OrderedDict(sorted(word_dict.items(), key = lambda x:x[1], reverse=True))

x = np.arange(len(sort_word_dict)) + [1] * len(sort_word_dict)
# print(x)

plt.scatter(x,sort_word_dict.values())
plt.xscale("log")
plt.yscale("log")
plt.xlabel("単語の出現頻度順位")
plt.ylabel("出現頻度")
plt.title("Zipfの法則")
plt.savefig("39.png")

