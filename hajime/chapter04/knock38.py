# knock-38
# 単語の出現頻度のヒストグラムを描け．
# ただし，横軸は出現頻度を表し，1から単語の出現頻度の最大値までの線形目盛とする．
# 縦軸はx軸で示される出現頻度となった単語の異なり数（種類数）である．

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

freq = list(word_dict.values())

plt.hist(freq,bins=100)
plt.xlabel("単語の異なり度数")
plt.ylabel("出現頻度")
plt.title("単語の出現頻度のヒストグラム")
plt.savefig("38.png")

