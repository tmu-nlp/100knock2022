# knock-37
# 「猫」とよく共起する（共起頻度が高い）10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

import knock30
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import defaultdict #default-set
from collections import OrderedDict
cat_word_dict = defaultdict(lambda: 0)

# word_dictの計算時に"猫"と共起する条件を追加

cat_bool = [0] * len(knock30.sentence_list)

for i, line in enumerate(knock30.sentence_list):
    for morph in line:
        if morph["surface"] == "猫":
            cat_bool[i] = 1


for i, line in enumerate(knock30.sentence_list):
    for morph in line:
        if cat_bool[i] == 1 and morph["surface"] != "猫" and morph["pos"] != "記号":
            cat_word_dict[morph["surface"]] += 1

sort_word_dict = OrderedDict(sorted(cat_word_dict.items(), key = lambda x:x[1], reverse=True)[:10])

# print(sort_word_dict)

label = []
data = []
for k,v in sort_word_dict.items():
    label.append(k)
    data.append(v)
x = [1,2,3,4,5,6,7,8,9,10]
# print(label)
plt.bar(x,data)
plt.xticks(x,label)
plt.xlabel("語")
plt.ylabel("出現頻度")
plt.title("猫と共起頻度の高い上位10語")
plt.savefig("37.png")