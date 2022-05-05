import knock30
import japanize_matplotlib #pltで日本語を使うため
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

cat_dict = defaultdict(lambda: 0)
in_cat = False #文に猫があるかのブール

for line in knock30.morpheme_line_list:
    for morpheme in line:
        if (morpheme["基本形"]) == "猫":
            in_cat = True

    if in_cat:       
        for morpheme in line:
            if morpheme["品詞"] != "記号":
                cat_dict[morpheme['基本形']] += 1
    in_cat = False

del cat_dict['猫']
cat_dict_sort = sorted(cat_dict.items(), key=lambda x:x[1], reverse=True)
top_10_cat = np.array(cat_dict_sort[:10])
word_list = top_10_cat[:, 0]
count_list = top_10_cat[:, 1]

plt.bar(word_list, count_list.astype(np.int64)) #頻度の方はint型に変換
plt.xlabel("単語")
plt.ylabel("出現回数")
plt.title("猫との共起頻度トップ10の単語")
plt.savefig("knock37_output")