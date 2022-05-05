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

top_10 = count_dict_sort[:10] #上位10の単語とその頻度を抽出
top_10 = np.array(top_10) #列ごとに抽出したいからnumpy配列化
word_list = top_10[:, 0] #1列目(単語)を抽出
count_list = top_10[:, 1] #2列目(頻度)

plt.bar(word_list, count_list.astype(np.int64)) #頻度の方はint型に変換
plt.xlabel("単語")
plt.ylabel("出現回数")
plt.title("出現回数トップ10の単語")
plt.savefig("knock36_output")