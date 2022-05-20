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
frec_list = count_np_sort[:, 1]
lank_list = np.arange(1, frec_list.shape[0] + 1) #出現順位(frec_listに対応する）

plt.scatter(lank_list, frec_list.astype(np.int64))
plt.xscale('log')
plt.yscale('log')

plt.xlabel("log 出現頻度順位")
plt.ylabel("log 出現頻度")
plt.title('Zipfの法則')
plt.savefig("./output/knock39_output")