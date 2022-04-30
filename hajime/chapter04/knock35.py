# knock-35
# 文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ．

import knock30
from collections import defaultdict #default-set
from collections import OrderedDict
word_dict = defaultdict(lambda: 0)

for line in knock30.sentence_list:
    for morph in line:
        if morph["pos"] == "記号":
            continue
        word_dict[morph["base"]] += 1

sort_word_dict = OrderedDict(sorted(word_dict.items(), key = lambda x:x[1], reverse=True))

# print(sort_word_dict)
for k,v in sort_word_dict.items():
    print(f"{k} : {v}")
