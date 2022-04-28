# knock-32
# 動詞の基本形をすべて抽出せよ．

import knock30

for line in knock30.sentence_list:
    for morph in line:
        if morph['pos'] == "動詞":
            print(morph['base'])
