# knock-31
# 動詞の表層形をすべて抽出せよ．

import knock30

for line in knock30.sentence_list:
    for morph in line:
        if morph['pos'] == "動詞":
            print(morph['surface'])
