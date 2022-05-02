# knock-34
# 名詞の連接（連続して出現する名詞）を最長一致で抽出せよ．

import knock30

for line in knock30.sentence_list:
    art_norm = ""
    cnt = 0
    for morph in line:
        if morph["pos"] == "名詞":
            art_norm += morph["surface"]
            cnt += 1
        else:
            if cnt > 1:
                print(art_norm)
            art_norm = ""
            cnt = 0    
    # sen = ""
    # for morph in line:
    #     sen += morph["surface"]
    # print(sen)