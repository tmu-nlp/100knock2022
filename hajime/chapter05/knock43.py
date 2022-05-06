# knock43
# 名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．
# ただし，句読点などの記号は出力しないようにせよ．

import knock41

for sentence in knock41.sentences:
    for chunk in sentence.chunks:
        if chunk.dst != -1:
            modiin = []
            modifor = []
            normbool = 0
            verbbool = 0
            for morph in chunk.morphs:
                if morph.pos != "記号":
                    modiin.append(morph.surface)
                if morph.pos == "名詞":
                    normbool = 1
            for morph in sentence.chunks[chunk.dst].morphs:
                if morph.pos != "記号":
                    modifor.append(morph.surface)
                if morph.pos == "動詞":
                    verbbool = 1
            phrasein = ''.join(modiin)
            phraseout = ''.join(modifor)
            if normbool and verbbool:
                print(f"{phrasein}\t{phraseout}")
