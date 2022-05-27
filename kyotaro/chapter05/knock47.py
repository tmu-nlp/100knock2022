"""
動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい. 46のプログラムを以下の仕様を満たすように改変せよ
"""

import sys
from common import Morph
from common import Chunk
from common import Sentence
from common import set_matrioshk

file_name = sys.argv[1]
sentences, chunks, morphs = set_matrioshk(file_name)
    

for sentence in sentences:
    for chunk in sentence.chunks:
        for morph in chunk.morphs:
            if morph.pos == "動詞":
                noumod = list()
                mod = list()
                for s, i in enumerate(chunk.srcs):
                    if len(sentence.chunks[i].morphs) == 2 and sentence.chunks[i].morphs[0].pos1 == "サ変接続" and sentence.chunks[i].morphs[1].surface == "を":  # 「サ変接続＋を」となっているか判定する
                        verb = sentence.chunks[i].morphs[0].surface + sentence.chunks[i].morphs[1].surface + morph.base  # 条件を満たすならverbに追加
                        for j in chunk.srcs:  # 追加で助詞を追加
                            noumod.append(" ")
                            for morph_a in sentence.chunks[j].morphs:
                                if morph_a.pos == "助詞" and i != j:  # 品詞が助詞であるかつ、上の条件を満たすようなものではないかを判定
                                    mod.append(morph_a.surface)  # 助詞を追加
                            if mod:
                                for morph_b in sentence.chunks[j].morphs:
                                    if morph_b.pos != "記号" and i != j:
                                        noumod.append(morph_b.surface)

                        if verb and mod:
                            mod = " ".join(mod)
                            noumod = "".join(noumod)
                            print(verb + "\t" + mod + "\t" + noumod)
                        break