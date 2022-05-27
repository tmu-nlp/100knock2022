"""
今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい． 
動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ． 
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
                mod = []  # 動詞が出てきたらリストの用意
                noumod = []
                mod.append(morph.base)  # 動詞をリストに格納する
                for i in chunk.srcs:
                    noumod.append(" ")  # 区切りを追加
                    for morph in sentence.chunks[i].morphs:  # 見る形態素を係り元の方に変更
                        if morph.pos == "助詞":
                            mod.append(morph.surface)  # 係り元に助詞が含まれるならその助詞をリストに追加
                    if len(mod) > 1:  # 動詞以外が入っていたら
                        for morph in sentence.chunks[i].morphs:
                            if morph.pos != "記号":
                                noumod.append(morph.surface)

                            
                if mod and noumod:
                    verb = mod[0]
                    modifee = " ".join(mod[1::])  # 係り元をスペース区切りで連結
                    noumod = "".join(noumod)  # 項が入っているリストをstrに変換
                    
                    print(verb + "\t" + modifee + " " + noumod)  # 動詞の部分 mod[0] と係り元の modifee と項の noumodをタブ区切りで連結
                    break  # 左端の部分だけで実行