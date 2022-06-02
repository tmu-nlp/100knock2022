"""
名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．ただし，句読点などの記号は出力しないようにせよ．
"""

import sys

from common import Morph
from common import Chunk
from common import Sentence
from common import set_matrioshk

file_name = sys.argv[1]
sentences, chunks, morphs = set_matrioshk(file_name)
    
for chunk in sentences[1].chunks:
    if chunk.dst != -1:
        modifer = list()  # 係り元のChunk用
        modifer_pos = list()  # 係り元の各Morphの品詞を格納
        modifee = list()  # 係り先の各Morphのsurface
        modifee_pos = list()  # 係り先の各Morphの品詞を格納
        for morph in chunk.morphs:
            if morph.pos != "記号": 
                modifer_pos.append(morph.pos)  # 係り元の品詞を格納
                if "名詞" in modifer_pos:  # 名詞を含むか判別
                    modifer.append(morph.surface)  # 名詞を含んでいれば係り元のChunkに格納
        
        for morph in sentences[1].chunks[int(chunk.dst)].morphs:
            if morph.pos != "記号":
                modifee_pos.append(morph.pos)  # 係り先の品詞を格納
                if "動詞" in modifee_pos:  # 動詞を含むか判別
                    modifee.append(morph.surface)  # 動詞を含んでいれば係り先のChunkに格納
        
        if modifer and modifee:  # 両方が存在していれば出力
            print("".join(modifer) + "\t" + "".join(modifee))
