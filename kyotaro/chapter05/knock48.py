"""
文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ． ただし，構文木上のパスは以下の仕様を満たすものとする．
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
        noun = list()  # 注目する名詞句
        noun_pos = list()  # 名詞句かどうか判定するためのリスト
        modifee = list()  # 最初が名詞句だった場合、その後につなげるリスト
        for morph in chunk.morphs:
            noun_pos.append(morph.pos)  # Chunk単位で品詞を追加
        if "名詞" in noun_pos:  # Chunkの中のMorphに名詞があるか判定
            for morph in chunk.morphs:
                if morph.pos != "記号":
                    noun.append(morph.surface)  # 名詞句かつ記号以外のMorphを注目する名詞句として追加
                    while chunk.dst != -1:
                        modifee.append("".join(morph.surface for morph in sentence.chunks[chunk.dst].morphs if morph.pos != "記号"))  # 終了文節まで係り先の番号のChunkを追加
                        chunk = sentence.chunks[chunk.dst]  # Chunkの更新
        
        if noun and modifee:  # 名詞句かつその後につながるものがあった場合
            noun = "".join(noun)
            modifee = " -> ".join(modifee)
            print(noun + " -> " + modifee)

