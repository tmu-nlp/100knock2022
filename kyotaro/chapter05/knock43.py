"""
名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．ただし，句読点などの記号は出力しないようにせよ．
"""

import sys
import re
from collections import defaultdict

class Morph:
    def __init__(self, data):
        morph = data.replace("\t", ",").split(",")
        self.surface = morph[0]
        self.base = morph[7]
        self.pos = morph[1]
        self.pos1 = morph[2]

class Chunk:
    def __init__(self, morphs, dst):
        self.morphs = morphs
        self.dst = dst
        self.srcs = []
    
class Sentence:
    def __init__(self, chunks):
        self.chunks = chunks
        for i, chunk in enumerate(self.chunks):
            if chunk.dst != -1:
                self.chunks[chunk.dst].srcs.append(i)



sentences = []
chunks = []
morphs = []

with open(sys.argv[1], "r") as data:
    for line in data:
        if line[0] == "*":
            line = line.strip()
            if len(morphs) > 0:
                chunks.append(Chunk(morphs, dst))
                morphs = []
            dst = int(re.search(r'(.*?)D', line.split()[2]).group(1))

        elif line != "EOS\n":  # EOSが出ない場合
            morphs.append(Morph(line))

        else:
            if len(morphs) > 0:
                chunks.append(Chunk(morphs, dst))
                sentences.append(Sentence(chunks))
                morphs = []
                chunks = []
                dst = None
    
for chunk in sentences[1].chunks:
    if chunk.dst != -1:
        modifer = list()
        modifer_pos = list()
        modifee = list()
        modifee_pos = list()
        for morph in chunk.morphs:
            if morph.pos != "記号":
                modifer_pos.append(morph.pos)
                if "名詞" in modifer_pos:
                    modifer.append(morph.surface)
        
        for morph in sentences[1].chunks[int(chunk.src)].morphs:
            if morph.pos != "記号":
                modifee_pos.append(morph.pos)
                if "動詞" in modifee_pos:
                    modifee.append(morph.surface)
        
        if modifer and modifee:
            print("".join(modifer) + "\t" + "".join(modifee))
