"""
係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．ただし，句読点などの記号は出力しないようにせよ．
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
        modifer = "".join([morph.surface if morph.pos != "記号" else "" for morph in chunk.morphs])  # 係り元
        modifee = "".join([morph.surface if morph.pos != "記号" else "" for morph in sentences[1].chunks[int(chunk.dst)].morphs])  # 係り先
        print(modifer, "\t", modifee)  # タブ区切りで出力