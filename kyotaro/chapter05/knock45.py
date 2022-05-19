"""問題文長いので省略"""

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

ans = defaultdict(lambda: 0)
    
for sentence in sentences:
    for chunk in sentence.chunks:
        modifer = list()
        modifee = list()
        for morph in chunk.morphs:
            if morph.pos == "動詞":
                modifer.append(morph.base)
        
        for morph in sentence.chunks[int(chunk.dst)].morphs:
                if morph.pos == "助詞":
                    modifee.append(morph.surface)
        
        if modifer and modifee:
            fer = "".join(modifer)
            fee = "".join(modifee)
            ans[f'{fer} {fee}'] += 1

for key, value in sorted(ans.items(), key = lambda x:x[1], reverse = True):
    print(key + "\t" + str(value))
