"""
与えられた文の係り受け木を有向グラフとして可視化せよ. 可視化には, Graphviz等を用いるとよい
"""


import sys
import re
from collections import defaultdict
from graphviz import Digraph
import pydot
from IPython.display import display_png, Image

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

edges = list()
for chunk in sentences[1].chunks:
    if chunk.dst != -1:
        modifer = "".join([morph.surface if morph.pos != "記号" else "" for morph in chunk.morphs])  # 係り元
        modifee = "".join([morph.surface if morph.pos != "記号" else "" for morph in sentences[1].chunks[int(chunk.dst)].morphs])  # 係り先
        edges.append([modifer, modifee])  # edgeを設定

n = pydot.Node('node')
n.fontname = 'IPAGothic'  # 日本語に設定
g = pydot.graph_from_edges(edges, directed=True)  # 設定したedgeから追加するノードを設定
g.add_node(n)
g.write_png('44.png')
display_png(Image('44.png'))