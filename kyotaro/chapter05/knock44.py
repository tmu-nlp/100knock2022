"""
与えられた文の係り受け木を有向グラフとして可視化せよ. 可視化には, Graphviz等を用いるとよい
"""

import sys
from graphviz import Digraph
import pydot
from IPython.display import display_png, Image

from common import Morph
from common import Chunk
from common import Sentence
from common import set_matrioshk

file_name = sys.argv[1]
sentences, chunks, morphs = set_matrioshk(file_name)

edges = list()  # edgeの入るリストを設定
for i, chunk in enumerate(sentences[1].chunks):
    if chunk.dst != -1:
        modifer = "".join([morph.surface if morph.pos != "記号" else "" for morph in chunk.morphs] + [" ", str(i)])  # 係り元
        modifee = "".join([morph.surface if morph.pos != "記号" else "" for morph in sentences[1].chunks[int(chunk.dst)].morphs] + [" ", str(chunk.dst)])  # 係り先
        edges.append([modifer, modifee])  # edgeを設定

n = pydot.Node('node')
n.fontname = 'IPAGothic'  # 日本語に設定
g = pydot.graph_from_edges(edges, directed=True)  # 設定したedgeから追加するノードを設定
g.add_node(n)
g.write_png('44.png')
display_png(Image('44.png'))