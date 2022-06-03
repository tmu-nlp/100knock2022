#knock44 係り受け木の可視化
#与えられた文の係り受け木を有向グラフとして可視化せよ．
#可視化には，Graphviz等を用いるとよい.
import knock41
from graphviz import Digraph

fname = "ai.ja.txt.parsed"
text = knock41.parse_chunk(fname)
for i, sentence in enumerate(text[1:3]):#冒頭2文のグラフを出力
    dsts = []
    chunks = []
    for chunk in sentence:
        string = ""
        for morph in chunk.morphs:
            if morph.pos != "記号":
                string += morph.surface
        chunks.append(string)#文節テキスト保存
        dsts.append(chunk.dst)#係先の文節番号保存
    dg = Digraph(format="png")
    for j in range(len(chunks)):
        dg.node(chunks[j])
    for j in range(len(chunks)):
        if dsts[j] != -1:
            dg.edge(chunks[j],chunks[dsts[j]])
    dg.render(f"./output44_{i}")