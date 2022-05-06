# knock44
# 与えられた文の係り受け木を有向グラフとして可視化せよ．可視化には，Graphviz等を用いるとよい．
# brew install graphviz
# pip3 install graphviz

import knock41
from graphviz import Digraph

num = 0

for sentence in knock41.sentences:
    dg = Digraph(format='png')
    for chunk in sentence.chunks:
        if chunk.dst != -1:
            modiin = []
            modifor = []
            for morph in chunk.morphs:
                if morph.pos != "記号":
                    modiin.append(morph.surface)
            for morph in sentence.chunks[chunk.dst].morphs:
                if morph.pos != "記号":
                    modifor.append(morph.surface)
            phrasein = ''.join(modiin)
            phraseout = ''.join(modifor)
            dg.edge(phrasein, phraseout)
            # print(f"{phrasein}\t{phraseout}")
    dg.render('./44/' + str(num))
    num += 1
